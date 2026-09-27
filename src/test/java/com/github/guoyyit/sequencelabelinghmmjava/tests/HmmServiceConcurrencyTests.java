package com.github.guoyyit.sequencelabelinghmmjava.tests;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.guoyyit.sequencelabelinghmmjava.api.dto.DecodeRequest;
import com.github.guoyyit.sequencelabelinghmmjava.api.dto.DecodeResponse;
import com.github.guoyyit.sequencelabelinghmmjava.api.dto.TrainRequest;
import com.github.guoyyit.sequencelabelinghmmjava.api.error.ApiException;
import com.github.guoyyit.sequencelabelinghmmjava.api.error.ErrorCode;
import com.github.guoyyit.sequencelabelinghmmjava.config.AppProperties;
import com.github.guoyyit.sequencelabelinghmmjava.hmm.TrainSample;
import com.github.guoyyit.sequencelabelinghmmjava.model.ModelRegistry;
import com.github.guoyyit.sequencelabelinghmmjava.model.ModelStore;
import com.github.guoyyit.sequencelabelinghmmjava.service.HmmService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Queue;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Concurrent training, decoding and version switching stay self-consistent
 * (requirement 4).
 */
class HmmServiceConcurrencyTests {

    @TempDir
    Path dataDir;

    private HmmService newService() {
        AppProperties props = new AppProperties();
        props.setDataDir(dataDir.toString());
        return new HmmService(new ModelRegistry(), new ModelStore(props, new ObjectMapper()));
    }

    private static TrainRequest trainRequest(long seed) {
        return new TrainRequest(2, 5, 1e-4, seed, List.of("N", "V"),
                List.of(
                        new TrainSample(List.of("dogs", "bark"), List.of("N", "V")),
                        new TrainSample(List.of("cats", "sleep"), List.of("N", "V")),
                        new TrainSample(List.of("birds", "fly"), List.of("N", "V"))));
    }

    @Test
    void decodeWithoutActiveModelIsRejected() {
        HmmService service = newService();
        ApiException ex = assertThrows(ApiException.class,
                () -> service.decode(new DecodeRequest(null, List.of("dogs"))));
        assertEquals(ErrorCode.NO_ACTIVE_MODEL, ex.code());
    }

    @Test
    void concurrentTrainingAndDecodingStaysConsistent() throws InterruptedException {
        HmmService service = newService();
        long v1 = service.train(trainRequest(1L)).version();

        Queue<Throwable> failures = new ConcurrentLinkedQueue<>();
        Set<Long> seenVersions = ConcurrentHashMap.newKeySet();
        AtomicBoolean stop = new AtomicBoolean(false);
        int decoderThreads = 6;
        int trainerThreads = 3;
        ExecutorService pool = Executors.newFixedThreadPool(decoderThreads + trainerThreads);
        CountDownLatch start = new CountDownLatch(1);
        CountDownLatch done = new CountDownLatch(decoderThreads + trainerThreads);

        // Reference result for the fixed historical version v1.
        DecodeResponse reference = service.decode(new DecodeRequest(v1, List.of("dogs", "bark")));

        for (int i = 0; i < decoderThreads; i++) {
            pool.submit(() -> {
                try {
                    start.await();
                    while (!stop.get()) {
                        // Historical version query: must always return identical results.
                        DecodeResponse historical =
                                service.decode(new DecodeRequest(v1, List.of("dogs", "bark")));
                        if (historical.modelVersion() != v1
                                || historical.logLikelihood() != reference.logLikelihood()
                                || !historical.bestPath().equals(reference.bestPath())) {
                            failures.add(new AssertionError("historical version v" + v1 + " changed under concurrency"));
                        }
                        // Active-version decode: must return some committed version.
                        DecodeResponse active =
                                service.decode(new DecodeRequest(null, List.of("cats", "sleep")));
                        seenVersions.add(active.modelVersion());
                    }
                } catch (Throwable t) {
                    failures.add(t);
                } finally {
                    done.countDown();
                }
            });
        }
        for (int i = 0; i < trainerThreads; i++) {
            final long seed = 100L + i;
            pool.submit(() -> {
                try {
                    start.await();
                    for (int k = 0; k < 5; k++) {
                        service.train(trainRequest(seed + k));
                    }
                } catch (Throwable t) {
                    failures.add(t);
                } finally {
                    done.countDown();
                }
            });
        }

        start.countDown();
        Thread.sleep(500);
        stop.set(true);
        assertTrue(done.await(60, TimeUnit.SECONDS), "workers did not finish");
        pool.shutdownNow();

        assertTrue(failures.isEmpty(), () -> "concurrent failures: " + failures);
        // Every version id ever observed must be a committed version.
        Set<Long> committed = ConcurrentHashMap.newKeySet();
        service.listVersions().forEach(v -> committed.add(v.version()));
        assertTrue(committed.containsAll(seenVersions),
                "decoders observed uncommitted versions: " + seenVersions + " vs " + committed);
        assertEquals(1 + trainerThreads * 5, service.listVersions().size());
    }

    @Test
    void concurrentActivationNeverMixesVersions() throws InterruptedException {
        HmmService service = newService();
        long v1 = service.train(trainRequest(1L)).version();
        long v2 = service.train(trainRequest(2L)).version();

        Queue<Throwable> failures = new ConcurrentLinkedQueue<>();
        AtomicBoolean stop = new AtomicBoolean(false);
        int decoderThreads = 6;
        ExecutorService pool = Executors.newFixedThreadPool(decoderThreads + 1);
        CountDownLatch start = new CountDownLatch(1);
        CountDownLatch done = new CountDownLatch(decoderThreads + 1);

        for (int i = 0; i < decoderThreads; i++) {
            pool.submit(() -> {
                try {
                    start.await();
                    while (!stop.get()) {
                        DecodeResponse r = service.decode(new DecodeRequest(null, List.of("dogs", "bark")));
                        if (r.modelVersion() != v1 && r.modelVersion() != v2) {
                            failures.add(new AssertionError("decode used unexpected version " + r.modelVersion()));
                        }
                    }
                } catch (Throwable t) {
                    failures.add(t);
                } finally {
                    done.countDown();
                }
            });
        }
        pool.submit(() -> {
            try {
                start.await();
                while (!stop.get()) {
                    service.activate(v1);
                    service.activate(v2);
                }
            } catch (Throwable t) {
                failures.add(t);
            } finally {
                done.countDown();
            }
        });

        start.countDown();
        Thread.sleep(500);
        stop.set(true);
        assertTrue(done.await(60, TimeUnit.SECONDS), "workers did not finish");
        pool.shutdownNow();
        assertTrue(failures.isEmpty(), () -> "concurrent failures: " + failures);
    }
}
