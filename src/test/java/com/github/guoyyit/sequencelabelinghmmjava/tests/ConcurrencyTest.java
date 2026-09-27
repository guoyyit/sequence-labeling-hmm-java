package com.github.guoyyit.sequencelabelinghmmjava.tests;

import com.github.guoyyit.sequencelabelinghmmjava.domain.DecodeResult;
import com.github.guoyyit.sequencelabelinghmmjava.domain.LabeledSample;
import com.github.guoyyit.sequencelabelinghmmjava.domain.ModelVersion;
import com.github.guoyyit.sequencelabelinghmmjava.domain.TrainingConfig;
import com.github.guoyyit.sequencelabelinghmmjava.registry.ModelRegistry;
import com.github.guoyyit.sequencelabelinghmmjava.registry.ModelStore;
import com.github.guoyyit.sequencelabelinghmmjava.service.SequenceLabelingService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Concurrent decoding during atomic version switches and concurrent training.
 */
class ConcurrencyTest {

    @TempDir
    Path dataDir;

    private SequenceLabelingService newService() {
        return new SequenceLabelingService(new ModelRegistry(new ModelStore(dataDir)));
    }

    private static List<LabeledSample> corpusA() {
        return List.of(
                new LabeledSample(List.of("x", "y", "x"), List.of("S1", "S2", "S1")),
                new LabeledSample(List.of("y", "x", "y"), List.of("S2", "S1", "S2")),
                new LabeledSample(List.of("x", "x", "y"), List.of("S1", "S1", "S2")));
    }

    private static List<LabeledSample> corpusB() {
        return List.of(
                new LabeledSample(List.of("x", "y", "x"), List.of("S2", "S1", "S2")),
                new LabeledSample(List.of("y", "y", "x"), List.of("S2", "S2", "S1")),
                new LabeledSample(List.of("y", "x", "x"), List.of("S2", "S1", "S1")));
    }

    @Test
    void decodeNeverSeesMixedParametersDuringSwitch() throws Exception {
        SequenceLabelingService service = newService();
        ModelVersion v1 = service.train(corpusA(), TrainingConfig.defaults());
        ModelVersion v2 = service.train(corpusB(), TrainingConfig.defaults());
        service.activate(v1.versionId());

        List<String> probe = List.of("x", "y", "x", "y", "x");
        DecodeResult expectV1 = service.decode(probe, v1.versionId());
        DecodeResult expectV2 = service.decode(probe, v2.versionId());
        // The corpora disagree on labels, so the two versions must decode differently.
        assertNotEquals(expectV1.path(), expectV2.path());

        int readers = 8;
        int switches = 300;
        ExecutorService pool = Executors.newFixedThreadPool(readers + 1);
        CountDownLatch start = new CountDownLatch(1);
        java.util.concurrent.atomic.AtomicBoolean running = new java.util.concurrent.atomic.AtomicBoolean(true);
        ConcurrentLinkedQueue<Throwable> errors = new ConcurrentLinkedQueue<>();
        List<Future<?>> futures = new ArrayList<>();

        for (int r = 0; r < readers; r++) {
            futures.add(pool.submit(() -> {
                try {
                    start.await();
                    while (running.get()) {
                        DecodeResult result = service.decode(probe, null);
                        boolean matchesV1 = result.path().equals(expectV1.path())
                                && result.logProbability() == expectV1.logProbability()
                                && result.versionId().equals(v1.versionId());
                        boolean matchesV2 = result.path().equals(expectV2.path())
                                && result.logProbability() == expectV2.logProbability()
                                && result.versionId().equals(v2.versionId());
                        if (!matchesV1 && !matchesV2) {
                            fail("decode observed a mix of two versions: " + result);
                        }
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } catch (Throwable t) {
                    errors.add(t);
                }
            }));
        }
        futures.add(pool.submit(() -> {
            try {
                start.await();
                for (int i = 0; i < switches; i++) {
                    service.activate(i % 2 == 0 ? v2.versionId() : v1.versionId());
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            } finally {
                running.set(false);
            }
        }));

        start.countDown();
        pool.shutdown();
        assertTrue(pool.awaitTermination(60, TimeUnit.SECONDS));
        if (!errors.isEmpty()) {
            fail("reader thread failed: " + errors.peek());
        }
    }

    private static void assertNotEquals(Object a, Object b) {
        org.junit.jupiter.api.Assertions.assertNotEquals(a, b);
    }

    @Test
    void concurrentTrainingProducesDistinctConsistentVersions() throws Exception {
        SequenceLabelingService service = newService();
        int trainers = 6;
        ExecutorService pool = Executors.newFixedThreadPool(trainers);
        CountDownLatch start = new CountDownLatch(1);
        List<Future<ModelVersion>> futures = new ArrayList<>();
        for (int i = 0; i < trainers; i++) {
            int seed = i;
            futures.add(pool.submit(() -> {
                start.await();
                return service.train(corpusA(),
                        new TrainingConfig(4, 5, 1e-6, seed, 0.01));
            }));
        }
        start.countDown();
        pool.shutdown();
        assertTrue(pool.awaitTermination(60, TimeUnit.SECONDS));

        Set<String> ids = new HashSet<>();
        for (Future<ModelVersion> future : futures) {
            ids.add(future.get().versionId());
        }
        assertEquals(trainers, ids.size(), "every training run must get its own version id");
        assertEquals(trainers, service.versions().size());

        // Every committed version decodes consistently with itself.
        List<String> probe = List.of("x", "y");
        for (ModelVersion version : service.versions()) {
            DecodeResult a = service.decode(probe, version.versionId());
            DecodeResult b = service.decode(probe, version.versionId());
            assertEquals(a.path(), b.path());
            assertEquals(a.logProbability(), b.logProbability(), 0.0);
        }
    }

    @Test
    void trainingAndHistoryQueriesAreConsistentUnderConcurrency() throws Exception {
        SequenceLabelingService service = newService();
        ModelVersion first = service.train(corpusA(), TrainingConfig.defaults());
        service.activate(first.versionId());

        ExecutorService pool = Executors.newFixedThreadPool(4);
        CountDownLatch start = new CountDownLatch(1);
        ConcurrentLinkedQueue<Throwable> errors = new ConcurrentLinkedQueue<>();
        List<Future<?>> futures = new ArrayList<>();
        // One trainer keeps committing new versions.
        futures.add(pool.submit(() -> {
            try {
                start.await();
                for (int i = 0; i < 10; i++) {
                    service.train(corpusB(), new TrainingConfig(null, 3, 1e-6, i, 0.01));
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }));
        // Readers keep listing versions and decoding the active one.
        for (int r = 0; r < 3; r++) {
            futures.add(pool.submit(() -> {
                try {
                    start.await();
                    for (int i = 0; i < 200; i++) {
                        List<ModelVersion> versions = service.versions();
                        Set<String> ids = new HashSet<>();
                        for (ModelVersion v : versions) {
                            ids.add(v.versionId());
                        }
                        assertEquals(ids.size(), versions.size(), "duplicate version ids");
                        DecodeResult result = service.decode(List.of("x", "y"), null);
                        assertTrue(ids.contains(result.versionId()),
                                "decode used a version not visible in the registry");
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } catch (Throwable t) {
                    errors.add(t);
                }
            }));
        }
        start.countDown();
        pool.shutdown();
        assertTrue(pool.awaitTermination(60, TimeUnit.SECONDS));
        if (!errors.isEmpty()) {
            fail("concurrent reader failed: " + errors.peek());
        }
        assertEquals(11, service.versions().size());
    }
}
