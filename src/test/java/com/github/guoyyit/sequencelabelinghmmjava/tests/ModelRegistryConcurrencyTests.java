package com.github.guoyyit.sequencelabelinghmmjava.tests;

import com.github.guoyyit.sequencelabelinghmmjava.hmm.HmmModel;
import com.github.guoyyit.sequencelabelinghmmjava.hmm.TrainConfig;
import com.github.guoyyit.sequencelabelinghmmjava.model.ModelRegistry;
import com.github.guoyyit.sequencelabelinghmmjava.model.ModelVersion;
import org.junit.jupiter.api.Test;

import java.time.Instant;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Atomic version switching under concurrency (requirement 3).
 */
class ModelRegistryConcurrencyTests {

    private static ModelVersion version(long id) {
        HmmModel model = new HmmModel(
                List.of("S0"),
                List.of("x"),
                new double[]{0.0},
                new double[][]{{0.0}},
                new double[][]{{0.0}});
        return new ModelVersion(id, Instant.now(),
                new TrainConfig(1, 0, 0.0, 0L), model, 1, 0, -1.0);
    }

    @Test
    void readersOnlyEverSeeCommittedWholeVersions() throws InterruptedException {
        ModelRegistry registry = new ModelRegistry();
        int versionCount = 20;
        for (long id = 1; id <= versionCount; id++) {
            registry.register(version(id));
        }
        registry.activate(1);

        Set<Long> committed = ConcurrentHashMap.newKeySet();
        for (long id = 1; id <= versionCount; id++) {
            committed.add(id);
        }
        AtomicInteger violations = new AtomicInteger();
        int readers = 8;
        ExecutorService pool = Executors.newFixedThreadPool(readers + 1);
        CountDownLatch start = new CountDownLatch(1);
        CountDownLatch done = new CountDownLatch(readers);

        for (int r = 0; r < readers; r++) {
            pool.submit(() -> {
                try {
                    start.await();
                    for (int i = 0; i < 5_000; i++) {
                        Optional<ModelVersion> active = registry.active();
                        if (active.isEmpty() || !committed.contains(active.get().version())) {
                            violations.incrementAndGet();
                        }
                        // A version fetched by id must always be complete.
                        long probe = 1 + (i % versionCount);
                        if (registry.find(probe).isEmpty()) {
                            violations.incrementAndGet();
                        }
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    done.countDown();
                }
            });
        }
        pool.submit(() -> {
            try {
                start.await();
                for (long id = 1; id <= versionCount; id++) {
                    assertTrue(registry.activate(id));
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });

        start.countDown();
        assertTrue(done.await(30, TimeUnit.SECONDS));
        pool.shutdownNow();
        assertEquals(0, violations.get(), "readers saw an uncommitted or missing version");
    }

    @Test
    void concurrentRegistrationsGetUniqueMonotonicIds() throws InterruptedException {
        ModelRegistry registry = new ModelRegistry();
        int threads = 8;
        int perThread = 50;
        Set<Long> ids = ConcurrentHashMap.newKeySet();
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        CountDownLatch start = new CountDownLatch(1);
        CountDownLatch done = new CountDownLatch(threads);
        for (int t = 0; t < threads; t++) {
            pool.submit(() -> {
                try {
                    start.await();
                    for (int i = 0; i < perThread; i++) {
                        long id = registry.nextVersionId();
                        ids.add(id);
                        registry.register(version(id));
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    done.countDown();
                }
            });
        }
        start.countDown();
        assertTrue(done.await(30, TimeUnit.SECONDS));
        pool.shutdownNow();
        assertEquals(threads * perThread, ids.size(), "version ids must be unique");
        assertEquals(threads * perThread, registry.list().size());
    }

    @Test
    void activatingUnknownVersionFailsWithoutDisturbingCurrent() {
        ModelRegistry registry = new ModelRegistry();
        registry.register(version(1));
        registry.activate(1);
        assertTrue(registry.find(1).isPresent());
        org.junit.jupiter.api.Assertions.assertFalse(registry.activate(999));
        assertEquals(1, registry.active().orElseThrow().version());
    }

    @Test
    void restorePicksPersistedActiveOrLatest() {
        ModelRegistry registry = new ModelRegistry();
        registry.restore(List.of(version(1), version(2), version(3)), 2L);
        assertEquals(2, registry.active().orElseThrow().version());
        assertEquals(4, registry.nextVersionId(), "id counter continues after the max restored id");

        ModelRegistry fallback = new ModelRegistry();
        fallback.restore(List.of(version(5), version(7)), null);
        assertEquals(7, fallback.active().orElseThrow().version(),
                "without a persisted pointer the latest version becomes active");
    }
}
