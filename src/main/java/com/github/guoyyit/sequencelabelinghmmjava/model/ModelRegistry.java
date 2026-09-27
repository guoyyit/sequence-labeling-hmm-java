package com.github.guoyyit.sequencelabelinghmmjava.model;

import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/**
 * In-memory versioned model registry with atomic active-version switching.
 *
 * <p>Readers take the active {@link AtomicReference} once per request, so a
 * switch can never produce a request that mixes parameters from two versions:
 * it either sees the whole old version or the whole new one. Committed
 * versions are immutable and stay readable forever.</p>
 *
 * <p>Mutations are serialized through {@code commitLock}; reads are lock-free.</p>
 */
@Component
public class ModelRegistry {

    private final ConcurrentHashMap<Long, ModelVersion> versions = new ConcurrentHashMap<>();
    private final AtomicReference<ModelVersion> activeRef = new AtomicReference<>();
    private final AtomicLong versionCounter = new AtomicLong(0);
    private final Object commitLock = new Object();

    /**
     * Commits a fully-built version. The version becomes visible to readers
     * only after this call returns, never in a partially initialized state.
     */
    public ModelVersion register(ModelVersion version) {
        synchronized (commitLock) {
            ModelVersion previous = versions.putIfAbsent(version.version(), version);
            if (previous != null) {
                throw new IllegalStateException("version already registered: " + version.version());
            }
            return version;
        }
    }

    /** The currently active version, if any. The reference is obtained atomically. */
    public Optional<ModelVersion> active() {
        return Optional.ofNullable(activeRef.get());
    }

    public Optional<ModelVersion> find(long version) {
        return Optional.ofNullable(versions.get(version));
    }

    /** All committed versions, ordered by version id. */
    public List<ModelVersion> list() {
        List<ModelVersion> all = new ArrayList<>(versions.values());
        all.sort(Comparator.comparingLong(ModelVersion::version));
        return List.copyOf(all);
    }

    /**
     * Atomically switches the active pointer to an already-committed version.
     *
     * @return {@code true} if the version exists and the switch happened
     */
    public boolean activate(long version) {
        synchronized (commitLock) {
            ModelVersion target = versions.get(version);
            if (target == null) {
                return false;
            }
            activeRef.set(target);
            return true;
        }
    }

    /** Allocates the next version id. Ids are monotonically increasing from 1. */
    public long nextVersionId() {
        return versionCounter.incrementAndGet();
    }

    /**
     * Bulk-restores persisted state on startup. Must be called before the
     * registry serves traffic; existing in-memory content is replaced.
     */
    public void restore(List<ModelVersion> restored, Long activeVersion) {
        synchronized (commitLock) {
            versions.clear();
            activeRef.set(null);
            long maxId = 0;
            for (ModelVersion v : restored) {
                versions.put(v.version(), v);
                maxId = Math.max(maxId, v.version());
            }
            versionCounter.set(maxId);
            if (activeVersion != null) {
                ModelVersion target = versions.get(activeVersion);
                if (target != null) {
                    activeRef.set(target);
                }
            }
            if (activeRef.get() == null && maxId > 0) {
                activeRef.set(versions.get(maxId));
            }
        }
    }
}
