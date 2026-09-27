package com.github.guoyyit.sequencelabelinghmmjava.registry;

import com.github.guoyyit.sequencelabelinghmmjava.domain.HmmParameters;
import com.github.guoyyit.sequencelabelinghmmjava.domain.ModelVersion;
import com.github.guoyyit.sequencelabelinghmmjava.domain.TrainingConfig;
import com.github.guoyyit.sequencelabelinghmmjava.domain.TrainingMetrics;
import com.github.guoyyit.sequencelabelinghmmjava.error.ErrorCode;
import com.github.guoyyit.sequencelabelinghmmjava.error.ServiceException;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Thread-safe registry of committed model versions.
 *
 * <p>Versions are immutable once committed. The active version pointer is an
 * {@link AtomicReference}; switching swaps it in one step, so a reader always
 * sees one complete committed version and never a mix of old and new
 * parameters. Commits and activations are serialized through a lock and
 * persisted (version file first, then the index), so registered versions
 * survive restarts.</p>
 */
public class ModelRegistry {

    private final ModelStore store;
    private final ConcurrentHashMap<String, ModelVersion> versions = new ConcurrentHashMap<>();
    private final CopyOnWriteArrayList<String> versionOrder = new CopyOnWriteArrayList<>();
    private final AtomicReference<String> activeId = new AtomicReference<>();
    private final ReentrantLock commitLock = new ReentrantLock();
    private int counter;

    public ModelRegistry(ModelStore store) {
        this.store = store;
        restore();
    }

    private void restore() {
        Optional<ModelStore.RegistryState> state = store.loadState();
        if (state.isPresent()) {
            for (String id : state.get().versionIds()) {
                ModelVersion version = store.loadVersion(id).orElseThrow(() ->
                        new ServiceException(ErrorCode.PERSISTENCE_FAILURE,
                                "registry index references missing version " + id));
                registerLoaded(version);
            }
            String active = state.get().activeVersionId();
            if (active != null) {
                if (!versions.containsKey(active)) {
                    throw new ServiceException(ErrorCode.PERSISTENCE_FAILURE,
                            "registry index references missing active version " + active);
                }
                activeId.set(active);
            }
        } else {
            for (ModelVersion version : store.loadAllVersions()) {
                registerLoaded(version);
            }
        }
    }

    private void registerLoaded(ModelVersion version) {
        versions.put(version.versionId(), version);
        versionOrder.add(version.versionId());
        counter = Math.max(counter, numericSuffix(version.versionId()));
    }

    private static int numericSuffix(String versionId) {
        String digits = versionId.replaceAll("[^0-9]", "");
        return digits.isEmpty() ? 0 : Integer.parseInt(digits);
    }

    /** Commits a new immutable version and persists it. Does not activate it. */
    public ModelVersion commit(HmmParameters parameters, TrainingConfig config, TrainingMetrics metrics) {
        commitLock.lock();
        try {
            String id = "v" + (++counter);
            ModelVersion version = new ModelVersion(id, Instant.now(), config, parameters, metrics);
            store.saveVersion(version);
            versions.put(id, version);
            versionOrder.add(id);
            persistState();
            return version;
        } finally {
            commitLock.unlock();
        }
    }

    /** Atomically switches the active pointer to a committed version. */
    public ModelVersion activate(String versionId) {
        ModelVersion version = versions.get(versionId);
        if (version == null) {
            throw new ServiceException(ErrorCode.MODEL_VERSION_NOT_FOUND,
                    "unknown model version: " + versionId);
        }
        commitLock.lock();
        try {
            persistStateWithActive(versionId);
            activeId.set(versionId);
            return version;
        } finally {
            commitLock.unlock();
        }
    }

    /** @return the current active version, empty when none was activated. */
    public Optional<ModelVersion> active() {
        String id = activeId.get();
        return id == null ? Optional.empty() : Optional.ofNullable(versions.get(id));
    }

    public Optional<ModelVersion> find(String versionId) {
        return Optional.ofNullable(versions.get(versionId));
    }

    public List<ModelVersion> list() {
        List<ModelVersion> result = new ArrayList<>(versionOrder.size());
        for (String id : versionOrder) {
            ModelVersion version = versions.get(id);
            if (version != null) {
                result.add(version);
            }
        }
        return result;
    }

    private void persistState() {
        store.saveState(new ModelStore.RegistryState(List.copyOf(versionOrder), activeId.get()));
    }

    private void persistStateWithActive(String newActiveId) {
        store.saveState(new ModelStore.RegistryState(List.copyOf(versionOrder), newActiveId));
    }
}
