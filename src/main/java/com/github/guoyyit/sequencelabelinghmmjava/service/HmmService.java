package com.github.guoyyit.sequencelabelinghmmjava.service;

import com.github.guoyyit.sequencelabelinghmmjava.api.dto.DecodeRequest;
import com.github.guoyyit.sequencelabelinghmmjava.api.dto.DecodeResponse;
import com.github.guoyyit.sequencelabelinghmmjava.api.dto.ModelVersionView;
import com.github.guoyyit.sequencelabelinghmmjava.api.dto.TrainRequest;
import com.github.guoyyit.sequencelabelinghmmjava.api.error.ApiException;
import com.github.guoyyit.sequencelabelinghmmjava.api.error.ErrorCode;
import com.github.guoyyit.sequencelabelinghmmjava.hmm.DecodeResult;
import com.github.guoyyit.sequencelabelinghmmjava.hmm.HmmDecoder;
import com.github.guoyyit.sequencelabelinghmmjava.hmm.HmmModel;
import com.github.guoyyit.sequencelabelinghmmjava.hmm.HmmTrainer;
import com.github.guoyyit.sequencelabelinghmmjava.hmm.TrainConfig;
import com.github.guoyyit.sequencelabelinghmmjava.hmm.TrainSample;
import com.github.guoyyit.sequencelabelinghmmjava.model.ModelRegistry;
import com.github.guoyyit.sequencelabelinghmmjava.model.ModelStore;
import com.github.guoyyit.sequencelabelinghmmjava.model.ModelVersion;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

/**
 * Application service orchestrating training, registration, decoding and
 * activation.
 *
 * <p>Validation runs completely before any training starts, so an invalid
 * sample can never leave a partial update behind. Commits (train/activate)
 * are serialized through {@code commitLock} and persist to disk before the
 * new state becomes visible in the registry; decoding is lock-free and always
 * reads one whole committed version.</p>
 */
@Service
public class HmmService {

    private static final int DEFAULT_MAX_ITERATIONS = 20;
    private static final double DEFAULT_TOLERANCE = 1e-4;
    private static final long DEFAULT_SEED = 0L;

    private final ModelRegistry registry;
    private final ModelStore store;
    private final HmmTrainer trainer = new HmmTrainer();
    private final HmmDecoder decoder = new HmmDecoder();
    private final Object commitLock = new Object();

    public HmmService(ModelRegistry registry, ModelStore store) {
        this.registry = registry;
        this.store = store;
        // Restart recovery: reload every committed version and the active pointer.
        registry.restore(store.loadVersions(), store.loadActiveVersion().orElse(null));
    }

    // ------------------------------------------------------------------
    // training
    // ------------------------------------------------------------------

    /** Validates all samples, trains deterministically, commits and activates a new version. */
    public ModelVersionView train(TrainRequest request) {
        if (request == null) {
            throw new ApiException(ErrorCode.INVALID_CONFIG, "request body is required");
        }
        TrainConfig config = toConfig(request);
        List<TrainSample> samples = validateSamples(request.samples(), config);

        HmmTrainer.TrainOutcome outcome;
        try {
            outcome = trainer.train(samples, config);
        } catch (IllegalArgumentException e) {
            throw new ApiException(ErrorCode.INVALID_CONFIG, e.getMessage());
        }

        synchronized (commitLock) {
            long id = registry.nextVersionId();
            ModelVersion version = new ModelVersion(
                    id,
                    Instant.now(),
                    config,
                    outcome.model(),
                    samples.size(),
                    outcome.iterationsRun(),
                    outcome.finalLogLikelihood());
            // Persist first: a version becomes visible only after it is durable.
            store.saveVersion(version);
            registry.register(version);
            registry.activate(id);
            store.saveRegistryState(id, versionIds());
            return toView(version, true);
        }
    }

    private TrainConfig toConfig(TrainRequest request) {
        Integer numStates = request.numStates();
        if (numStates == null || numStates < 1) {
            throw new ApiException(ErrorCode.INVALID_CONFIG, "numStates is required and must be >= 1");
        }
        int maxIterations = request.maxIterations() == null ? DEFAULT_MAX_ITERATIONS : request.maxIterations();
        if (maxIterations < 0) {
            throw new ApiException(ErrorCode.INVALID_CONFIG, "maxIterations must be >= 0");
        }
        double tolerance = request.tolerance() == null ? DEFAULT_TOLERANCE : request.tolerance();
        if (Double.isNaN(tolerance) || tolerance < 0.0) {
            throw new ApiException(ErrorCode.INVALID_CONFIG, "tolerance must be >= 0");
        }
        long seed = request.seed() == null ? DEFAULT_SEED : request.seed();
        List<String> stateLabels = request.stateLabels();
        if (stateLabels == null || stateLabels.isEmpty()) {
            // Resolve defaults eagerly so the stored config is fully explicit
            // and label validation has a concrete state set to check against.
            stateLabels = new ArrayList<>(numStates);
            for (int i = 0; i < numStates; i++) {
                stateLabels.add("S" + i);
            }
        } else {
            if (stateLabels.size() != numStates) {
                throw new ApiException(ErrorCode.INVALID_CONFIG,
                        "stateLabels size " + stateLabels.size() + " does not match numStates " + numStates);
            }
            if (stateLabels.stream().anyMatch(l -> l == null || l.isBlank())) {
                throw new ApiException(ErrorCode.INVALID_CONFIG, "stateLabels must not contain null or blank labels");
            }
            if (stateLabels.stream().distinct().count() != stateLabels.size()) {
                throw new ApiException(ErrorCode.INVALID_CONFIG, "stateLabels must be unique");
            }
        }
        return new TrainConfig(numStates, maxIterations, tolerance, seed, stateLabels);
    }

    /** Validates every sample up front; nothing is trained unless all samples pass. */
    private List<TrainSample> validateSamples(List<TrainSample> samples, TrainConfig config) {
        if (samples == null || samples.isEmpty()) {
            throw new ApiException(ErrorCode.INVALID_CONFIG, "at least one training sample is required");
        }
        List<TrainSample> validated = new ArrayList<>(samples.size());
        for (int i = 0; i < samples.size(); i++) {
            TrainSample sample = samples.get(i);
            String where = "samples[" + i + "]";
            if (sample == null) {
                throw new ApiException(ErrorCode.MISSING_OBSERVATION, where + " is missing");
            }
            List<String> observations = sample.observations();
            if (observations == null) {
                throw new ApiException(ErrorCode.MISSING_OBSERVATION, where + ".observations is missing");
            }
            if (observations.isEmpty()) {
                throw new ApiException(ErrorCode.EMPTY_SEQUENCE, where + ".observations must not be empty");
            }
            if (observations.stream().anyMatch(o -> o == null || o.isBlank())) {
                throw new ApiException(ErrorCode.MISSING_OBSERVATION,
                        where + ".observations contains a null or blank symbol");
            }
            List<String> labels = sample.labels();
            if (labels == null || labels.size() != observations.size()) {
                throw new ApiException(ErrorCode.INVALID_LABEL,
                        where + ".labels must be present and match the observations length");
            }
            for (String label : labels) {
                if (label == null || config.stateLabels().indexOf(label) < 0) {
                    throw new ApiException(ErrorCode.INVALID_LABEL,
                            where + " contains unknown label '" + label + "'");
                }
            }
            validated.add(sample);
        }
        return List.copyOf(validated);
    }

    // ------------------------------------------------------------------
    // decoding
    // ------------------------------------------------------------------

    /** Decodes with the requested version or the active one. Lock-free read path. */
    public DecodeResponse decode(DecodeRequest request) {
        if (request == null) {
            throw new ApiException(ErrorCode.MISSING_OBSERVATION, "request body is required");
        }
        List<String> observations = request.observations();
        if (observations == null) {
            throw new ApiException(ErrorCode.MISSING_OBSERVATION, "observations is required");
        }
        if (observations.isEmpty()) {
            throw new ApiException(ErrorCode.EMPTY_SEQUENCE, "observations must not be empty");
        }
        if (observations.stream().anyMatch(o -> o == null || o.isBlank())) {
            throw new ApiException(ErrorCode.MISSING_OBSERVATION, "observations contains a null or blank symbol");
        }

        // One atomic snapshot read: the whole request uses exactly this version.
        ModelVersion version = resolveVersion(request.modelVersion());
        HmmModel model = version.model();

        int[] ids = new int[observations.size()];
        for (int i = 0; i < observations.size(); i++) {
            Integer idx = model.symbolIndexOf(observations.get(i));
            if (idx == null) {
                throw new ApiException(ErrorCode.UNKNOWN_SYMBOL,
                        "unknown symbol '" + observations.get(i) + "' at position " + i);
            }
            ids[i] = idx;
        }
        DecodeResult result = decoder.decode(model, ids);
        return new DecodeResponse(
                version.version(),
                result.logLikelihood(),
                result.posteriors(),
                result.bestPath(),
                result.bestPathLogProbability());
    }

    private ModelVersion resolveVersion(Long requested) {
        if (requested != null) {
            return registry.find(requested)
                    .orElseThrow(() -> new ApiException(ErrorCode.MODEL_NOT_FOUND,
                            "no model version " + requested));
        }
        return registry.active()
                .orElseThrow(() -> new ApiException(ErrorCode.NO_ACTIVE_MODEL,
                        "no active model version; train one first"));
    }

    // ------------------------------------------------------------------
    // version management
    // ------------------------------------------------------------------

    public List<ModelVersionView> listVersions() {
        long activeId = registry.active().map(ModelVersion::version).orElse(-1L);
        return registry.list().stream()
                .map(v -> toView(v, v.version() == activeId))
                .toList();
    }

    /** Atomically switches the active version and persists the new pointer. */
    public ModelVersionView activate(long version) {
        synchronized (commitLock) {
            if (!registry.activate(version)) {
                throw new ApiException(ErrorCode.MODEL_NOT_FOUND, "no model version " + version);
            }
            store.saveRegistryState(version, versionIds());
            return toView(registry.find(version).orElseThrow(), true);
        }
    }

    private List<Long> versionIds() {
        return registry.list().stream().map(ModelVersion::version).toList();
    }

    private static ModelVersionView toView(ModelVersion v, boolean active) {
        return new ModelVersionView(
                v.version(),
                active,
                v.createdAt(),
                v.model().numStates(),
                v.sampleCount(),
                v.iterationsRun(),
                v.finalLogLikelihood());
    }
}
