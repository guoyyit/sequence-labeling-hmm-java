package com.github.guoyyit.sequencelabelinghmmjava.service;

import com.github.guoyyit.sequencelabelinghmmjava.domain.DecodeResult;
import com.github.guoyyit.sequencelabelinghmmjava.domain.HmmParameters;
import com.github.guoyyit.sequencelabelinghmmjava.domain.LabeledSample;
import com.github.guoyyit.sequencelabelinghmmjava.domain.ModelVersion;
import com.github.guoyyit.sequencelabelinghmmjava.domain.ScoreResult;
import com.github.guoyyit.sequencelabelinghmmjava.domain.TrainingConfig;
import com.github.guoyyit.sequencelabelinghmmjava.error.ErrorCode;
import com.github.guoyyit.sequencelabelinghmmjava.error.ServiceException;
import com.github.guoyyit.sequencelabelinghmmjava.hmm.HmmInference;
import com.github.guoyyit.sequencelabelinghmmjava.hmm.HmmTrainer;
import com.github.guoyyit.sequencelabelinghmmjava.registry.ModelRegistry;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Application service orchestrating training, versioned registration,
 * atomic activation and concurrent decoding.
 *
 * <p>Training runs are serialized through a lock; each run is a pure function
 * of its (validated) input and commits one new immutable version. Decoding and
 * scoring are lock-free: they read the active version through an atomic
 * reference, so they never observe a mix of old and new parameters.</p>
 */
public class SequenceLabelingService {

    private final ModelRegistry registry;
    private final ReentrantLock trainLock = new ReentrantLock();

    public SequenceLabelingService(ModelRegistry registry) {
        this.registry = registry;
    }

    /**
     * Validates the whole batch, trains deterministically and commits a new
     * version. An invalid sample aborts the batch without any partial update.
     */
    public ModelVersion train(List<LabeledSample> samples, TrainingConfig config) {
        trainLock.lock();
        try {
            HmmTrainer.TrainingOutcome outcome = HmmTrainer.train(samples, config);
            TrainingConfig effective = config == null ? TrainingConfig.defaults() : config;
            return registry.commit(outcome.parameters(), effective, outcome.metrics());
        } finally {
            trainLock.unlock();
        }
    }

    /** Decodes with the given version, or the active version when null. */
    public DecodeResult decode(List<String> observations, String versionId) {
        ModelVersion version = resolveVersion(versionId);
        int[] ids = toSymbolIds(observations, version.parameters());
        HmmInference.ViterbiResult result = HmmInference.viterbi(version.parameters(), ids);
        List<String> path = new ArrayList<>(result.statePath().length);
        for (int state : result.statePath()) {
            path.add(version.parameters().getLabels().get(state));
        }
        return new DecodeResult(path, result.logProbability(), version.versionId());
    }

    /** Computes per-position state (posterior) probabilities and the log-likelihood. */
    public ScoreResult score(List<String> observations, boolean logSpace, String versionId) {
        ModelVersion version = resolveVersion(versionId);
        int[] ids = toSymbolIds(observations, version.parameters());
        HmmInference.ForwardBackwardResult fb =
                HmmInference.forwardBackward(version.parameters(), ids);
        double[][] logGamma = fb.logGamma();
        double[][] probabilities = logGamma;
        if (!logSpace) {
            probabilities = new double[logGamma.length][];
            for (int t = 0; t < logGamma.length; t++) {
                probabilities[t] = new double[logGamma[t].length];
                for (int i = 0; i < logGamma[t].length; i++) {
                    probabilities[t][i] = StrictMath.exp(logGamma[t][i]);
                }
            }
        }
        return new ScoreResult(fb.logLikelihood(), probabilities, logSpace,
                version.parameters().getLabels(), version.versionId());
    }

    /** Atomically switches the active version. */
    public ModelVersion activate(String versionId) {
        return registry.activate(versionId);
    }

    public List<ModelVersion> versions() {
        return registry.list();
    }

    public ModelVersion activeVersion() {
        return registry.active().orElseThrow(() ->
                new ServiceException(ErrorCode.NO_ACTIVE_MODEL, "no active model version"));
    }

    /** @return the active version id, or null when none is active. */
    public String activeVersionId() {
        return registry.active().map(ModelVersion::versionId).orElse(null);
    }

    private ModelVersion resolveVersion(String versionId) {
        if (versionId == null || versionId.isBlank()) {
            return activeVersion();
        }
        return registry.find(versionId).orElseThrow(() ->
                new ServiceException(ErrorCode.MODEL_VERSION_NOT_FOUND,
                        "unknown model version: " + versionId));
    }

    /**
     * Validates the observation sequence and maps it to symbol ids. Missing,
     * empty and unknown observations are rejected with distinguishable errors;
     * nothing is silently defaulted.
     */
    private int[] toSymbolIds(List<String> observations, HmmParameters params) {
        if (observations == null) {
            throw new ServiceException(ErrorCode.MISSING_OBSERVATION,
                    "observation sequence is missing");
        }
        if (observations.isEmpty()) {
            throw new ServiceException(ErrorCode.EMPTY_SEQUENCE,
                    "observation sequence is empty");
        }
        int[] ids = new int[observations.size()];
        for (int t = 0; t < observations.size(); t++) {
            String obs = observations.get(t);
            if (obs == null || obs.isBlank()) {
                throw new ServiceException(ErrorCode.MISSING_OBSERVATION,
                        "missing observation at position " + t);
            }
            Integer id = params.indexOfSymbol(obs);
            if (id == null) {
                throw new ServiceException(ErrorCode.UNKNOWN_SYMBOL,
                        "unknown observation symbol at position " + t + ": " + obs);
            }
            ids[t] = id;
        }
        return ids;
    }
}
