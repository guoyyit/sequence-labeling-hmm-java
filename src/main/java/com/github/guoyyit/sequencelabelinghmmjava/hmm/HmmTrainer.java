package com.github.guoyyit.sequencelabelinghmmjava.hmm;

import com.github.guoyyit.sequencelabelinghmmjava.domain.HmmParameters;
import com.github.guoyyit.sequencelabelinghmmjava.domain.LabeledSample;
import com.github.guoyyit.sequencelabelinghmmjava.domain.TrainingConfig;
import com.github.guoyyit.sequencelabelinghmmjava.domain.TrainingMetrics;
import com.github.guoyyit.sequencelabelinghmmjava.error.ErrorCode;
import com.github.guoyyit.sequencelabelinghmmjava.error.ServiceException;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * Deterministic HMM trainer.
 *
 * <p>Pipeline: validate the whole batch (any invalid sample aborts the batch,
 * leaving no partial update) -&gt; de-duplicate identical samples and sort them
 * into a canonical order (so repeated or out-of-order arrival of the same
 * sample set yields bit-identical parameters) -&gt; supervised maximum-likelihood
 * estimation with add-alpha smoothing -&gt; optional Baum-Welch refinement
 * (at most {@code maxIterations} rounds, stopping when the log-likelihood
 * improvement drops below {@code convergenceThreshold}).</p>
 *
 * <p>States never observed in the labels (only possible when
 * {@code stateCount} exceeds the number of distinct labels) are initialized
 * from a seeded RNG, keeping every run with the same seed reproducible.</p>
 */
public final class HmmTrainer {

    private HmmTrainer() {
    }

    /** Training output: parameters plus diagnostics. */
    public record TrainingOutcome(HmmParameters parameters, TrainingMetrics metrics) {
    }

    /** Expected sufficient statistics of one E-step. */
    private static final class ExpectedCounts {
        final double[] init;
        final double[][] trans;
        final double[][] emit;
        double logLikelihood;

        ExpectedCounts(int states, int symbols) {
            this.init = new double[states];
            this.trans = new double[states][states];
            this.emit = new double[states][symbols];
        }
    }

    public static TrainingOutcome train(List<LabeledSample> samples, TrainingConfig config) {
        TrainingConfig cfg = config == null ? TrainingConfig.defaults() : config;
        validateConfig(cfg);
        List<LabeledSample> canonical = canonicalize(validate(samples));

        List<String> vocabulary = List.copyOf(new TreeSet<>(collectObservations(canonical)));
        List<String> labels = List.copyOf(new TreeSet<>(collectLabels(canonical)));
        int stateCount = resolveStateCount(cfg, labels.size());

        int[][] observationIds = toObservationIds(canonical, vocabulary);
        int[][] labelIds = toLabelIds(canonical, labels);

        HmmParameters initial = supervisedEstimate(
                observationIds, labelIds, stateCount, vocabulary, labels, cfg);

        int symbols = vocabulary.size();
        ExpectedCounts ec = eStep(initial, observationIds, stateCount, symbols);
        double logLikelihood = ec.logLikelihood;
        HmmParameters current = initial;

        int iterations = 0;
        boolean converged = cfg.maxIterations() == 0;
        while (iterations < cfg.maxIterations()) {
            current = mStep(ec, current, stateCount, symbols, vocabulary, labels, cfg.smoothingAlpha());
            iterations++;
            ec = eStep(current, observationIds, stateCount, symbols);
            double improvement = ec.logLikelihood - logLikelihood;
            logLikelihood = ec.logLikelihood;
            if (StrictMath.abs(improvement) < cfg.convergenceThreshold()) {
                converged = true;
                break;
            }
        }

        TrainingMetrics metrics = new TrainingMetrics(
                samples.size(), canonical.size(), logLikelihood, iterations, converged);
        return new TrainingOutcome(current, metrics);
    }

    // ------------------------------------------------------------------
    // Validation
    // ------------------------------------------------------------------

    private static void validateConfig(TrainingConfig cfg) {
        if (cfg.stateCount() != null && cfg.stateCount() < 1) {
            throw new ServiceException(ErrorCode.INVALID_CONFIG,
                    "stateCount must be >= 1, got " + cfg.stateCount());
        }
        if (cfg.maxIterations() < 0) {
            throw new ServiceException(ErrorCode.INVALID_CONFIG,
                    "maxIterations must be >= 0, got " + cfg.maxIterations());
        }
        if (Double.isNaN(cfg.convergenceThreshold()) || cfg.convergenceThreshold() < 0) {
            throw new ServiceException(ErrorCode.INVALID_CONFIG,
                    "convergenceThreshold must be >= 0, got " + cfg.convergenceThreshold());
        }
        if (Double.isNaN(cfg.smoothingAlpha()) || cfg.smoothingAlpha() < 0) {
            throw new ServiceException(ErrorCode.INVALID_CONFIG,
                    "smoothingAlpha must be >= 0, got " + cfg.smoothingAlpha());
        }
    }

    private static List<LabeledSample> validate(List<LabeledSample> samples) {
        if (samples == null || samples.isEmpty()) {
            throw new ServiceException(ErrorCode.EMPTY_SEQUENCE, "no training samples provided");
        }
        List<LabeledSample> validated = new ArrayList<>(samples.size());
        for (int i = 0; i < samples.size(); i++) {
            LabeledSample sample = samples.get(i);
            if (sample == null) {
                throw new ServiceException(ErrorCode.EMPTY_SEQUENCE, "sample " + i + " is null");
            }
            if (sample.observations() == null) {
                throw new ServiceException(ErrorCode.MISSING_OBSERVATION,
                        "sample " + i + " has no observation sequence");
            }
            if (sample.observations().isEmpty()) {
                throw new ServiceException(ErrorCode.EMPTY_SEQUENCE,
                        "sample " + i + " has an empty observation sequence");
            }
            for (int t = 0; t < sample.observations().size(); t++) {
                String obs = sample.observations().get(t);
                if (obs == null || obs.isBlank()) {
                    throw new ServiceException(ErrorCode.MISSING_OBSERVATION,
                            "sample " + i + " is missing an observation at position " + t);
                }
            }
            if (sample.labels() == null || sample.labels().size() != sample.observations().size()) {
                throw new ServiceException(ErrorCode.INVALID_LABEL,
                        "sample " + i + " label count does not match observation count");
            }
            for (int t = 0; t < sample.labels().size(); t++) {
                String label = sample.labels().get(t);
                if (label == null || label.isBlank()) {
                    throw new ServiceException(ErrorCode.INVALID_LABEL,
                            "sample " + i + " has an invalid label at position " + t);
                }
            }
            validated.add(sample);
        }
        return validated;
    }

    /**
     * De-duplicates identical samples and sorts the rest into a canonical
     * order, making the result independent of arrival order and repetitions.
     */
    private static List<LabeledSample> canonicalize(List<LabeledSample> samples) {
        TreeMap<String, LabeledSample> byKey = new TreeMap<>();
        for (LabeledSample sample : samples) {
            byKey.putIfAbsent(canonicalKey(sample), sample);
        }
        return new ArrayList<>(byKey.values());
    }

    private static String canonicalKey(LabeledSample sample) {
        StringBuilder sb = new StringBuilder();
        appendLengthPrefixed(sb, sample.observations());
        sb.append('#');
        appendLengthPrefixed(sb, sample.labels());
        return sb.toString();
    }

    private static void appendLengthPrefixed(StringBuilder sb, List<String> values) {
        for (String value : values) {
            sb.append(value.length()).append(':').append(value).append(';');
        }
    }

    // ------------------------------------------------------------------
    // Indexing
    // ------------------------------------------------------------------

    private static List<String> collectObservations(List<LabeledSample> samples) {
        List<String> all = new ArrayList<>();
        for (LabeledSample sample : samples) {
            all.addAll(sample.observations());
        }
        return all;
    }

    private static List<String> collectLabels(List<LabeledSample> samples) {
        List<String> all = new ArrayList<>();
        for (LabeledSample sample : samples) {
            all.addAll(sample.labels());
        }
        return all;
    }

    private static int resolveStateCount(TrainingConfig cfg, int distinctLabels) {
        int stateCount = cfg.stateCount() == null ? distinctLabels : cfg.stateCount();
        if (stateCount < distinctLabels) {
            throw new ServiceException(ErrorCode.INVALID_CONFIG,
                    "stateCount " + stateCount + " is smaller than the distinct label count "
                            + distinctLabels);
        }
        return stateCount;
    }

    private static int[][] toObservationIds(List<LabeledSample> samples, List<String> vocabulary) {
        int[][] ids = new int[samples.size()][];
        for (int i = 0; i < samples.size(); i++) {
            List<String> observations = samples.get(i).observations();
            ids[i] = new int[observations.size()];
            for (int t = 0; t < observations.size(); t++) {
                ids[i][t] = vocabulary.indexOf(observations.get(t));
            }
        }
        return ids;
    }

    private static int[][] toLabelIds(List<LabeledSample> samples, List<String> labels) {
        int[][] ids = new int[samples.size()][];
        for (int i = 0; i < samples.size(); i++) {
            List<String> sampleLabels = samples.get(i).labels();
            ids[i] = new int[sampleLabels.size()];
            for (int t = 0; t < sampleLabels.size(); t++) {
                ids[i][t] = labels.indexOf(sampleLabels.get(t));
            }
        }
        return ids;
    }

    // ------------------------------------------------------------------
    // Supervised MLE initialization
    // ------------------------------------------------------------------

    private static HmmParameters supervisedEstimate(
            int[][] observationIds, int[][] labelIds, int stateCount,
            List<String> vocabulary, List<String> labels, TrainingConfig cfg) {
        int symbols = vocabulary.size();
        double[] initCounts = new double[stateCount];
        double[][] transCounts = new double[stateCount][stateCount];
        double[][] emitCounts = new double[stateCount][symbols];
        boolean[] seen = new boolean[stateCount];

        for (int s = 0; s < observationIds.length; s++) {
            int[] obs = observationIds[s];
            int[] lab = labelIds[s];
            initCounts[lab[0]] += 1.0;
            seen[lab[0]] = true;
            for (int t = 0; t < obs.length; t++) {
                emitCounts[lab[t]][obs[t]] += 1.0;
                seen[lab[t]] = true;
                if (t + 1 < obs.length) {
                    transCounts[lab[t]][lab[t + 1]] += 1.0;
                }
            }
        }

        // Seeded random initialization for states never observed in labels.
        Random random = new Random(cfg.seed());
        for (int i = 0; i < stateCount; i++) {
            if (!seen[i]) {
                initCounts[i] = random.nextDouble();
                for (int j = 0; j < stateCount; j++) {
                    transCounts[i][j] = 1.0 + random.nextDouble();
                }
                for (int k = 0; k < symbols; k++) {
                    emitCounts[i][k] = 1.0 + random.nextDouble();
                }
            }
        }

        double alpha = cfg.smoothingAlpha();
        double[] logInit = normalizeCounts(initCounts, alpha, null);
        double[][] logTrans = normalizeRows(transCounts, alpha, null);
        double[][] logEmit = normalizeRows(emitCounts, alpha, null);
        return new HmmParameters(stateCount, labels, vocabulary, logInit, logTrans, logEmit);
    }

    // ------------------------------------------------------------------
    // Baum-Welch
    // ------------------------------------------------------------------

    private static ExpectedCounts eStep(HmmParameters params, int[][] sequences,
                                        int states, int symbols) {
        ExpectedCounts counts = new ExpectedCounts(states, symbols);
        double[] logPi = params.getLogInitial();
        double[][] logA = params.getLogTransition();
        double[][] logB = params.getLogEmission();

        double totalLL = 0.0;
        for (int[] seq : sequences) {
            HmmInference.ForwardBackwardResult fb = HmmInference.forwardBackward(params, seq);
            double ll = fb.logLikelihood();
            if (ll == HmmMath.LOG_ZERO) {
                // Sequence impossible under the current parameters (only
                // reachable with zero smoothing); contributes no counts.
                continue;
            }
            totalLL += ll;
            double[][] logAlpha = fb.logAlpha();
            double[][] logBeta = fb.logBeta();

            for (int t = 0; t < seq.length; t++) {
                int obs = seq[t];
                for (int i = 0; i < states; i++) {
                    double gamma = StrictMath.exp(logAlpha[t][i] + logBeta[t][i] - ll);
                    counts.emit[i][obs] += gamma;
                    if (t == 0) {
                        counts.init[i] += gamma;
                    }
                }
            }
            for (int t = 0; t + 1 < seq.length; t++) {
                int nextObs = seq[t + 1];
                for (int i = 0; i < states; i++) {
                    double ai = logAlpha[t][i];
                    if (ai == HmmMath.LOG_ZERO) {
                        continue;
                    }
                    for (int j = 0; j < states; j++) {
                        double xi = StrictMath.exp(
                                ai + logA[i][j] + logB[j][nextObs] + logBeta[t + 1][j] - ll);
                        counts.trans[i][j] += xi;
                    }
                }
            }
        }
        counts.logLikelihood = totalLL;
        return counts;
    }

    private static HmmParameters mStep(ExpectedCounts counts, HmmParameters previous,
                                       int states, int symbols, List<String> vocabulary,
                                       List<String> labels, double alpha) {
        double[] logInit = normalizeCounts(counts.init, alpha, previous.getLogInitial());
        double[][] logTrans = normalizeRows(counts.trans, alpha, previous.getLogTransition());
        double[][] logEmit = normalizeRows(counts.emit, alpha, previous.getLogEmission());
        return new HmmParameters(states, labels, vocabulary, logInit, logTrans, logEmit);
    }

    // ------------------------------------------------------------------
    // Normalization helpers
    // ------------------------------------------------------------------

    /**
     * log-normalizes a count vector with add-alpha smoothing. A zero-count
     * vector falls back to {@code fallback} when given, otherwise to the
     * uniform distribution (deterministic, never NaN).
     */
    private static double[] normalizeCounts(double[] counts, double alpha, double[] fallback) {
        int n = counts.length;
        double sum = 0.0;
        for (double c : counts) {
            sum += c + alpha;
        }
        double[] log = new double[n];
        if (sum == 0.0) {
            if (fallback != null) {
                return fallback.clone();
            }
            double uniform = -StrictMath.log(n);
            java.util.Arrays.fill(log, uniform);
            return log;
        }
        double logSum = StrictMath.log(sum);
        for (int i = 0; i < n; i++) {
            log[i] = StrictMath.log(counts[i] + alpha) - logSum;
        }
        return log;
    }

    private static double[][] normalizeRows(double[][] counts, double alpha, double[][] fallback) {
        double[][] log = new double[counts.length][];
        for (int i = 0; i < counts.length; i++) {
            log[i] = normalizeCounts(counts[i], alpha, fallback == null ? null : fallback[i]);
        }
        return log;
    }
}
