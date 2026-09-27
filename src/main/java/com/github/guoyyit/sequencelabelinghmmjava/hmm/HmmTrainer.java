package com.github.guoyyit.sequencelabelinghmmjava.hmm;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.TreeSet;

/**
 * Deterministic HMM trainer.
 *
 * <p>Pipeline: (1) samples are canonically sorted so that repeated or
 * out-of-order arrival yields byte-identical results; (2) supervised
 * maximum-likelihood counts with add-one smoothing produce the initial
 * parameters; (3) when {@code maxIterations > 0} a seeded jitter breaks
 * symmetry and Baum-Welch EM refines the parameters on the observation
 * sequences until the log-likelihood gain drops below {@code tolerance} or
 * the iteration budget is exhausted.</p>
 *
 * <p>The trainer is a pure function: it validates everything up front, keeps
 * all intermediate state local, and either returns a complete model or throws
 * — it can never leave a partial update behind.</p>
 */
public final class HmmTrainer {

    /** Pseudo-count floor used during EM re-estimation to keep probabilities positive. */
    private static final double EM_PSEUDO_COUNT = 1e-9;

    /** Amplitude of the deterministic pre-EM symmetry-breaking jitter. */
    private static final double JITTER_AMPLITUDE = 1e-3;

    /** Outcome of a training run: the model plus reproducibility metadata. */
    public record TrainOutcome(HmmModel model, int iterationsRun, double finalLogLikelihood) {
    }

    public TrainOutcome train(List<TrainSample> samples, TrainConfig config) {
        List<String> stateLabels = validateConfig(config);
        if (samples == null || samples.isEmpty()) {
            throw new IllegalArgumentException("at least one training sample is required");
        }
        List<TrainSample> sorted = canonicalOrder(samples);
        List<String> vocabulary = buildVocabulary(sorted);
        int[][] observationIds = encodeObservations(sorted, vocabulary);
        int[][] labelIds = encodeLabels(sorted, stateLabels);

        int states = stateLabels.size();
        // Integer count accumulation is exact and order-independent.
        double[] logPi = new double[states];
        double[][] logA = new double[states][states];
        double[][] logB = new double[states][vocabulary.size()];
        supervisedInitialParameters(observationIds, labelIds, states, vocabulary.size(), logPi, logA, logB);

        int iterationsRun = 0;
        double finalLogLikelihood = datasetLogLikelihood(logPi, logA, logB, observationIds, stateLabels, vocabulary);
        if (config.maxIterations() > 0) {
            jitter(logPi, logA, logB, config.seed());
            EmResult em = baumWelch(logPi, logA, logB, observationIds, config.maxIterations(), config.tolerance());
            iterationsRun = em.iterationsRun;
            finalLogLikelihood = em.finalLogLikelihood;
        }
        HmmModel model = new HmmModel(stateLabels, vocabulary, logPi, logA, logB);
        return new TrainOutcome(model, iterationsRun, finalLogLikelihood);
    }

    // ------------------------------------------------------------------
    // validation & canonicalization
    // ------------------------------------------------------------------

    private static List<String> validateConfig(TrainConfig config) {
        if (config == null) {
            throw new IllegalArgumentException("training config is required");
        }
        if (config.numStates() < 1) {
            throw new IllegalArgumentException("numStates must be >= 1, got " + config.numStates());
        }
        if (config.maxIterations() < 0) {
            throw new IllegalArgumentException("maxIterations must be >= 0, got " + config.maxIterations());
        }
        if (config.tolerance() < 0.0 || Double.isNaN(config.tolerance())) {
            throw new IllegalArgumentException("tolerance must be >= 0, got " + config.tolerance());
        }
        List<String> labels = config.stateLabels();
        if (labels == null || labels.isEmpty()) {
            labels = new ArrayList<>();
            for (int i = 0; i < config.numStates(); i++) {
                labels.add("S" + i);
            }
        }
        if (labels.size() != config.numStates()) {
            throw new IllegalArgumentException(
                    "stateLabels size " + labels.size() + " does not match numStates " + config.numStates());
        }
        if (labels.stream().anyMatch(l -> l == null || l.isBlank())) {
            throw new IllegalArgumentException("stateLabels must not contain null or blank labels");
        }
        if (new TreeSet<>(labels).size() != labels.size()) {
            throw new IllegalArgumentException("stateLabels must be unique");
        }
        return List.copyOf(labels);
    }

    /**
     * Sorts samples by an unambiguous length-prefixed canonical key so that any
     * arrival order (and any number of duplicates) produces the same sequence.
     */
    static List<TrainSample> canonicalOrder(List<TrainSample> samples) {
        List<TrainSample> copy = new ArrayList<>(samples);
        copy.sort(Comparator.comparing(HmmTrainer::canonicalKey));
        return copy;
    }

    private static String canonicalKey(TrainSample sample) {
        StringBuilder sb = new StringBuilder();
        appendEncoded(sb, sample.observations());
        sb.append('#');
        appendEncoded(sb, sample.labels());
        return sb.toString();
    }

    private static void appendEncoded(StringBuilder sb, List<String> values) {
        for (String v : values) {
            sb.append(v.length()).append(':').append(v).append(';');
        }
    }

    private static List<String> buildVocabulary(List<TrainSample> samples) {
        TreeSet<String> symbols = new TreeSet<>();
        for (TrainSample sample : samples) {
            symbols.addAll(sample.observations());
        }
        return List.copyOf(symbols);
    }

    private static int[][] encodeObservations(List<TrainSample> samples, List<String> vocabulary) {
        int[][] ids = new int[samples.size()][];
        for (int i = 0; i < samples.size(); i++) {
            List<String> obs = samples.get(i).observations();
            ids[i] = new int[obs.size()];
            for (int t = 0; t < obs.size(); t++) {
                // binarySearch is exact because the vocabulary was built from these samples.
                ids[i][t] = vocabularyIndex(vocabulary, obs.get(t));
            }
        }
        return ids;
    }

    private static int vocabularyIndex(List<String> vocabulary, String symbol) {
        int idx = java.util.Collections.binarySearch(vocabulary, symbol);
        if (idx < 0) {
            throw new IllegalArgumentException("symbol not in vocabulary: " + symbol);
        }
        return idx;
    }

    private static int[][] encodeLabels(List<TrainSample> samples, List<String> stateLabels) {
        int[][] ids = new int[samples.size()][];
        for (int i = 0; i < samples.size(); i++) {
            List<String> labels = samples.get(i).labels();
            ids[i] = new int[labels.size()];
            for (int t = 0; t < labels.size(); t++) {
                int idx = stateLabels.indexOf(labels.get(t));
                if (idx < 0) {
                    throw new IllegalArgumentException("unknown state label: " + labels.get(t));
                }
                ids[i][t] = idx;
            }
        }
        return ids;
    }

    // ------------------------------------------------------------------
    // supervised initialization (add-one smoothed MLE)
    // ------------------------------------------------------------------

    private static void supervisedInitialParameters(int[][] obs, int[][] labels, int states, int vocabSize,
                                                    double[] logPi, double[][] logA, double[][] logB) {
        long[] initCounts = new long[states];
        long[][] transCounts = new long[states][states];
        long[][] emitCounts = new long[states][vocabSize];
        for (int i = 0; i < obs.length; i++) {
            int[] lbl = labels[i];
            initCounts[lbl[0]]++;
            for (int t = 1; t < lbl.length; t++) {
                transCounts[lbl[t - 1]][lbl[t]]++;
            }
            for (int t = 0; t < lbl.length; t++) {
                emitCounts[lbl[t]][obs[i][t]]++;
            }
        }
        fillLogRow(logPi, initCounts);
        for (int s = 0; s < states; s++) {
            fillLogRow(logA[s], transCounts[s]);
            fillLogRow(logB[s], emitCounts[s]);
        }
    }

    /** Add-one smoothing keeps every parameter strictly positive for the EM start. */
    private static void fillLogRow(double[] logRow, long[] counts) {
        double total = counts.length;
        for (long c : counts) {
            total += c;
        }
        for (int i = 0; i < counts.length; i++) {
            logRow[i] = Math.log((counts[i] + 1.0) / total);
        }
    }

    // ------------------------------------------------------------------
    // seeded symmetry-breaking jitter
    // ------------------------------------------------------------------

    private static void jitter(double[] logPi, double[][] logA, double[][] logB, long seed) {
        Random random = new Random(seed);
        jitterRow(logPi, random);
        for (double[] row : logA) {
            jitterRow(row, random);
        }
        for (double[] row : logB) {
            jitterRow(row, random);
        }
    }

    private static void jitterRow(double[] logRow, Random random) {
        for (int i = 0; i < logRow.length; i++) {
            double factor = 1.0 + JITTER_AMPLITUDE * (random.nextDouble() - 0.5);
            logRow[i] += Math.log(factor);
        }
        LogMath.logNormalizeInPlace(logRow);
    }

    // ------------------------------------------------------------------
    // Baum-Welch EM
    // ------------------------------------------------------------------

    private record EmResult(int iterationsRun, double finalLogLikelihood) {
    }

    private static EmResult baumWelch(double[] logPi, double[][] logA, double[][] logB,
                                      int[][] sequences, int maxIterations, double tolerance) {
        int states = logPi.length;
        int vocabSize = logB[0].length;
        List<String> stubLabels = stubLabels(states);
        List<String> stubVocab = stubVocabulary(vocabSize);
        // HmmModel defensively copies its arrays, so the working model must be
        // rebuilt after every M-step for the next E-step to see fresh parameters.
        HmmModel workModel = new HmmModel(stubLabels, stubVocab, logPi, logA, logB);

        double prevLogLik = datasetLogLikelihood(workModel, sequences);
        int iteration = 0;
        while (iteration < maxIterations) {
            double[] initCounts = new double[states];
            double[][] transCounts = new double[states][states];
            double[][] emitCounts = new double[states][vocabSize];
            double totalLogLik = 0.0;
            for (int[] obs : sequences) {
                totalLogLik += accumulateExpectedCounts(workModel, obs, initCounts, transCounts, emitCounts);
            }
            reestimate(logPi, logA, logB, initCounts, transCounts, emitCounts);
            workModel = new HmmModel(stubLabels, stubVocab, logPi, logA, logB);
            iteration++;
            double gain = totalLogLik - prevLogLik;
            prevLogLik = totalLogLik;
            if (Math.abs(gain) < tolerance) {
                break;
            }
        }
        // Report the likelihood of the parameters actually returned.
        return new EmResult(iteration, datasetLogLikelihood(workModel, sequences));
    }

    /** E-step for one sequence: adds gamma/xi expected counts, returns its log-likelihood. */
    private static double accumulateExpectedCounts(HmmModel model, int[] obs,
                                                   double[] initCounts, double[][] transCounts, double[][] emitCounts) {
        HmmDecoder.ForwardBackward fb = HmmDecoder.forwardBackward(model, obs);
        int states = model.numStates();
        int len = obs.length;
        double[][] logA = model.logA();
        double[][] logB = model.logB();

        for (int t = 0; t < len; t++) {
            for (int s = 0; s < states; s++) {
                double gamma = Math.exp(fb.alpha()[t][s] + fb.beta()[t][s] - fb.logLik());
                emitCounts[s][obs[t]] += gamma;
                if (t == 0) {
                    initCounts[s] += gamma;
                }
            }
        }
        for (int t = 0; t + 1 < len; t++) {
            for (int a = 0; a < states; a++) {
                for (int b = 0; b < states; b++) {
                    double logXi = fb.alpha()[t][a] + logA[a][b] + logB[b][obs[t + 1]]
                            + fb.beta()[t + 1][b] - fb.logLik();
                    transCounts[a][b] += Math.exp(logXi);
                }
            }
        }
        return fb.logLik();
    }

    /** M-step: normalize expected counts back into log probabilities with a tiny floor. */
    private static void reestimate(double[] logPi, double[][] logA, double[][] logB,
                                   double[] initCounts, double[][] transCounts, double[][] emitCounts) {
        normalizeInto(logPi, initCounts);
        for (int s = 0; s < logPi.length; s++) {
            normalizeInto(logA[s], transCounts[s]);
            normalizeInto(logB[s], emitCounts[s]);
        }
    }

    private static void normalizeInto(double[] logRow, double[] counts) {
        double total = 0.0;
        for (double c : counts) {
            total += c + EM_PSEUDO_COUNT;
        }
        for (int i = 0; i < counts.length; i++) {
            logRow[i] = Math.log((counts[i] + EM_PSEUDO_COUNT) / total);
        }
    }

    // ------------------------------------------------------------------
    // helpers
    // ------------------------------------------------------------------

    private static double datasetLogLikelihood(double[] logPi, double[][] logA, double[][] logB,
                                               int[][] sequences, List<String> stateLabels, List<String> vocabulary) {
        HmmModel model = new HmmModel(stateLabels, vocabulary, logPi, logA, logB);
        return datasetLogLikelihood(model, sequences);
    }

    private static double datasetLogLikelihood(HmmModel model, int[][] sequences) {
        double total = 0.0;
        for (int[] obs : sequences) {
            total += HmmDecoder.forwardBackward(model, obs).logLik();
        }
        return total;
    }

    private static List<String> stubLabels(int states) {
        List<String> labels = new ArrayList<>(states);
        for (int i = 0; i < states; i++) {
            labels.add("S" + i);
        }
        return labels;
    }

    private static List<String> stubVocabulary(int vocabSize) {
        List<String> vocab = new ArrayList<>(vocabSize);
        for (int i = 0; i < vocabSize; i++) {
            vocab.add("sym" + i);
        }
        return vocab;
    }
}
