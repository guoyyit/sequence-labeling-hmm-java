package com.github.guoyyit.sequencelabelinghmmjava.hmm;

import java.util.ArrayList;
import java.util.List;

/**
 * Log-space forward-backward and Viterbi decoding.
 *
 * <p>Every computation runs on log probabilities with
 * {@link LogMath#logSumExp(double[])} accumulation, so arbitrarily long
 * sequences neither underflow to zero nor overflow, and the returned
 * likelihoods are exact log-space values.</p>
 */
public final class HmmDecoder {

    /**
     * Decodes one observation sequence given as vocabulary indices.
     *
     * @param model the (immutable) model to decode with
     * @param obs   observation symbol indices, must be non-empty
     * @return likelihood, per-position posterior state probabilities and the Viterbi path
     */
    public DecodeResult decode(HmmModel model, int[] obs) {
        if (obs == null || obs.length == 0) {
            throw new IllegalArgumentException("observation sequence must not be empty");
        }
        ForwardBackward fb = forwardBackward(model, obs);
        double[][] posteriors = posteriors(fb);
        int[] bestStates = viterbi(model, obs);
        List<String> bestPath = new ArrayList<>(bestStates.length);
        for (int s : bestStates) {
            bestPath.add(model.stateLabels().get(s));
        }
        return new DecodeResult(fb.logLik, posteriors, bestPath, bestPathLogProbability(model, obs, bestStates));
    }

    /** Package-private holder for alpha/beta tables, shared with the trainer. */
    record ForwardBackward(double[][] alpha, double[][] beta, double logLik) {
    }

    /** Runs the log-space forward and backward recursions. */
    static ForwardBackward forwardBackward(HmmModel model, int[] obs) {
        int states = model.numStates();
        int len = obs.length;
        double[] logPi = model.logPi();
        double[][] logA = model.logA();
        double[][] logB = model.logB();

        double[][] alpha = new double[len][states];
        for (int s = 0; s < states; s++) {
            alpha[0][s] = logPi[s] + logB[s][obs[0]];
        }
        double[] scratch = new double[states];
        for (int t = 1; t < len; t++) {
            for (int s = 0; s < states; s++) {
                for (int prev = 0; prev < states; prev++) {
                    scratch[prev] = alpha[t - 1][prev] + logA[prev][s];
                }
                alpha[t][s] = LogMath.logSumExp(scratch) + logB[s][obs[t]];
            }
        }
        double logLik = LogMath.logSumExp(alpha[len - 1]);

        double[][] beta = new double[len][states];
        // beta[T-1][s] = log(1) = 0 by definition.
        for (int t = len - 2; t >= 0; t--) {
            for (int s = 0; s < states; s++) {
                for (int next = 0; next < states; next++) {
                    scratch[next] = logA[s][next] + logB[next][obs[t + 1]] + beta[t + 1][next];
                }
                beta[t][s] = LogMath.logSumExp(scratch);
            }
        }
        return new ForwardBackward(alpha, beta, logLik);
    }

    /** gamma[t][s] = P(state_t = s | observations), each row normalized to sum to 1. */
    private static double[][] posteriors(ForwardBackward fb) {
        int len = fb.alpha.length;
        int states = fb.alpha[0].length;
        double[][] gamma = new double[len][states];
        for (int t = 0; t < len; t++) {
            double rowSum = LogMath.LOG_ZERO;
            for (int s = 0; s < states; s++) {
                // alpha+beta-logLik is <= 0 in exact arithmetic, so exp is safe.
                gamma[t][s] = Math.exp(fb.alpha[t][s] + fb.beta[t][s] - fb.logLik);
                rowSum = LogMath.logAdd(rowSum, Math.log(gamma[t][s]));
            }
            if (rowSum != LogMath.LOG_ZERO && rowSum != 0.0) {
                for (int s = 0; s < states; s++) {
                    gamma[t][s] /= Math.exp(rowSum);
                }
            }
        }
        return gamma;
    }

    /** Log-space Viterbi; ties resolve to the lowest state index for determinism. */
    static int[] viterbi(HmmModel model, int[] obs) {
        int states = model.numStates();
        int len = obs.length;
        double[] logPi = model.logPi();
        double[][] logA = model.logA();
        double[][] logB = model.logB();

        double[][] delta = new double[len][states];
        int[][] back = new int[len][states];
        for (int s = 0; s < states; s++) {
            delta[0][s] = logPi[s] + logB[s][obs[0]];
        }
        for (int t = 1; t < len; t++) {
            for (int s = 0; s < states; s++) {
                double best = LogMath.LOG_ZERO;
                int bestPrev = 0;
                for (int prev = 0; prev < states; prev++) {
                    double cand = delta[t - 1][prev] + logA[prev][s];
                    if (cand > best) {
                        best = cand;
                        bestPrev = prev;
                    }
                }
                delta[t][s] = best + logB[s][obs[t]];
                back[t][s] = bestPrev;
            }
        }
        int[] path = new int[len];
        double best = LogMath.LOG_ZERO;
        for (int s = 0; s < states; s++) {
            if (delta[len - 1][s] > best) {
                best = delta[len - 1][s];
                path[len - 1] = s;
            }
        }
        for (int t = len - 1; t > 0; t--) {
            path[t - 1] = back[t][path[t]];
        }
        return path;
    }

    /** Log probability of the observation sequence together with a specific state path. */
    static double bestPathLogProbability(HmmModel model, int[] obs, int[] path) {
        double[] logPi = model.logPi();
        double[][] logA = model.logA();
        double[][] logB = model.logB();
        double logP = logPi[path[0]] + logB[path[0]][obs[0]];
        for (int t = 1; t < obs.length; t++) {
            logP += logA[path[t - 1]][path[t]] + logB[path[t]][obs[t]];
        }
        return logP;
    }
}
