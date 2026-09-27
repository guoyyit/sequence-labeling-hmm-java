package com.github.guoyyit.sequencelabelinghmmjava.hmm;

import com.github.guoyyit.sequencelabelinghmmjava.domain.HmmParameters;

/**
 * Log-space HMM inference: Viterbi decoding and forward-backward posteriors.
 * All computations stay in log space, so arbitrarily long sequences neither
 * overflow nor underflow. Ties are broken by lowest state index, which keeps
 * results deterministic.
 */
public final class HmmInference {

    private HmmInference() {
    }

    /** Result of the forward-backward pass, all matrices in log space. */
    public record ForwardBackwardResult(
            double[][] logAlpha,
            double[][] logBeta,
            double[][] logGamma,
            double logLikelihood) {
    }

    /** Viterbi outcome: optimal state-index path and its log probability. */
    public record ViterbiResult(int[] statePath, double logProbability) {
    }

    /**
     * Viterbi decoding in log space.
     *
     * @param observations symbol indices, length >= 1
     */
    public static ViterbiResult viterbi(HmmParameters params, int[] observations) {
        int states = params.getStateCount();
        int length = observations.length;
        double[] logPi = params.getLogInitial();
        double[][] logA = params.getLogTransition();
        double[][] logB = params.getLogEmission();

        double[][] delta = new double[length][states];
        int[][] back = new int[length][states];

        int first = observations[0];
        for (int i = 0; i < states; i++) {
            delta[0][i] = logPi[i] + logB[i][first];
        }
        for (int t = 1; t < length; t++) {
            int obs = observations[t];
            for (int j = 0; j < states; j++) {
                double best = HmmMath.LOG_ZERO;
                int bestPrev = 0;
                for (int i = 0; i < states; i++) {
                    double candidate = delta[t - 1][i] + logA[i][j];
                    // Strict '>' keeps the lowest index on ties: deterministic.
                    if (candidate > best) {
                        best = candidate;
                        bestPrev = i;
                    }
                }
                delta[t][j] = best + logB[j][obs];
                back[t][j] = bestPrev;
            }
        }

        double bestFinal = HmmMath.LOG_ZERO;
        int lastState = 0;
        for (int i = 0; i < states; i++) {
            if (delta[length - 1][i] > bestFinal) {
                bestFinal = delta[length - 1][i];
                lastState = i;
            }
        }

        int[] path = new int[length];
        path[length - 1] = lastState;
        for (int t = length - 1; t > 0; t--) {
            path[t - 1] = back[t][path[t]];
        }
        return new ViterbiResult(path, bestFinal);
    }

    /**
     * Forward-backward pass in log space.
     *
     * @return log alpha, log beta, log posterior gamma and the log-likelihood
     */
    public static ForwardBackwardResult forwardBackward(HmmParameters params, int[] observations) {
        int states = params.getStateCount();
        int length = observations.length;
        double[] logPi = params.getLogInitial();
        double[][] logA = params.getLogTransition();
        double[][] logB = params.getLogEmission();

        double[][] logAlpha = new double[length][states];
        double[][] logBeta = new double[length][states];

        int first = observations[0];
        for (int i = 0; i < states; i++) {
            logAlpha[0][i] = logPi[i] + logB[i][first];
        }
        double[] column = new double[states];
        for (int t = 1; t < length; t++) {
            int obs = observations[t];
            for (int j = 0; j < states; j++) {
                for (int i = 0; i < states; i++) {
                    column[i] = logAlpha[t - 1][i] + logA[i][j];
                }
                logAlpha[t][j] = HmmMath.logSumExp(column) + logB[j][obs];
            }
        }

        double logLikelihood = HmmMath.logSumExp(logAlpha[length - 1]);

        for (int t = length - 2; t >= 0; t--) {
            int nextObs = observations[t + 1];
            for (int i = 0; i < states; i++) {
                for (int j = 0; j < states; j++) {
                    column[j] = logA[i][j] + logB[j][nextObs] + logBeta[t + 1][j];
                }
                logBeta[t][i] = HmmMath.logSumExp(column);
            }
        }

        double[][] logGamma = new double[length][states];
        for (int t = 0; t < length; t++) {
            for (int i = 0; i < states; i++) {
                logGamma[t][i] = logAlpha[t][i] + logBeta[t][i] - logLikelihood;
            }
        }
        return new ForwardBackwardResult(logAlpha, logBeta, logGamma, logLikelihood);
    }

    /** Log-likelihood of the observation sequence under the model. */
    public static double forwardLogLikelihood(HmmParameters params, int[] observations) {
        int states = params.getStateCount();
        double[] logPi = params.getLogInitial();
        double[][] logA = params.getLogTransition();
        double[][] logB = params.getLogEmission();

        double[] prev = new double[states];
        double[] curr = new double[states];
        double[] column = new double[states];
        int first = observations[0];
        for (int i = 0; i < states; i++) {
            prev[i] = logPi[i] + logB[i][first];
        }
        for (int t = 1; t < observations.length; t++) {
            int obs = observations[t];
            for (int j = 0; j < states; j++) {
                for (int i = 0; i < states; i++) {
                    column[i] = prev[i] + logA[i][j];
                }
                curr[j] = HmmMath.logSumExp(column) + logB[j][obs];
            }
            double[] tmp = prev;
            prev = curr;
            curr = tmp;
        }
        return HmmMath.logSumExp(prev);
    }
}
