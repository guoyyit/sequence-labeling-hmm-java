package com.github.guoyyit.sequencelabelinghmmjava.hmm;

/**
 * Numerically stable log-space arithmetic helpers.
 *
 * <p>All probabilities in this service are carried as natural logarithms so that
 * long observation sequences cannot cause floating-point underflow or overflow.</p>
 */
public final class LogMath {

    /** log(0): the additive identity in log space. */
    public static final double LOG_ZERO = Double.NEGATIVE_INFINITY;

    private LogMath() {
    }

    /**
     * Computes {@code log(exp(a) + exp(b))} without materializing exp(a) or exp(b).
     */
    public static double logAdd(double a, double b) {
        if (a == LOG_ZERO) {
            return b;
        }
        if (b == LOG_ZERO) {
            return a;
        }
        if (a < b) {
            double t = a;
            a = b;
            b = t;
        }
        // a >= b here; exp(b - a) is in (0, 1], no overflow possible.
        return a + Math.log1p(Math.exp(b - a));
    }

    /**
     * Computes {@code log(sum_i exp(values[i]))} with the max-shift trick.
     * Returns {@link #LOG_ZERO} for an empty array or all-zero-probability input.
     */
    public static double logSumExp(double[] values) {
        double max = LOG_ZERO;
        for (double v : values) {
            if (v > max) {
                max = v;
            }
        }
        if (max == LOG_ZERO) {
            return LOG_ZERO;
        }
        double sum = 0.0;
        for (double v : values) {
            if (v != LOG_ZERO) {
                sum += Math.exp(v - max);
            }
        }
        return max + Math.log(sum);
    }

    /**
     * Normalizes log-space weights in place so that {@code logSumExp(values) == 0}.
     * An all-LOG_ZERO row is left untouched (degenerate zero distribution).
     */
    public static void logNormalizeInPlace(double[] values) {
        double z = logSumExp(values);
        if (z == LOG_ZERO) {
            return;
        }
        for (int i = 0; i < values.length; i++) {
            values[i] -= z;
        }
    }
}
