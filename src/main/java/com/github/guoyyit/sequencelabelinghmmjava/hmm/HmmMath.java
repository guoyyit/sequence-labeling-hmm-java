package com.github.guoyyit.sequencelabelinghmmjava.hmm;

/**
 * Log-space numeric helpers. All methods use {@link StrictMath} so results are
 * bit-for-bit reproducible across platforms and JVMs.
 */
public final class HmmMath {

    /** Log of zero probability. */
    public static final double LOG_ZERO = Double.NEGATIVE_INFINITY;

    private HmmMath() {
    }

    /** log(exp(a) + exp(b)), numerically stable for very small magnitudes. */
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
        return a + StrictMath.log1p(StrictMath.exp(b - a));
    }

    /** log(sum(exp(values))), numerically stable for very small magnitudes. */
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
                sum += StrictMath.exp(v - max);
            }
        }
        return max + StrictMath.log(sum);
    }
}
