package com.github.guoyyit.sequencelabelinghmmjava.domain;

/**
 * Training hyper-parameters. {@code stateCount == null} means "derive from distinct labels".
 * The same samples plus the same config must always reproduce the same parameters.
 */
public record TrainingConfig(
        Integer stateCount,
        int maxIterations,
        double convergenceThreshold,
        long seed,
        double smoothingAlpha) {

    public static final int DEFAULT_MAX_ITERATIONS = 20;
    public static final double DEFAULT_CONVERGENCE_THRESHOLD = 1e-6;
    public static final long DEFAULT_SEED = 42L;
    public static final double DEFAULT_SMOOTHING_ALPHA = 0.01;

    public static TrainingConfig defaults() {
        return new TrainingConfig(null, DEFAULT_MAX_ITERATIONS, DEFAULT_CONVERGENCE_THRESHOLD,
                DEFAULT_SEED, DEFAULT_SMOOTHING_ALPHA);
    }
}
