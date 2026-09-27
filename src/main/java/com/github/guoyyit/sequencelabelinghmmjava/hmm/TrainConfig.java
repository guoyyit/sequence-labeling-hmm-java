package com.github.guoyyit.sequencelabelinghmmjava.hmm;

import java.util.List;

/**
 * Immutable training configuration.
 */
public record TrainConfig(
        int numStates,
        int maxIterations,
        double tolerance,
        long seed,
        List<String> stateLabels) {

    public TrainConfig(int numStates, int maxIterations, double tolerance, long seed) {
        this(numStates, maxIterations, tolerance, seed, null);
    }
}
