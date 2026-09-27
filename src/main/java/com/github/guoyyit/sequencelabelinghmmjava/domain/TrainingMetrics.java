package com.github.guoyyit.sequencelabelinghmmjava.domain;

/**
 * Diagnostics produced by a training run.
 */
public record TrainingMetrics(
        int sampleCount,
        int uniqueSampleCount,
        double finalLogLikelihood,
        int iterations,
        boolean converged) {
}
