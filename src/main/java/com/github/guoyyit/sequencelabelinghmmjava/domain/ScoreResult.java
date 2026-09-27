package com.github.guoyyit.sequencelabelinghmmjava.domain;

import java.util.List;

/**
 * Per-position state (posterior) probabilities for an observation sequence.
 * {@code probabilities[position][stateIndex]}; values are plain probabilities,
 * or log-probabilities when {@code logSpace} is true.
 */
public record ScoreResult(
        double logLikelihood,
        double[][] probabilities,
        boolean logSpace,
        List<String> labels,
        String versionId) {
}
