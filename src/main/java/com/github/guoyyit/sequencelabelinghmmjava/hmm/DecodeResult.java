package com.github.guoyyit.sequencelabelinghmmjava.hmm;

import java.util.List;

/**
 * Result of decoding one observation sequence.
 */
public record DecodeResult(
        double logLikelihood,
        double[][] posteriors,
        List<String> bestPath,
        double bestPathLogProbability) {
}
