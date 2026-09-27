package com.github.guoyyit.sequencelabelinghmmjava.api.dto;

import java.util.List;

/**
 * Decoding response payload.
 */
public record DecodeResponse(
        long modelVersion,
        double logLikelihood,
        double[][] posteriors,
        List<String> bestPath,
        double bestPathLogProbability) {
}
