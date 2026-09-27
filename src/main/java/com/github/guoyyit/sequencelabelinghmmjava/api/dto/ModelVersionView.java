package com.github.guoyyit.sequencelabelinghmmjava.api.dto;

import java.time.Instant;

/**
 * Read-only view of a registered model version.
 */
public record ModelVersionView(
        long version,
        boolean active,
        Instant createdAt,
        int numStates,
        int sampleCount,
        int iterationsRun,
        double finalLogLikelihood) {
}
