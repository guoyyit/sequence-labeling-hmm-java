package com.github.guoyyit.sequencelabelinghmmjava.api.dto;

/** Nullable fields fall back to {@link com.github.guoyyit.sequencelabelinghmmjava.domain.TrainingConfig} defaults. */
public record ConfigDto(
        Integer stateCount,
        Integer maxIterations,
        Double convergenceThreshold,
        Long seed,
        Double smoothingAlpha) {
}
