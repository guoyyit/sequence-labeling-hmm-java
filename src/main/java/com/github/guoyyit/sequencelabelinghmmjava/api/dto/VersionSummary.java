package com.github.guoyyit.sequencelabelinghmmjava.api.dto;

import com.github.guoyyit.sequencelabelinghmmjava.domain.TrainingConfig;
import com.github.guoyyit.sequencelabelinghmmjava.domain.TrainingMetrics;

import java.time.Instant;

/** Version metadata without the (potentially large) parameter matrices. */
public record VersionSummary(
        String versionId,
        Instant createdAt,
        TrainingConfig config,
        TrainingMetrics metrics,
        boolean active) {
}
