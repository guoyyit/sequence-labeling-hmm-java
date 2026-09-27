package com.github.guoyyit.sequencelabelinghmmjava.domain;

import java.time.Instant;

/**
 * An immutable, committed model version. Readers only ever see fully
 * constructed instances of this type.
 */
public record ModelVersion(
        String versionId,
        Instant createdAt,
        TrainingConfig config,
        HmmParameters parameters,
        TrainingMetrics metrics) {
}
