package com.github.guoyyit.sequencelabelinghmmjava.model;

import com.github.guoyyit.sequencelabelinghmmjava.hmm.HmmModel;
import com.github.guoyyit.sequencelabelinghmmjava.hmm.TrainConfig;

import java.time.Instant;

/**
 * One committed, immutable model version.
 */
public record ModelVersion(
        long version,
        Instant createdAt,
        TrainConfig config,
        HmmModel model,
        int sampleCount,
        int iterationsRun,
        double finalLogLikelihood) {
}
