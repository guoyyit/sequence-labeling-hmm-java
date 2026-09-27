package com.github.guoyyit.sequencelabelinghmmjava.api.dto;

import com.github.guoyyit.sequencelabelinghmmjava.hmm.TrainSample;

import java.util.List;

/**
 * Training request payload.
 */
public record TrainRequest(
        Integer numStates,
        Integer maxIterations,
        Double tolerance,
        Long seed,
        List<String> stateLabels,
        List<TrainSample> samples) {
}
