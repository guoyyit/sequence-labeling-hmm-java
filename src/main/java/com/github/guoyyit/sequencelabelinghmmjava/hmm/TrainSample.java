package com.github.guoyyit.sequencelabelinghmmjava.hmm;

import java.util.List;

/**
 * One labeled training sample: an observation sequence with its gold label sequence.
 */
public record TrainSample(List<String> observations, List<String> labels) {
}
