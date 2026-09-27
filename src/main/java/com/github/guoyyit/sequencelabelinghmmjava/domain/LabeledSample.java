package com.github.guoyyit.sequencelabelinghmmjava.domain;

import java.util.List;

/**
 * One labeled training sample: an observation sequence with its gold label sequence.
 */
public record LabeledSample(List<String> observations, List<String> labels) {
}
