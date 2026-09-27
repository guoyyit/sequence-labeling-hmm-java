package com.github.guoyyit.sequencelabelinghmmjava.domain;

import java.util.List;

/**
 * Viterbi decoding outcome: the most likely label path and its log probability.
 */
public record DecodeResult(List<String> path, double logProbability, String versionId) {
}
