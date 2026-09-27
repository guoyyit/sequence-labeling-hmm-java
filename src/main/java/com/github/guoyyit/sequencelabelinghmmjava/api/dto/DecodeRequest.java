package com.github.guoyyit.sequencelabelinghmmjava.api.dto;

import java.util.List;

/**
 * Decoding request payload.
 */
public record DecodeRequest(Long modelVersion, List<String> observations) {
}
