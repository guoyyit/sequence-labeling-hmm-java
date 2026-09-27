package com.github.guoyyit.sequencelabelinghmmjava.api.dto;

import java.time.Instant;

public record ErrorResponse(String code, String message, Instant timestamp) {
}
