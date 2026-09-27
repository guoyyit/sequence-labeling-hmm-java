package com.github.guoyyit.sequencelabelinghmmjava.api.error;

/**
 * Serialized error contract returned to clients.
 */
public record ErrorResponse(String code, String message) {
}
