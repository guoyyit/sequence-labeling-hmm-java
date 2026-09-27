package com.github.guoyyit.sequencelabelinghmmjava.api.error;

/**
 * Machine-distinguishable error codes for the API error contract.
 */
public enum ErrorCode {
    INVALID_LABEL,
    MISSING_OBSERVATION,
    EMPTY_SEQUENCE,
    UNKNOWN_SYMBOL,
    INVALID_CONFIG,
    MODEL_NOT_FOUND,
    NO_ACTIVE_MODEL,
    INTERNAL_ERROR
}
