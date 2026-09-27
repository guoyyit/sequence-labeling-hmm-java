package com.github.guoyyit.sequencelabelinghmmjava.error;

/**
 * Machine-distinguishable error codes for the service error contract.
 */
public enum ErrorCode {
    EMPTY_SEQUENCE,
    MISSING_OBSERVATION,
    INVALID_LABEL,
    UNKNOWN_SYMBOL,
    INVALID_CONFIG,
    MODEL_VERSION_NOT_FOUND,
    NO_ACTIVE_MODEL,
    PERSISTENCE_FAILURE
}
