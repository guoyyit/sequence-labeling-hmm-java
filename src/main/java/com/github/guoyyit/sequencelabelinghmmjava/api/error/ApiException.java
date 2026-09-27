package com.github.guoyyit.sequencelabelinghmmjava.api.error;

/**
 * Domain exception carrying a machine-readable error code.
 */
public class ApiException extends RuntimeException {

    private final ErrorCode code;

    public ApiException(ErrorCode code, String message) {
        super(message);
        this.code = code;
    }

    public ErrorCode code() {
        return code;
    }
}
