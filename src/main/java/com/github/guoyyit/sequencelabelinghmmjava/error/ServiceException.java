package com.github.guoyyit.sequencelabelinghmmjava.error;

/**
 * Business exception carrying a machine-distinguishable {@link ErrorCode}.
 */
public class ServiceException extends RuntimeException {

    private final ErrorCode code;

    public ServiceException(ErrorCode code, String message) {
        super(message);
        this.code = code;
    }

    public ServiceException(ErrorCode code, String message, Throwable cause) {
        super(message, cause);
        this.code = code;
    }

    public ErrorCode getCode() {
        return code;
    }
}
