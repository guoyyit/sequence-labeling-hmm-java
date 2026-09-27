package com.github.guoyyit.sequencelabelinghmmjava.api;

import com.github.guoyyit.sequencelabelinghmmjava.api.dto.ErrorResponse;
import com.github.guoyyit.sequencelabelinghmmjava.error.ErrorCode;
import com.github.guoyyit.sequencelabelinghmmjava.error.ServiceException;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

import java.time.Instant;
import java.util.EnumMap;
import java.util.Map;

/**
 * Maps {@link ServiceException} error codes to distinguishable HTTP responses:
 * every error body carries a machine-readable {@code code} field.
 */
@RestControllerAdvice
public class GlobalExceptionHandler {

    private static final Map<ErrorCode, HttpStatus> STATUS = new EnumMap<>(ErrorCode.class);

    static {
        STATUS.put(ErrorCode.EMPTY_SEQUENCE, HttpStatus.BAD_REQUEST);
        STATUS.put(ErrorCode.MISSING_OBSERVATION, HttpStatus.BAD_REQUEST);
        STATUS.put(ErrorCode.INVALID_LABEL, HttpStatus.BAD_REQUEST);
        STATUS.put(ErrorCode.UNKNOWN_SYMBOL, HttpStatus.BAD_REQUEST);
        STATUS.put(ErrorCode.INVALID_CONFIG, HttpStatus.BAD_REQUEST);
        STATUS.put(ErrorCode.MODEL_VERSION_NOT_FOUND, HttpStatus.NOT_FOUND);
        STATUS.put(ErrorCode.NO_ACTIVE_MODEL, HttpStatus.CONFLICT);
        STATUS.put(ErrorCode.PERSISTENCE_FAILURE, HttpStatus.INTERNAL_SERVER_ERROR);
    }

    @ExceptionHandler(ServiceException.class)
    public ResponseEntity<ErrorResponse> handleServiceException(ServiceException ex) {
        HttpStatus status = STATUS.getOrDefault(ex.getCode(), HttpStatus.INTERNAL_SERVER_ERROR);
        return ResponseEntity.status(status)
                .body(new ErrorResponse(ex.getCode().name(), ex.getMessage(), Instant.now()));
    }
}
