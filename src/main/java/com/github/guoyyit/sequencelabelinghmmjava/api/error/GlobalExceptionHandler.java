package com.github.guoyyit.sequencelabelinghmmjava.api.error;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

/**
 * Maps domain exceptions to the HTTP error contract:
 * every failure carries a machine-readable {@link ErrorCode} so clients can
 * distinguish invalid labels, missing observations, empty sequences and
 * unknown symbols instead of guessing from status codes alone.
 */
@RestControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(ApiException.class)
    public ResponseEntity<ErrorResponse> handleApi(ApiException ex) {
        HttpStatus status = switch (ex.code()) {
            case INVALID_LABEL, MISSING_OBSERVATION, EMPTY_SEQUENCE, UNKNOWN_SYMBOL, INVALID_CONFIG ->
                    HttpStatus.BAD_REQUEST;
            case MODEL_NOT_FOUND -> HttpStatus.NOT_FOUND;
            case NO_ACTIVE_MODEL -> HttpStatus.CONFLICT;
            case INTERNAL_ERROR -> HttpStatus.INTERNAL_SERVER_ERROR;
        };
        return ResponseEntity.status(status).body(new ErrorResponse(ex.code().name(), ex.getMessage()));
    }

    /** A missing or malformed JSON body is a client error, never a silent default. */
    @ExceptionHandler(HttpMessageNotReadableException.class)
    public ResponseEntity<ErrorResponse> handleUnreadableBody(HttpMessageNotReadableException ex) {
        return ResponseEntity.badRequest()
                .body(new ErrorResponse(ErrorCode.INVALID_CONFIG.name(), "request body is missing or malformed"));
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<ErrorResponse> handleUnexpected(Exception ex) {
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(new ErrorResponse(ErrorCode.INTERNAL_ERROR.name(), "unexpected internal error"));
    }
}
