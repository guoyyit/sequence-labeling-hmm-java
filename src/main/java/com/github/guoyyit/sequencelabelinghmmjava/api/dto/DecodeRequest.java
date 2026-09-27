package com.github.guoyyit.sequencelabelinghmmjava.api.dto;

import java.util.List;

/** {@code versionId} null means "use the active version". */
public record DecodeRequest(List<String> observations, String versionId) {
}
