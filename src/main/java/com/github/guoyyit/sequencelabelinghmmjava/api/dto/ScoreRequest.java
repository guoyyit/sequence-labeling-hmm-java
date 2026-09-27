package com.github.guoyyit.sequencelabelinghmmjava.api.dto;

import java.util.List;

public record ScoreRequest(List<String> observations, Boolean logSpace, String versionId) {
}
