package com.github.guoyyit.sequencelabelinghmmjava.api.dto;

import java.util.List;

public record TrainRequest(List<SampleDto> samples, ConfigDto config) {
}
