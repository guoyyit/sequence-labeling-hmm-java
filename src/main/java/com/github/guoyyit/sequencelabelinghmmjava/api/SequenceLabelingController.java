package com.github.guoyyit.sequencelabelinghmmjava.api;

import com.github.guoyyit.sequencelabelinghmmjava.api.dto.ConfigDto;
import com.github.guoyyit.sequencelabelinghmmjava.api.dto.DecodeRequest;
import com.github.guoyyit.sequencelabelinghmmjava.api.dto.SampleDto;
import com.github.guoyyit.sequencelabelinghmmjava.api.dto.ScoreRequest;
import com.github.guoyyit.sequencelabelinghmmjava.api.dto.TrainRequest;
import com.github.guoyyit.sequencelabelinghmmjava.api.dto.VersionSummary;
import com.github.guoyyit.sequencelabelinghmmjava.domain.DecodeResult;
import com.github.guoyyit.sequencelabelinghmmjava.domain.LabeledSample;
import com.github.guoyyit.sequencelabelinghmmjava.domain.ModelVersion;
import com.github.guoyyit.sequencelabelinghmmjava.domain.ScoreResult;
import com.github.guoyyit.sequencelabelinghmmjava.domain.TrainingConfig;
import com.github.guoyyit.sequencelabelinghmmjava.service.SequenceLabelingService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * REST API for training, version management and sequence decoding.
 */
@RestController
@RequestMapping("/api/v1")
public class SequenceLabelingController {

    private final SequenceLabelingService service;

    public SequenceLabelingController(SequenceLabelingService service) {
        this.service = service;
    }

    @PostMapping("/models/train")
    public VersionSummary train(@RequestBody TrainRequest request) {
        List<LabeledSample> samples = new ArrayList<>();
        if (request != null && request.samples() != null) {
            for (SampleDto dto : request.samples()) {
                samples.add(dto == null ? null : new LabeledSample(dto.observations(), dto.labels()));
            }
        }
        ModelVersion version = service.train(samples, toConfig(request == null ? null : request.config()));
        return toSummary(version, false);
    }

    @PostMapping("/models/{versionId}/activate")
    public Map<String, String> activate(@PathVariable String versionId) {
        ModelVersion version = service.activate(versionId);
        return Map.of("activeVersionId", version.versionId());
    }

    @GetMapping("/models")
    public List<VersionSummary> versions() {
        String activeId = service.activeVersionId();
        List<VersionSummary> summaries = new ArrayList<>();
        for (ModelVersion version : service.versions()) {
            summaries.add(toSummary(version, version.versionId().equals(activeId)));
        }
        return summaries;
    }

    @GetMapping("/models/active")
    public VersionSummary active() {
        return toSummary(service.activeVersion(), true);
    }

    @PostMapping("/decode")
    public DecodeResult decode(@RequestBody DecodeRequest request) {
        return service.decode(request == null ? null : request.observations(),
                request == null ? null : request.versionId());
    }

    @PostMapping("/score")
    public ScoreResult score(@RequestBody ScoreRequest request) {
        boolean logSpace = request != null && Boolean.TRUE.equals(request.logSpace());
        return service.score(request == null ? null : request.observations(), logSpace,
                request == null ? null : request.versionId());
    }

    private static TrainingConfig toConfig(ConfigDto dto) {
        if (dto == null) {
            return null;
        }
        TrainingConfig defaults = TrainingConfig.defaults();
        return new TrainingConfig(
                dto.stateCount() != null ? dto.stateCount() : defaults.stateCount(),
                dto.maxIterations() != null ? dto.maxIterations() : defaults.maxIterations(),
                dto.convergenceThreshold() != null ? dto.convergenceThreshold() : defaults.convergenceThreshold(),
                dto.seed() != null ? dto.seed() : defaults.seed(),
                dto.smoothingAlpha() != null ? dto.smoothingAlpha() : defaults.smoothingAlpha());
    }

    private static VersionSummary toSummary(ModelVersion version, boolean active) {
        return new VersionSummary(version.versionId(), version.createdAt(),
                version.config(), version.metrics(), active);
    }
}
