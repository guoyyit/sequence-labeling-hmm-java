package com.github.guoyyit.sequencelabelinghmmjava.api;

import com.github.guoyyit.sequencelabelinghmmjava.api.dto.DecodeRequest;
import com.github.guoyyit.sequencelabelinghmmjava.api.dto.DecodeResponse;
import com.github.guoyyit.sequencelabelinghmmjava.api.dto.ModelVersionView;
import com.github.guoyyit.sequencelabelinghmmjava.api.dto.TrainRequest;
import com.github.guoyyit.sequencelabelinghmmjava.service.HmmService;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

/**
 * REST endpoints for the sequence labeling service.
 */
@RestController
@RequestMapping("/api/models")
public class HmmController {

    private final HmmService service;

    public HmmController(HmmService service) {
        this.service = service;
    }

    /** Trains from labeled samples, registers the new version and activates it. */
    @PostMapping("/train")
    @ResponseStatus(HttpStatus.CREATED)
    public ModelVersionView train(@RequestBody TrainRequest request) {
        return service.train(request);
    }

    /** Decodes an observation sequence with the requested or the active version. */
    @PostMapping("/decode")
    public DecodeResponse decode(@RequestBody DecodeRequest request) {
        return service.decode(request);
    }

    /** Lists all committed model versions. */
    @GetMapping
    public List<ModelVersionView> listModels() {
        return service.listVersions();
    }

    /** Atomically switches the active version. */
    @PostMapping("/{version}/activate")
    public ModelVersionView activate(@PathVariable long version) {
        return service.activate(version);
    }
}
