package com.github.guoyyit.sequencelabelinghmmjava.tests;

import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.test.web.servlet.MockMvc;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.hamcrest.Matchers.hasSize;
import static org.hamcrest.Matchers.lessThanOrEqualTo;
import static org.hamcrest.Matchers.notNullValue;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * End-to-end HTTP contract: training, activation, decoding and the
 * machine-distinguishable error responses.
 */
@SpringBootTest
@AutoConfigureMockMvc
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class ApiContractTest {

    private static final Path DATA_DIR = createDataDir();

    private static Path createDataDir() {
        try {
            return Files.createTempDirectory("hmm-api-contract-test");
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    @DynamicPropertySource
    static void properties(DynamicPropertyRegistry registry) {
        registry.add("hmm.data-dir", () -> DATA_DIR.toString());
    }

    @Autowired
    private MockMvc mvc;

    @Test
    @Order(1)
    void trainRegistersVersion() throws Exception {
        mvc.perform(post("/api/v1/models/train")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {"samples":[
                                  {"observations":["the","cat","sat"],"labels":["D","N","V"]},
                                  {"observations":["the","dog","ran"],"labels":["D","N","V"]},
                                  {"observations":["a","cat","ran"],"labels":["D","N","V"]}
                                ],
                                "config":{"maxIterations":10,"seed":7}}
                                """))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.versionId").value("v1"))
                .andExpect(jsonPath("$.metrics.sampleCount").value(3))
                .andExpect(jsonPath("$.active").value(false));
    }

    @Test
    @Order(2)
    void activateSwitchesActiveVersion() throws Exception {
        mvc.perform(post("/api/v1/models/v1/activate"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.activeVersionId").value("v1"));

        mvc.perform(get("/api/v1/models"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$", hasSize(1)))
                .andExpect(jsonPath("$[0].active").value(true));

        mvc.perform(get("/api/v1/models/active"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.versionId").value("v1"));
    }

    @Test
    @Order(3)
    void decodeAndScoreReturnResults() throws Exception {
        mvc.perform(post("/api/v1/decode")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"observations\":[\"the\",\"cat\",\"sat\"]}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.path", hasSize(3)))
                .andExpect(jsonPath("$.path[0]").value("D"))
                .andExpect(jsonPath("$.logProbability", lessThanOrEqualTo(0.0)))
                .andExpect(jsonPath("$.versionId").value("v1"));

        mvc.perform(post("/api/v1/score")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"observations\":[\"the\",\"cat\"],\"logSpace\":true}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.logLikelihood", notNullValue()))
                .andExpect(jsonPath("$.probabilities", hasSize(2)))
                .andExpect(jsonPath("$.labels", hasSize(3)));
    }

    @Test
    @Order(4)
    void errorContractIsDistinguishable() throws Exception {
        // Unknown observation symbol.
        mvc.perform(post("/api/v1/decode")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"observations\":[\"the\",\"alien\"]}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("UNKNOWN_SYMBOL"));

        // Empty sequence.
        mvc.perform(post("/api/v1/decode")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"observations\":[]}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("EMPTY_SEQUENCE"));

        // Missing observation element.
        mvc.perform(post("/api/v1/decode")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"observations\":[\"the\",null]}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("MISSING_OBSERVATION"));

        // Invalid label (count mismatch) in training.
        mvc.perform(post("/api/v1/models/train")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {"samples":[{"observations":["the","cat"],"labels":["D"]}]}
                                """))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("INVALID_LABEL"));

        // Invalid config: stateCount below the distinct label count.
        mvc.perform(post("/api/v1/models/train")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {"samples":[{"observations":["the","cat"],"labels":["D","N"]}],
                                 "config":{"stateCount":1}}
                                """))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("INVALID_CONFIG"));

        // Unknown version.
        mvc.perform(post("/api/v1/models/v999/activate"))
                .andExpect(status().isNotFound())
                .andExpect(jsonPath("$.code").value("MODEL_VERSION_NOT_FOUND"));
    }
}
