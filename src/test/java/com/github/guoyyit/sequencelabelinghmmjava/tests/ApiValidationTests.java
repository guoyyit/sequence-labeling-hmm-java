package com.github.guoyyit.sequencelabelinghmmjava.tests;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * HTTP error contract: each invalid input is rejected with a distinguishable
 * error code, never silently replaced by a default (requirement 5).
 */
@SpringBootTest(properties = "hmm.data-dir=${java.io.tmpdir}/hmm-api-test-${random.uuid}")
@AutoConfigureMockMvc
class ApiValidationTests {

    @Autowired
    private MockMvc mvc;

    private static final String VALID_TRAIN_BODY = """
            {
              "numStates": 2,
              "maxIterations": 5,
              "tolerance": 0.0001,
              "seed": 42,
              "stateLabels": ["N", "V"],
              "samples": [
                {"observations": ["dogs", "bark"], "labels": ["N", "V"]},
                {"observations": ["cats", "sleep"], "labels": ["N", "V"]}
              ]
            }
            """;

    private String train(String body) throws Exception {
        MvcResult result = mvc.perform(post("/api/models/train")
                        .contentType(MediaType.APPLICATION_JSON).content(body))
                .andReturn();
        return result.getResponse().getContentAsString();
    }

    @Test
    void invalidLabelIsRejectedWithDistinctCode() throws Exception {
        mvc.perform(post("/api/models/train")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {
                                  "numStates": 2,
                                  "stateLabels": ["N", "V"],
                                  "samples": [
                                    {"observations": ["dogs", "bark"], "labels": ["N", "WRONG"]}
                                  ]
                                }
                                """))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("INVALID_LABEL"));
    }

    @Test
    void mismatchedLabelLengthIsRejectedAsInvalidLabel() throws Exception {
        mvc.perform(post("/api/models/train")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {
                                  "numStates": 2,
                                  "stateLabels": ["N", "V"],
                                  "samples": [
                                    {"observations": ["dogs", "bark"], "labels": ["N"]}
                                  ]
                                }
                                """))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("INVALID_LABEL"));
    }

    @Test
    void missingObservationsFieldIsRejected() throws Exception {
        mvc.perform(post("/api/models/train")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {
                                  "numStates": 2,
                                  "stateLabels": ["N", "V"],
                                  "samples": [
                                    {"labels": ["N", "V"]}
                                  ]
                                }
                                """))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("MISSING_OBSERVATION"));
    }

    @Test
    void emptySequenceIsRejected() throws Exception {
        mvc.perform(post("/api/models/train")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {
                                  "numStates": 2,
                                  "stateLabels": ["N", "V"],
                                  "samples": [
                                    {"observations": [], "labels": []}
                                  ]
                                }
                                """))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("EMPTY_SEQUENCE"));
    }

    @Test
    void invalidConfigIsRejected() throws Exception {
        mvc.perform(post("/api/models/train")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {
                                  "numStates": 0,
                                  "samples": [
                                    {"observations": ["a"], "labels": ["S0"]}
                                  ]
                                }
                                """))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("INVALID_CONFIG"));
    }

    @Test
    void unknownSymbolAtDecodeTimeIsRejected() throws Exception {
        train(VALID_TRAIN_BODY);
        mvc.perform(post("/api/models/decode")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {"observations": ["dogs", "NEVER_SEEN"]}
                                """))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("UNKNOWN_SYMBOL"));
    }

    @Test
    void emptyDecodeSequenceIsRejected() throws Exception {
        train(VALID_TRAIN_BODY);
        mvc.perform(post("/api/models/decode")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {"observations": []}
                                """))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("EMPTY_SEQUENCE"));
    }

    @Test
    void missingDecodeObservationsIsRejected() throws Exception {
        train(VALID_TRAIN_BODY);
        mvc.perform(post("/api/models/decode")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("MISSING_OBSERVATION"));
    }

    @Test
    void unknownModelVersionIsNotFound() throws Exception {
        train(VALID_TRAIN_BODY);
        mvc.perform(post("/api/models/decode")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {"modelVersion": 999, "observations": ["dogs"]}
                                """))
                .andExpect(status().isNotFound())
                .andExpect(jsonPath("$.code").value("MODEL_NOT_FOUND"));

        mvc.perform(post("/api/models/999/activate"))
                .andExpect(status().isNotFound())
                .andExpect(jsonPath("$.code").value("MODEL_NOT_FOUND"));
    }

    @Test
    void malformedBodyIsRejected() throws Exception {
        mvc.perform(post("/api/models/train")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{not json"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("INVALID_CONFIG"));
    }

    @Test
    void fullTrainDecodeActivateRoundTrip() throws Exception {
        String trained = train(VALID_TRAIN_BODY);
        long version = ((Number) com.jayway.jsonpath.JsonPath.read(trained, "$.version")).longValue();

        mvc.perform(post("/api/models/decode")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {"observations": ["dogs", "bark"]}
                                """))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.modelVersion").value(version))
                .andExpect(jsonPath("$.bestPath.length()").value(2))
                .andExpect(jsonPath("$.posteriors.length()").value(2));

        mvc.perform(get("/api/models"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[?(@.version == " + version + ")].active").value(true));

        mvc.perform(post("/api/models/" + version + "/activate"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.active").value(true));
    }

    @Test
    void failedTrainingLeavesNoPartialVersionBehind() throws Exception {
        String before = mvc.perform(get("/api/models")).andReturn()
                .getResponse().getContentAsString();
        int versionsBefore = com.jayway.jsonpath.JsonPath.read(before, "$.length()");

        // One bad sample among good ones: the whole request must be rejected.
        mvc.perform(post("/api/models/train")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {
                                  "numStates": 2,
                                  "stateLabels": ["N", "V"],
                                  "samples": [
                                    {"observations": ["dogs", "bark"], "labels": ["N", "V"]},
                                    {"observations": ["cats", "sleep"], "labels": ["N", "BOGUS"]}
                                  ]
                                }
                                """))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("INVALID_LABEL"));

        String after = mvc.perform(get("/api/models")).andReturn()
                .getResponse().getContentAsString();
        int versionsAfter = com.jayway.jsonpath.JsonPath.read(after, "$.length()");
        assertEquals(versionsBefore, versionsAfter, "a rejected training must not register anything");
    }
}
