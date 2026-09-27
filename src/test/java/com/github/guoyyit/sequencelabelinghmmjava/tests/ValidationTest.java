package com.github.guoyyit.sequencelabelinghmmjava.tests;

import com.github.guoyyit.sequencelabelinghmmjava.domain.LabeledSample;
import com.github.guoyyit.sequencelabelinghmmjava.domain.ScoreResult;
import com.github.guoyyit.sequencelabelinghmmjava.domain.TrainingConfig;
import com.github.guoyyit.sequencelabelinghmmjava.error.ErrorCode;
import com.github.guoyyit.sequencelabelinghmmjava.error.ServiceException;
import com.github.guoyyit.sequencelabelinghmmjava.registry.ModelRegistry;
import com.github.guoyyit.sequencelabelinghmmjava.registry.ModelStore;
import com.github.guoyyit.sequencelabelinghmmjava.service.SequenceLabelingService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Input validation and the error contract: illegal labels, missing
 * observations, empty sequences and unknown symbols are rejected with
 * distinguishable error codes; nothing is silently defaulted.
 */
class ValidationTest {

    @TempDir
    Path dataDir;

    private SequenceLabelingService newService() {
        return new SequenceLabelingService(new ModelRegistry(new ModelStore(dataDir)));
    }

    private static List<LabeledSample> oneGoodSample() {
        return List.of(new LabeledSample(List.of("a", "b"), List.of("X", "Y")));
    }

    private static void assertError(ErrorCode expected, Runnable action) {
        ServiceException ex = assertThrows(ServiceException.class, action::run);
        assertEquals(expected, ex.getCode(),
                "expected " + expected + " but got " + ex.getCode() + ": " + ex.getMessage());
    }

    @Test
    void trainingRejectsInvalidSamples() {
        SequenceLabelingService service = newService();

        assertError(ErrorCode.EMPTY_SEQUENCE, () -> service.train(List.of(), null));
        assertError(ErrorCode.MISSING_OBSERVATION,
                () -> service.train(List.of(new LabeledSample(null, List.of("X"))), null));
        assertError(ErrorCode.EMPTY_SEQUENCE,
                () -> service.train(List.of(new LabeledSample(List.of(), List.of())), null));
        assertError(ErrorCode.MISSING_OBSERVATION,
                () -> service.train(List.of(new LabeledSample(
                        Arrays.asList("a", null), List.of("X", "Y"))), null));
        assertError(ErrorCode.MISSING_OBSERVATION,
                () -> service.train(List.of(new LabeledSample(List.of("a", " "), List.of("X", "Y"))), null));
        assertError(ErrorCode.INVALID_LABEL,
                () -> service.train(List.of(new LabeledSample(List.of("a", "b"), List.of("X"))), null));
        assertError(ErrorCode.INVALID_LABEL,
                () -> service.train(List.of(new LabeledSample(List.of("a", "b"), null)), null));
        assertError(ErrorCode.INVALID_LABEL,
                () -> service.train(List.of(new LabeledSample(
                        Arrays.asList("a", "b"), Arrays.asList("X", ""))), null));
    }

    @Test
    void failedTrainingLeavesNoPartialUpdate() {
        SequenceLabelingService service = newService();
        List<LabeledSample> batch = new ArrayList<>(oneGoodSample());
        batch.add(new LabeledSample(List.of("a", "b"), List.of("X"))); // invalid

        assertError(ErrorCode.INVALID_LABEL, () -> service.train(batch, null));
        assertEquals(0, service.versions().size(), "failed batch must not commit any version");

        // The registry is still fully usable afterwards.
        service.train(oneGoodSample(), TrainingConfig.defaults());
        assertEquals(1, service.versions().size());
    }

    @Test
    void decodeValidatesInputAndModelState() {
        SequenceLabelingService service = newService();

        // No model registered/active yet.
        assertError(ErrorCode.NO_ACTIVE_MODEL, () -> service.decode(List.of("a"), null));
        assertError(ErrorCode.MODEL_VERSION_NOT_FOUND, () -> service.decode(List.of("a"), "v99"));
        assertError(ErrorCode.MODEL_VERSION_NOT_FOUND, () -> service.activate("v99"));

        var version = service.train(oneGoodSample(), null);
        // Registered but not yet activated.
        assertError(ErrorCode.NO_ACTIVE_MODEL, () -> service.decode(List.of("a"), null));

        service.activate(version.versionId());
        assertError(ErrorCode.EMPTY_SEQUENCE, () -> service.decode(List.of(), null));
        assertError(ErrorCode.MISSING_OBSERVATION, () -> service.decode(null, null));
        assertError(ErrorCode.MISSING_OBSERVATION,
                () -> service.decode(Arrays.asList("a", null), null));
        assertError(ErrorCode.UNKNOWN_SYMBOL, () -> service.decode(List.of("a", "zzz"), null));
        assertError(ErrorCode.UNKNOWN_SYMBOL, () -> service.score(List.of("???"), false, null));
    }

    @Test
    void scoreReturnsProbabilitiesOrLogSpace() {
        SequenceLabelingService service = newService();
        var version = service.train(List.of(
                new LabeledSample(List.of("a", "b", "a"), List.of("X", "Y", "X")),
                new LabeledSample(List.of("b", "a"), List.of("Y", "X"))), null);
        service.activate(version.versionId());

        ScoreResult plain = service.score(List.of("a", "b"), false, null);
        assertEquals(2, plain.probabilities().length);
        for (double[] row : plain.probabilities()) {
            double sum = 0.0;
            for (double p : row) {
                assertTrue(p >= 0.0 && p <= 1.0, "probability out of range: " + p);
                sum += p;
            }
            assertEquals(1.0, sum, 1e-9);
        }

        ScoreResult logged = service.score(List.of("a", "b"), true, null);
        for (double[] row : logged.probabilities()) {
            for (double logP : row) {
                assertTrue(logP <= 0.0 && Double.isFinite(logP),
                        "log probability must be finite and <= 0: " + logP);
            }
        }
        assertEquals(plain.logLikelihood(), logged.logLikelihood(), 0.0);
        assertTrue(Double.isFinite(plain.logLikelihood()));
    }
}
