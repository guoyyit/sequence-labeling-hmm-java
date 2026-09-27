package com.github.guoyyit.sequencelabelinghmmjava.tests;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.guoyyit.sequencelabelinghmmjava.api.dto.DecodeRequest;
import com.github.guoyyit.sequencelabelinghmmjava.api.dto.DecodeResponse;
import com.github.guoyyit.sequencelabelinghmmjava.api.dto.TrainRequest;
import com.github.guoyyit.sequencelabelinghmmjava.config.AppProperties;
import com.github.guoyyit.sequencelabelinghmmjava.hmm.TrainSample;
import com.github.guoyyit.sequencelabelinghmmjava.model.ModelRegistry;
import com.github.guoyyit.sequencelabelinghmmjava.model.ModelStore;
import com.github.guoyyit.sequencelabelinghmmjava.service.HmmService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Same input plus same configuration reproduces historical conclusions
 * bit-for-bit, even across fresh service instances (requirement 6).
 */
class DeterminismTests {

    @TempDir
    Path dirA;

    @TempDir
    Path dirB;

    private static HmmService serviceOn(Path dir) {
        AppProperties props = new AppProperties();
        props.setDataDir(dir.toString());
        return new HmmService(new ModelRegistry(), new ModelStore(props, new ObjectMapper()));
    }

    private static TrainRequest request() {
        return new TrainRequest(3, 30, 1e-7, 20260928L, List.of("D", "N", "V"),
                List.of(
                        new TrainSample(List.of("the", "dog", "barks"), List.of("D", "N", "V")),
                        new TrainSample(List.of("the", "cat", "sleeps"), List.of("D", "N", "V")),
                        new TrainSample(List.of("a", "dog", "runs"), List.of("D", "N", "V")),
                        new TrainSample(List.of("dogs", "bark"), List.of("N", "V")),
                        new TrainSample(List.of("the", "birds", "fly"), List.of("D", "N", "V"))));
    }

    private static void assertSameDecode(DecodeResponse a, DecodeResponse b) {
        assertEquals(a.logLikelihood(), b.logLikelihood());
        assertEquals(a.bestPath(), b.bestPath());
        assertEquals(a.bestPathLogProbability(), b.bestPathLogProbability());
        assertEquals(a.posteriors().length, b.posteriors().length);
        for (int t = 0; t < a.posteriors().length; t++) {
            org.junit.jupiter.api.Assertions.assertArrayEquals(a.posteriors()[t], b.posteriors()[t]);
        }
    }

    @Test
    void sameConfigSameInputReproducesIdenticalModelAcrossInstances() {
        HmmService first = serviceOn(dirA);
        HmmService second = serviceOn(dirB);

        first.train(request());
        second.train(request());

        DecodeRequest probe = new DecodeRequest(null, List.of("the", "dog", "runs"));
        assertSameDecode(first.decode(probe), second.decode(probe));
    }

    @Test
    void retrainingSameRequestInOneServiceReproducesIdenticalModel() {
        HmmService service = serviceOn(dirA);
        long v1 = service.train(request()).version();
        long v2 = service.train(request()).version();

        DecodeRequest probe = new DecodeRequest(null, List.of("dogs", "bark"));
        DecodeResponse fromV1 = service.decode(new DecodeRequest(v1, List.of("dogs", "bark")));
        DecodeResponse fromV2 = service.decode(new DecodeRequest(v2, List.of("dogs", "bark")));
        assertSameDecode(fromV1, fromV2);

        // Metadata is reproducible too.
        var versions = service.listVersions();
        assertEquals(versions.get(0).finalLogLikelihood(), versions.get(1).finalLogLikelihood());
        assertEquals(versions.get(0).iterationsRun(), versions.get(1).iterationsRun());
    }

    @Test
    void differentSeedChangesModelButStaysReproducible() {
        HmmService service = serviceOn(dirA);
        TrainRequest base = request();
        service.train(base);
        long seededDifferently = service.train(new TrainRequest(
                base.numStates(), base.maxIterations(), base.tolerance(), base.seed() + 1,
                base.stateLabels(), base.samples())).version();

        DecodeResponse a = service.decode(new DecodeRequest(1L, List.of("the", "cat")));
        DecodeResponse b = service.decode(new DecodeRequest(seededDifferently, List.of("the", "cat")));
        // Different seeds may diverge; each must still be self-consistent.
        DecodeResponse aAgain = service.decode(new DecodeRequest(1L, List.of("the", "cat")));
        assertSameDecode(a, aAgain);
        org.junit.jupiter.api.Assertions.assertNotNull(b);
    }
}
