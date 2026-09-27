package com.github.guoyyit.sequencelabelinghmmjava.tests;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.guoyyit.sequencelabelinghmmjava.api.dto.DecodeRequest;
import com.github.guoyyit.sequencelabelinghmmjava.api.dto.DecodeResponse;
import com.github.guoyyit.sequencelabelinghmmjava.api.dto.ModelVersionView;
import com.github.guoyyit.sequencelabelinghmmjava.api.dto.TrainRequest;
import com.github.guoyyit.sequencelabelinghmmjava.config.AppProperties;
import com.github.guoyyit.sequencelabelinghmmjava.hmm.TrainSample;
import com.github.guoyyit.sequencelabelinghmmjava.model.ModelRegistry;
import com.github.guoyyit.sequencelabelinghmmjava.model.ModelStore;
import com.github.guoyyit.sequencelabelinghmmjava.service.HmmService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Restart recovery: committed versions survive a process restart (requirement 4).
 */
class PersistenceTests {

    @TempDir
    Path dataDir;

    private HmmService newService() {
        AppProperties props = new AppProperties();
        props.setDataDir(dataDir.toString());
        return new HmmService(new ModelRegistry(), new ModelStore(props, new ObjectMapper()));
    }

    private static TrainRequest trainRequest(long seed) {
        return new TrainRequest(2, 10, 1e-5, seed, List.of("N", "V"),
                List.of(
                        new TrainSample(List.of("dogs", "bark"), List.of("N", "V")),
                        new TrainSample(List.of("cats", "sleep"), List.of("N", "V")),
                        new TrainSample(List.of("birds", "fly"), List.of("N", "V"))));
    }

    @Test
    void committedVersionsSurviveRestart() {
        HmmService first = newService();
        ModelVersionView v1 = first.train(trainRequest(1L));
        ModelVersionView v2 = first.train(trainRequest(2L));
        assertEquals(1L, v1.version());
        assertEquals(2L, v2.version());
        assertTrue(v2.active(), "newly trained version is auto-activated");

        // Simulate a restart: brand-new service instances over the same directory.
        HmmService restarted = newService();
        List<ModelVersionView> versions = restarted.listVersions();
        assertEquals(2, versions.size());
        assertEquals(2L, versions.get(1).version());
        assertTrue(versions.get(1).active(), "active pointer must be restored");
        assertEquals(v1.finalLogLikelihood(), versions.get(0).finalLogLikelihood());
    }

    @Test
    void decodeAfterRestartMatchesDecodeBeforeRestart() {
        HmmService first = newService();
        first.train(trainRequest(7L));
        DecodeResponse before = first.decode(new DecodeRequest(null, List.of("dogs", "bark")));

        HmmService restarted = newService();
        DecodeResponse after = restarted.decode(new DecodeRequest(null, List.of("dogs", "bark")));

        assertEquals(before.modelVersion(), after.modelVersion());
        assertEquals(before.logLikelihood(), after.logLikelihood());
        assertEquals(before.bestPath(), after.bestPath());
        assertEquals(before.bestPathLogProbability(), after.bestPathLogProbability());
    }

    @Test
    void activationChoiceSurvivesRestart() {
        HmmService first = newService();
        first.train(trainRequest(1L));
        first.train(trainRequest(2L));
        first.activate(1L);

        HmmService restarted = newService();
        List<ModelVersionView> versions = restarted.listVersions();
        assertTrue(versions.get(0).active(), "the explicitly activated version must stay active");
        assertEquals(1L, restarted.decode(new DecodeRequest(null, List.of("dogs"))).modelVersion());
    }

    @Test
    void versionFilesAreWrittenAtomicallyWithoutLeftoverTempFiles() throws Exception {
        HmmService service = newService();
        service.train(trainRequest(3L));
        service.train(trainRequest(4L));

        Path models = dataDir.resolve("models");
        assertTrue(Files.isDirectory(models));
        try (var files = Files.list(models)) {
            List<String> names = files.map(p -> p.getFileName().toString()).sorted().toList();
            assertEquals(List.of("v1.json", "v2.json"), names, "no .tmp files may remain");
        }
        assertTrue(Files.exists(dataDir.resolve("registry.json")));

        // Version ids keep increasing across restarts (no reuse after reload).
        HmmService restarted = newService();
        assertEquals(3L, restarted.train(trainRequest(5L)).version());
    }
}
