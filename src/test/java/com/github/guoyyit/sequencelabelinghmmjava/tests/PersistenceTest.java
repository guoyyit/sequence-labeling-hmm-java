package com.github.guoyyit.sequencelabelinghmmjava.tests;

import com.github.guoyyit.sequencelabelinghmmjava.domain.DecodeResult;
import com.github.guoyyit.sequencelabelinghmmjava.domain.LabeledSample;
import com.github.guoyyit.sequencelabelinghmmjava.domain.ModelVersion;
import com.github.guoyyit.sequencelabelinghmmjava.domain.TrainingConfig;
import com.github.guoyyit.sequencelabelinghmmjava.registry.ModelRegistry;
import com.github.guoyyit.sequencelabelinghmmjava.registry.ModelStore;
import com.github.guoyyit.sequencelabelinghmmjava.service.SequenceLabelingService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Registered versions must survive a restart: a fresh registry built on the
 * same data directory reloads every version and the active pointer.
 */
class PersistenceTest {

    @TempDir
    Path dataDir;

    private SequenceLabelingService newService() {
        return new SequenceLabelingService(new ModelRegistry(new ModelStore(dataDir)));
    }

    private static List<LabeledSample> corpus() {
        return List.of(
                new LabeledSample(List.of("a", "b", "a"), List.of("X", "Y", "X")),
                new LabeledSample(List.of("b", "a", "b"), List.of("Y", "X", "Y")),
                new LabeledSample(List.of("a", "a", "b"), List.of("X", "X", "Y")));
    }

    @Test
    void registeredVersionsSurviveRestart() {
        SequenceLabelingService first = newService();
        ModelVersion v1 = first.train(corpus(), TrainingConfig.defaults());
        ModelVersion v2 = first.train(corpus(), new TrainingConfig(3, 5, 1e-6, 9L, 0.01));
        first.activate(v2.versionId());
        List<String> probe = List.of("a", "b", "a", "b");
        DecodeResult beforeRestart = first.decode(probe, null);

        // Files are really on disk.
        assertTrue(Files.exists(dataDir.resolve("registry.json")));
        assertTrue(Files.exists(dataDir.resolve("versions").resolve(v1.versionId() + ".json")));
        assertTrue(Files.exists(dataDir.resolve("versions").resolve(v2.versionId() + ".json")));

        // Simulate a restart: brand-new store/registry/service on the same dir.
        SequenceLabelingService restarted = newService();
        List<ModelVersion> versions = restarted.versions();
        assertEquals(List.of(v1.versionId(), v2.versionId()),
                versions.stream().map(ModelVersion::versionId).toList());
        assertEquals(v2.versionId(), restarted.activeVersion().versionId());

        // Parameters round-trip exactly.
        assertArrayEquals(v2.parameters().getLogInitial(),
                restarted.activeVersion().parameters().getLogInitial(), 0.0);
        for (int i = 0; i < v2.parameters().getStateCount(); i++) {
            assertArrayEquals(v2.parameters().getLogTransition()[i],
                    restarted.activeVersion().parameters().getLogTransition()[i], 0.0);
            assertArrayEquals(v2.parameters().getLogEmission()[i],
                    restarted.activeVersion().parameters().getLogEmission()[i], 0.0);
        }

        // Decoding after restart reproduces the pre-restart result exactly.
        DecodeResult afterRestart = restarted.decode(probe, null);
        assertEquals(beforeRestart.path(), afterRestart.path());
        assertEquals(beforeRestart.logProbability(), afterRestart.logProbability(), 0.0);

        // Training after restart continues the version sequence without collisions.
        ModelVersion v3 = restarted.train(corpus(), TrainingConfig.defaults());
        assertEquals("v3", v3.versionId());
        assertEquals(3, restarted.versions().size());
    }

    @Test
    void activePointerSurvivesIndependentOfCommitOrder() {
        SequenceLabelingService first = newService();
        ModelVersion v1 = first.train(corpus(), TrainingConfig.defaults());
        ModelVersion v2 = first.train(corpus(), TrainingConfig.defaults());
        first.activate(v1.versionId()); // activate the older version

        SequenceLabelingService restarted = newService();
        assertEquals(v1.versionId(), restarted.activeVersion().versionId());
        assertEquals(2, restarted.versions().size());
    }
}
