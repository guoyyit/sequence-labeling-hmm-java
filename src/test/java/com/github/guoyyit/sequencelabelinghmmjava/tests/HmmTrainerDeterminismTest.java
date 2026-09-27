package com.github.guoyyit.sequencelabelinghmmjava.tests;

import com.github.guoyyit.sequencelabelinghmmjava.domain.HmmParameters;
import com.github.guoyyit.sequencelabelinghmmjava.domain.LabeledSample;
import com.github.guoyyit.sequencelabelinghmmjava.domain.TrainingConfig;
import com.github.guoyyit.sequencelabelinghmmjava.domain.TrainingMetrics;
import com.github.guoyyit.sequencelabelinghmmjava.error.ErrorCode;
import com.github.guoyyit.sequencelabelinghmmjava.error.ServiceException;
import com.github.guoyyit.sequencelabelinghmmjava.hmm.HmmTrainer;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Determinism and reproducibility of training: repeated or shuffled arrival of
 * the same samples, and identical configs, must reproduce identical parameters.
 */
class HmmTrainerDeterminismTest {

    private static List<LabeledSample> corpus() {
        List<LabeledSample> samples = new ArrayList<>();
        String[][] sentences = {
                {"the", "cat", "sat"},
                {"the", "dog", "ran"},
                {"a", "cat", "ate"},
                {"the", "cat", "ran"},
                {"a", "dog", "sat"},
                {"the", "dog", "ate", "quickly"},
                {"a", "cat", "sat"},
                {"the", "bird", "flew"},
        };
        String[][] tags = {
                {"D", "N", "V"},
                {"D", "N", "V"},
                {"D", "N", "V"},
                {"D", "N", "V"},
                {"D", "N", "V"},
                {"D", "N", "V", "R"},
                {"D", "N", "V"},
                {"D", "N", "V"},
        };
        for (int i = 0; i < sentences.length; i++) {
            samples.add(new LabeledSample(List.of(sentences[i]), List.of(tags[i])));
        }
        return samples;
    }

    private static void assertSameParameters(HmmParameters a, HmmParameters b) {
        assertEquals(a.getStateCount(), b.getStateCount());
        assertEquals(a.getLabels(), b.getLabels());
        assertEquals(a.getVocabulary(), b.getVocabulary());
        assertArrayEquals(a.getLogInitial(), b.getLogInitial(), 0.0);
        for (int i = 0; i < a.getStateCount(); i++) {
            assertArrayEquals(a.getLogTransition()[i], b.getLogTransition()[i], 0.0);
            assertArrayEquals(a.getLogEmission()[i], b.getLogEmission()[i], 0.0);
        }
    }

    @Test
    void shuffledAndDuplicatedSamplesProduceIdenticalParameters() {
        TrainingConfig config = TrainingConfig.defaults();
        HmmTrainer.TrainingOutcome reference = HmmTrainer.train(corpus(), config);

        Random random = new Random(7);
        for (int trial = 0; trial < 5; trial++) {
            List<LabeledSample> shuffled = new ArrayList<>(corpus());
            // Duplicate some samples: arrival repetition must not change anything.
            shuffled.addAll(corpus().subList(0, 3));
            shuffled.add(corpus().get(5));
            Collections.shuffle(shuffled, random);

            HmmTrainer.TrainingOutcome outcome = HmmTrainer.train(shuffled, config);
            assertSameParameters(reference.parameters(), outcome.parameters());
            assertEquals(reference.metrics().finalLogLikelihood(),
                    outcome.metrics().finalLogLikelihood(), 0.0);
            assertEquals(reference.metrics().iterations(), outcome.metrics().iterations());
            assertEquals(reference.metrics().uniqueSampleCount(),
                    outcome.metrics().uniqueSampleCount());
        }
    }

    @Test
    void sameConfigAndSeedReproduceIdenticalResults() {
        TrainingConfig config = new TrainingConfig(5, 30, 1e-7, 123L, 0.01);
        HmmTrainer.TrainingOutcome first = HmmTrainer.train(corpus(), config);
        HmmTrainer.TrainingOutcome second = HmmTrainer.train(corpus(), config);
        assertSameParameters(first.parameters(), second.parameters());
        assertEquals(first.metrics().finalLogLikelihood(), second.metrics().finalLogLikelihood(), 0.0);
    }

    @Test
    void differentSeedsDivergeWhenExtraStatesExist() {
        // stateCount (5) exceeds the distinct label count (3), so the seed
        // governs the initialization of the two unobserved states.
        HmmTrainer.TrainingOutcome a = HmmTrainer.train(corpus(),
                new TrainingConfig(5, 10, 1e-7, 1L, 0.01));
        HmmTrainer.TrainingOutcome b = HmmTrainer.train(corpus(),
                new TrainingConfig(5, 10, 1e-7, 2L, 0.01));
        assertNotEquals(a.parameters().getLogEmission()[3][0],
                b.parameters().getLogEmission()[3][0]);
    }

    @Test
    void metricsAreConsistentAndConverge() {
        HmmTrainer.TrainingOutcome outcome = HmmTrainer.train(corpus(), TrainingConfig.defaults());
        TrainingMetrics metrics = outcome.metrics();
        assertEquals(8, metrics.sampleCount());
        assertEquals(8, metrics.uniqueSampleCount());
        assertTrue(Double.isFinite(metrics.finalLogLikelihood()));
        assertTrue(metrics.iterations() >= 0 && metrics.iterations() <= 20);
        if (metrics.iterations() < 20) {
            assertTrue(metrics.converged());
        }
    }

    @Test
    void invalidSampleAbortsWholeBatch() {
        List<LabeledSample> samples = new ArrayList<>(corpus());
        samples.add(new LabeledSample(List.of("the", "cat"), List.of("D"))); // length mismatch
        ServiceException ex = assertThrows(ServiceException.class,
                () -> HmmTrainer.train(samples, TrainingConfig.defaults()));
        assertEquals(ErrorCode.INVALID_LABEL, ex.getCode());
    }

    @Test
    void invalidConfigIsRejected() {
        List<LabeledSample> samples = corpus();
        ServiceException ex = assertThrows(ServiceException.class, () ->
                HmmTrainer.train(samples, new TrainingConfig(2, 10, 1e-6, 1L, 0.01)));
        assertEquals(ErrorCode.INVALID_CONFIG, ex.getCode()); // 2 < 3 distinct labels

        ex = assertThrows(ServiceException.class, () ->
                HmmTrainer.train(samples, new TrainingConfig(null, -1, 1e-6, 1L, 0.01)));
        assertEquals(ErrorCode.INVALID_CONFIG, ex.getCode());

        ex = assertThrows(ServiceException.class, () ->
                HmmTrainer.train(samples, new TrainingConfig(null, 10, -0.5, 1L, 0.01)));
        assertEquals(ErrorCode.INVALID_CONFIG, ex.getCode());
    }
}
