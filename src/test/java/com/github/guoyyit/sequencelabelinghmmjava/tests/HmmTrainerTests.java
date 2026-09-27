package com.github.guoyyit.sequencelabelinghmmjava.tests;

import com.github.guoyyit.sequencelabelinghmmjava.hmm.HmmModel;
import com.github.guoyyit.sequencelabelinghmmjava.hmm.HmmTrainer;
import com.github.guoyyit.sequencelabelinghmmjava.hmm.TrainConfig;
import com.github.guoyyit.sequencelabelinghmmjava.hmm.TrainSample;
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
 * Determinism and reproducibility of training (requirements 1 and 6).
 */
class HmmTrainerTests {

    private final HmmTrainer trainer = new HmmTrainer();

    private static List<TrainSample> sampleCorpus() {
        return List.of(
                new TrainSample(List.of("walk", "the", "dog"), List.of("V", "D", "N")),
                new TrainSample(List.of("the", "dog", "barks"), List.of("D", "N", "V")),
                new TrainSample(List.of("dogs", "bark"), List.of("N", "V")),
                new TrainSample(List.of("the", "cat", "sleeps"), List.of("D", "N", "V")),
                new TrainSample(List.of("cats", "sleep"), List.of("N", "V")));
    }

    private static TrainConfig config(long seed) {
        return new TrainConfig(3, 25, 1e-6, seed, List.of("D", "N", "V"));
    }

    private static void assertSameParameters(HmmModel a, HmmModel b) {
        assertEquals(a.stateLabels(), b.stateLabels());
        assertEquals(a.vocabulary(), b.vocabulary());
        assertArrayEquals(a.logPi(), b.logPi());
        for (int s = 0; s < a.numStates(); s++) {
            assertArrayEquals(a.logA()[s], b.logA()[s]);
            assertArrayEquals(a.logB()[s], b.logB()[s]);
        }
    }

    @Test
    void sameInputSameConfigProducesIdenticalParameters() {
        HmmModel first = trainer.train(sampleCorpus(), config(42L)).model();
        HmmModel second = trainer.train(sampleCorpus(), config(42L)).model();
        assertSameParameters(first, second);
    }

    @Test
    void shuffledArrivalOrderProducesIdenticalParameters() {
        HmmModel reference = trainer.train(sampleCorpus(), config(7L)).model();
        // Several deterministic shuffles, including reversed order.
        for (long shuffleSeed = 0; shuffleSeed < 5; shuffleSeed++) {
            List<TrainSample> shuffled = new ArrayList<>(sampleCorpus());
            Collections.shuffle(shuffled, new Random(shuffleSeed));
            assertSameParameters(reference, trainer.train(shuffled, config(7L)).model());
        }
    }

    @Test
    void duplicatedSamplesProduceIdenticalParametersRegardlessOfDuplicationPattern() {
        List<TrainSample> corpus = sampleCorpus();
        List<TrainSample> withDuplicates = new ArrayList<>(corpus);
        // Every sample arrives twice, interleaved in a scrambled order.
        withDuplicates.addAll(corpus);
        Collections.shuffle(withDuplicates, new Random(99L));

        List<TrainSample> withDuplicatesSorted = new ArrayList<>(corpus);
        withDuplicatesSorted.addAll(corpus);

        HmmModel a = trainer.train(withDuplicates, config(3L)).model();
        HmmModel b = trainer.train(withDuplicatesSorted, config(3L)).model();
        assertSameParameters(a, b);
    }

    @Test
    void differentSeedsMayDivergeButSameSeedAlwaysReproduces() {
        HmmModel seed1a = trainer.train(sampleCorpus(), config(1L)).model();
        HmmModel seed1b = trainer.train(sampleCorpus(), config(1L)).model();
        HmmModel seed2 = trainer.train(sampleCorpus(), config(2L)).model();
        assertSameParameters(seed1a, seed1b);
        // With EM refinement active, a different seed perturbs the start point.
        boolean differs = false;
        for (int s = 0; s < seed1a.numStates() && !differs; s++) {
            for (int i = 0; i < seed1a.logB()[s].length && !differs; i++) {
                differs = seed1a.logB()[s][i] != seed2.logB()[s][i];
            }
        }
        assertTrue(differs, "different seeds should lead to different EM refinements");
    }

    @Test
    void zeroIterationsYieldsPureSupervisedMle() {
        HmmTrainer.TrainOutcome outcome = trainer.train(sampleCorpus(), new TrainConfig(3, 0, 1e-6, 5L, List.of("D", "N", "V")));
        assertEquals(0, outcome.iterationsRun());
        // All parameters must be valid log probabilities.
        for (double p : outcome.model().logPi()) {
            assertTrue(p <= 0.0 && p > Double.NEGATIVE_INFINITY);
        }
    }

    @Test
    void invalidConfigIsRejected() {
        assertThrows(IllegalArgumentException.class,
                () -> trainer.train(sampleCorpus(), new TrainConfig(0, 10, 1e-6, 0L)));
        assertThrows(IllegalArgumentException.class,
                () -> trainer.train(sampleCorpus(), new TrainConfig(3, -1, 1e-6, 0L)));
        assertThrows(IllegalArgumentException.class,
                () -> trainer.train(sampleCorpus(), new TrainConfig(3, 10, -0.1, 0L)));
        assertThrows(IllegalArgumentException.class,
                () -> trainer.train(sampleCorpus(), new TrainConfig(2, 10, 1e-6, 0L, List.of("D", "N", "V"))));
    }

    @Test
    void emptySampleListIsRejected() {
        assertThrows(IllegalArgumentException.class,
                () -> trainer.train(List.of(), config(0L)));
    }

    @Test
    void unknownLabelIsRejected() {
        List<TrainSample> bad = List.of(
                new TrainSample(List.of("walk"), List.of("NOT_A_STATE")));
        assertThrows(IllegalArgumentException.class, () -> trainer.train(bad, config(0L)));
    }

    @Test
    void emIterationsIncreaseOrHoldLogLikelihood() {
        TrainConfig noEm = new TrainConfig(3, 0, 1e-6, 11L, List.of("D", "N", "V"));
        TrainConfig withEm = new TrainConfig(3, 30, 1e-9, 11L, List.of("D", "N", "V"));
        HmmTrainer.TrainOutcome mle = trainer.train(sampleCorpus(), noEm);
        HmmTrainer.TrainOutcome em = trainer.train(sampleCorpus(), withEm);
        assertNotEquals(0, em.iterationsRun());
        // EM never decreases the likelihood it optimizes; allow tiny float slack.
        assertTrue(em.finalLogLikelihood() >= mle.finalLogLikelihood() - 1e-6,
                "EM log-likelihood " + em.finalLogLikelihood() + " should be >= MLE " + mle.finalLogLikelihood());
    }
}
