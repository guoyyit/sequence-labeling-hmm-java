package com.github.guoyyit.sequencelabelinghmmjava.tests;

import com.github.guoyyit.sequencelabelinghmmjava.domain.HmmParameters;
import com.github.guoyyit.sequencelabelinghmmjava.hmm.HmmInference;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Numerical stability of log-space inference on long sequences.
 */
class HmmInferenceTest {

    private static HmmParameters tinyModel() {
        return new HmmParameters(
                2,
                List.of("N", "V"),
                List.of("a", "b"),
                new double[]{StrictMath.log(0.6), StrictMath.log(0.4)},
                new double[][]{
                        {StrictMath.log(0.7), StrictMath.log(0.3)},
                        {StrictMath.log(0.4), StrictMath.log(0.6)}},
                new double[][]{
                        {StrictMath.log(0.9), StrictMath.log(0.1)},
                        {StrictMath.log(0.2), StrictMath.log(0.8)}});
    }

    @Test
    void singleObservationLikelihoodIsExact() {
        HmmParameters model = tinyModel();
        // P("a") = 0.6*0.9 + 0.4*0.2 = 0.62
        double ll = HmmInference.forwardLogLikelihood(model, new int[]{0});
        assertEquals(StrictMath.log(0.62), ll, 1e-12);
    }

    @Test
    void longSequenceDoesNotUnderflow() {
        HmmParameters model = tinyModel();
        int length = 100_000;
        int[] observations = new int[length];
        for (int t = 0; t < length; t++) {
            observations[t] = t % 2;
        }

        double ll = HmmInference.forwardLogLikelihood(model, observations);
        // In probability space this would underflow to 0.0; log space must stay finite.
        assertTrue(Double.isFinite(ll), "log-likelihood must stay finite, got " + ll);
        assertTrue(ll < 0, "log-likelihood of a long sequence must be negative");

        HmmInference.ViterbiResult viterbi = HmmInference.viterbi(model, observations);
        assertEquals(length, viterbi.statePath().length);
        assertTrue(Double.isFinite(viterbi.logProbability()),
                "viterbi log probability must stay finite");
        // Viterbi path probability can never exceed the total sequence likelihood.
        assertTrue(viterbi.logProbability() <= ll + 1e-9);
    }

    @Test
    void posteriorsSumToOneOnLongSequence() {
        HmmParameters model = tinyModel();
        int length = 50_000;
        int[] observations = new int[length];
        for (int t = 0; t < length; t++) {
            observations[t] = (t * 7) % 2;
        }

        HmmInference.ForwardBackwardResult fb = HmmInference.forwardBackward(model, observations);
        assertTrue(Double.isFinite(fb.logLikelihood()));
        for (int t = 0; t < length; t += 4096) {
            double sum = 0.0;
            for (int i = 0; i < model.getStateCount(); i++) {
                double p = StrictMath.exp(fb.logGamma()[t][i]);
                assertTrue(Double.isFinite(p));
                sum += p;
            }
            assertEquals(1.0, sum, 1e-6, "posteriors must sum to 1 at position " + t);
        }
    }

    @Test
    void tinyProbabilitiesDoNotOverflow() {
        // Extremely skewed model: probabilities near 1e-300.
        double tiny = StrictMath.log(1e-300);
        double nearOne = StrictMath.log1p(-1e-300);
        HmmParameters model = new HmmParameters(
                2,
                List.of("N", "V"),
                List.of("a"),
                new double[]{nearOne, tiny},
                new double[][]{{nearOne, tiny}, {tiny, nearOne}},
                new double[][]{{0.0}, {0.0}});
        int[] observations = new int[10_000];

        double ll = HmmInference.forwardLogLikelihood(model, observations);
        assertTrue(Double.isFinite(ll), "must not overflow/underflow, got " + ll);
        HmmInference.ViterbiResult viterbi = HmmInference.viterbi(model, observations);
        assertTrue(Double.isFinite(viterbi.logProbability()));
        // The dominant path stays in state 0 the whole time.
        for (int state : viterbi.statePath()) {
            assertEquals(0, state);
        }
    }
}
