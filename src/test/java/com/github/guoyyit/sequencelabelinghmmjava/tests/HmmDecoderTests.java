package com.github.guoyyit.sequencelabelinghmmjava.tests;

import com.github.guoyyit.sequencelabelinghmmjava.hmm.DecodeResult;
import com.github.guoyyit.sequencelabelinghmmjava.hmm.HmmDecoder;
import com.github.guoyyit.sequencelabelinghmmjava.hmm.HmmModel;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Numerical stability and correctness of decoding (requirement 2).
 */
class HmmDecoderTests {

    private final HmmDecoder decoder = new HmmDecoder();

    /**
     * A tiny hand-computable model:
     * states {A, B}, symbols {x, y}; A prefers x, B prefers y, and the model
     * strongly prefers staying in its current state.
     */
    private static HmmModel tinyModel() {
        double[] logPi = {Math.log(0.6), Math.log(0.4)};
        double[][] logA = {
                {Math.log(0.9), Math.log(0.1)},
                {Math.log(0.1), Math.log(0.9)}};
        double[][] logB = {
                {Math.log(0.9), Math.log(0.1)},
                {Math.log(0.2), Math.log(0.8)}};
        return new HmmModel(List.of("A", "B"), List.of("x", "y"), logPi, logA, logB);
    }

    @Test
    void viterbiFindsTheHandComputableBestPath() {
        HmmModel model = tinyModel();
        DecodeResult result = decoder.decode(model, new int[]{0, 0, 1, 1});
        // Sticky states: the best explanation of x x y y is A A B B.
        assertEquals(List.of("A", "A", "B", "B"), result.bestPath());

        // Hand-verified joint log probability of (path, observations):
        // ln(0.6*0.9 * 0.9*0.9 * 0.1*0.8 * 0.9*0.8)
        double expected = Math.log(0.6) + Math.log(0.9)
                + Math.log(0.9) + Math.log(0.9)
                + Math.log(0.1) + Math.log(0.8)
                + Math.log(0.9) + Math.log(0.8);
        assertEquals(expected, result.bestPathLogProbability(), 1e-10);
    }

    @Test
    void forwardLikelihoodMatchesBruteForceEnumeration() {
        HmmModel model = tinyModel();
        int[] obs = {0, 1, 0};
        DecodeResult result = decoder.decode(model, new int[]{0, 1, 0});

        // Enumerate all 2^3 state paths in plain probability space.
        double[] pi = {0.6, 0.4};
        double[][] a = {{0.9, 0.1}, {0.1, 0.9}};
        double[][] b = {{0.9, 0.1}, {0.2, 0.8}};
        double total = 0.0;
        for (int s0 = 0; s0 < 2; s0++) {
            for (int s1 = 0; s1 < 2; s1++) {
                for (int s2 = 0; s2 < 2; s2++) {
                    total += pi[s0] * b[s0][obs[0]]
                            * a[s0][s1] * b[s1][obs[1]]
                            * a[s1][s2] * b[s2][obs[2]];
                }
            }
        }
        assertEquals(Math.log(total), result.logLikelihood(), 1e-10);
    }

    @Test
    void posteriorRowsSumToOne() {
        HmmModel model = tinyModel();
        DecodeResult result = decoder.decode(model, new int[]{0, 1, 1, 0, 0});
        for (double[] row : result.posteriors()) {
            double sum = 0.0;
            for (double p : row) {
                sum += p;
            }
            assertEquals(1.0, sum, 1e-9);
        }
    }

    @Test
    void longSequencesStayNumericallyStable() {
        HmmModel model = tinyModel();
        int len = 100_000;
        int[] obs = new int[len];
        for (int i = 0; i < len; i++) {
            obs[i] = i % 2;
        }
        DecodeResult result = decoder.decode(model, obs);

        // In linear space the likelihood would be ~1e-48000, i.e. exactly 0.0
        // as a double; in log space it must be a finite, correct-magnitude value.
        assertTrue(Double.isFinite(result.logLikelihood()),
                "log-likelihood must stay finite for long sequences");
        assertTrue(result.logLikelihood() < 0.0);
        // Rough magnitude check: per-symbol log-likelihood of this model is O(1).
        assertTrue(result.logLikelihood() > -3.0 * len && result.logLikelihood() < -0.1 * len);

        assertEquals(len, result.bestPath().size());
        assertTrue(Double.isFinite(result.bestPathLogProbability()));
        // The best-path joint probability can never exceed the total likelihood.
        assertTrue(result.bestPathLogProbability() <= result.logLikelihood() + 1e-6);

        double[] lastRow = result.posteriors()[len - 1];
        double sum = lastRow[0] + lastRow[1];
        assertEquals(1.0, sum, 1e-9);
    }

    @Test
    void singleObservationSequenceDecodes() {
        HmmModel model = tinyModel();
        DecodeResult result = decoder.decode(model, new int[]{0});
        assertEquals(1, result.bestPath().size());
        assertEquals("A", result.bestPath().get(0)); // 0.6*0.9 > 0.4*0.2
        assertEquals(Math.log(0.6 * 0.9 + 0.4 * 0.2), result.logLikelihood(), 1e-10);
    }

    @Test
    void emptySequenceIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> decoder.decode(tinyModel(), new int[0]));
    }
}
