package com.github.guoyyit.sequencelabelinghmmjava.hmm;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Immutable HMM parameter set. All probabilities are stored in log space.
 *
 * <p>Instances are deeply immutable: array references handed in are copied on
 * construction and never exposed again, so a registered model can be shared
 * safely across concurrent readers.</p>
 */
public final class HmmModel {

    private final List<String> stateLabels;
    private final List<String> vocabulary;
    private final Map<String, Integer> stateIndex;
    private final Map<String, Integer> symbolIndex;
    private final double[] logPi;
    private final double[][] logA;
    private final double[][] logB;

    public HmmModel(List<String> stateLabels,
                    List<String> vocabulary,
                    double[] logPi,
                    double[][] logA,
                    double[][] logB) {
        this.stateLabels = List.copyOf(stateLabels);
        this.vocabulary = List.copyOf(vocabulary);
        this.stateIndex = indexOf(this.stateLabels);
        this.symbolIndex = indexOf(this.vocabulary);
        this.logPi = logPi.clone();
        this.logA = deepCopy(logA);
        this.logB = deepCopy(logB);
    }

    private static Map<String, Integer> indexOf(List<String> keys) {
        Map<String, Integer> map = new HashMap<>();
        for (int i = 0; i < keys.size(); i++) {
            map.put(keys.get(i), i);
        }
        return Map.copyOf(map);
    }

    private static double[][] deepCopy(double[][] matrix) {
        double[][] copy = new double[matrix.length][];
        for (int i = 0; i < matrix.length; i++) {
            copy[i] = matrix[i].clone();
        }
        return copy;
    }

    public int numStates() {
        return logPi.length;
    }

    public List<String> stateLabels() {
        return stateLabels;
    }

    public List<String> vocabulary() {
        return vocabulary;
    }

    /** Returns the state index for a label, or {@code null} if the label is unknown. */
    public Integer stateIndexOf(String label) {
        return stateIndex.get(label);
    }

    /** Returns the vocabulary index for a symbol, or {@code null} if the symbol is unknown. */
    public Integer symbolIndexOf(String symbol) {
        return symbolIndex.get(symbol);
    }

    /**
     * Internal view of the log initial probabilities. Callers must treat the
     * returned array as read-only; only trusted in-package algorithms use this.
     */
    public double[] logPi() {
        return logPi;
    }

    /** Internal read-only view of log transition probabilities [from][to]. */
    public double[][] logA() {
        return logA;
    }

    /** Internal read-only view of log emission probabilities [state][symbol]. */
    public double[][] logB() {
        return logB;
    }
}
