package com.github.guoyyit.sequencelabelinghmmjava.domain;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Immutable HMM parameters, all probabilities stored in log space.
 * State i corresponds to {@code labels.get(i)}; symbol k corresponds to
 * {@code vocabulary.get(k)}. Array references are exposed for read-only use;
 * callers must not mutate them.
 */
public final class HmmParameters {

    private final int stateCount;
    private final List<String> labels;
    private final List<String> vocabulary;
    private final double[] logInitial;
    private final double[][] logTransition;
    private final double[][] logEmission;

    @JsonIgnore
    private final Map<String, Integer> labelIndex;
    @JsonIgnore
    private final Map<String, Integer> symbolIndex;

    @JsonCreator
    public HmmParameters(
            @JsonProperty("stateCount") int stateCount,
            @JsonProperty("labels") List<String> labels,
            @JsonProperty("vocabulary") List<String> vocabulary,
            @JsonProperty("logInitial") double[] logInitial,
            @JsonProperty("logTransition") double[][] logTransition,
            @JsonProperty("logEmission") double[][] logEmission) {
        this.stateCount = stateCount;
        this.labels = List.copyOf(labels);
        this.vocabulary = List.copyOf(vocabulary);
        this.logInitial = logInitial.clone();
        this.logTransition = copy(logTransition);
        this.logEmission = copy(logEmission);
        this.labelIndex = indexOf(this.labels);
        this.symbolIndex = indexOf(this.vocabulary);
    }

    private static double[][] copy(double[][] m) {
        double[][] c = new double[m.length][];
        for (int i = 0; i < m.length; i++) {
            c[i] = m[i].clone();
        }
        return c;
    }

    private static Map<String, Integer> indexOf(List<String> values) {
        Map<String, Integer> map = new HashMap<>();
        for (int i = 0; i < values.size(); i++) {
            map.put(values.get(i), i);
        }
        return Map.copyOf(map);
    }

    public int getStateCount() {
        return stateCount;
    }

    public List<String> getLabels() {
        return labels;
    }

    public List<String> getVocabulary() {
        return vocabulary;
    }

    public double[] getLogInitial() {
        return logInitial;
    }

    public double[][] getLogTransition() {
        return logTransition;
    }

    public double[][] getLogEmission() {
        return logEmission;
    }

    /** @return state index of the label, or null when the label is unknown. */
    @JsonIgnore
    public Integer indexOfLabel(String label) {
        return labelIndex.get(label);
    }

    /** @return symbol index of the observation, or null when unknown. */
    @JsonIgnore
    public Integer indexOfSymbol(String symbol) {
        return symbolIndex.get(symbol);
    }
}
