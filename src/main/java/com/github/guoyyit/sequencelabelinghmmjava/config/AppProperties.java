package com.github.guoyyit.sequencelabelinghmmjava.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * Application-level configuration properties.
 */
@ConfigurationProperties(prefix = "hmm")
public class AppProperties {

    private String dataDir = "./data";

    public String getDataDir() {
        return dataDir;
    }

    public void setDataDir(String dataDir) {
        this.dataDir = dataDir;
    }
}
