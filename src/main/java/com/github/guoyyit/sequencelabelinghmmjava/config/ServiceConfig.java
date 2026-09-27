package com.github.guoyyit.sequencelabelinghmmjava.config;

import com.github.guoyyit.sequencelabelinghmmjava.registry.ModelRegistry;
import com.github.guoyyit.sequencelabelinghmmjava.registry.ModelStore;
import com.github.guoyyit.sequencelabelinghmmjava.service.SequenceLabelingService;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.nio.file.Path;

/**
 * Wires the registry, store and service beans. The data directory is
 * configurable via {@code hmm.data-dir} (default {@code ./data}).
 */
@Configuration
public class ServiceConfig {

    @Bean
    public ModelStore modelStore(@Value("${hmm.data-dir:./data}") String dataDir) {
        return new ModelStore(Path.of(dataDir));
    }

    @Bean
    public ModelRegistry modelRegistry(ModelStore store) {
        return new ModelRegistry(store);
    }

    @Bean
    public SequenceLabelingService sequenceLabelingService(ModelRegistry registry) {
        return new SequenceLabelingService(registry);
    }
}
