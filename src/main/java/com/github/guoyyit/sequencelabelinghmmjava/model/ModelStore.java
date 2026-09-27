package com.github.guoyyit.sequencelabelinghmmjava.model;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.guoyyit.sequencelabelinghmmjava.config.AppProperties;
import com.github.guoyyit.sequencelabelinghmmjava.hmm.HmmModel;
import com.github.guoyyit.sequencelabelinghmmjava.hmm.TrainConfig;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;

/**
 * Local disk persistence for model versions and registry state.
 *
 * <p>Layout under {@code hmm.data-dir}:</p>
 * <pre>
 *   models/v&lt;version&gt;.json   one committed, immutable model version per file
 *   registry.json             active-version pointer and known version ids
 * </pre>
 *
 * <p>Every write goes to a temporary file in the same directory followed by an
 * atomic move, so a crash can never leave a half-written file that a reader
 * would mistake for a committed version.</p>
 */
@Component
public class ModelStore {

    private static final String MODELS_DIR = "models";
    private static final String REGISTRY_FILE = "registry.json";

    private final Path dataDir;
    private final ObjectMapper objectMapper;

    public ModelStore(AppProperties properties, ObjectMapper objectMapper) {
        this.dataDir = Path.of(properties.getDataDir());
        this.objectMapper = objectMapper;
    }

    // ------------------------------------------------------------------
    // write path
    // ------------------------------------------------------------------

    /** Persists one committed version atomically. */
    public void saveVersion(ModelVersion version) {
        ModelFile file = ModelFile.from(version);
        writeAtomic(modelsDir().resolve(fileName(version.version())), file);
    }

    /** Persists the registry pointer (active version + known ids) atomically. */
    public void saveRegistryState(Long activeVersion, List<Long> versions) {
        RegistryFile file = new RegistryFile(activeVersion, List.copyOf(versions));
        writeAtomic(dataDir.resolve(REGISTRY_FILE), file);
    }

    // ------------------------------------------------------------------
    // read path
    // ------------------------------------------------------------------

    /** Loads all committed versions found on disk, ordered by version id. */
    public List<ModelVersion> loadVersions() {
        Path dir = modelsDir();
        if (!Files.isDirectory(dir)) {
            return List.of();
        }
        List<ModelVersion> loaded = new ArrayList<>();
        try (Stream<Path> files = Files.list(dir)) {
            List<Path> versionFiles = files
                    .filter(p -> p.getFileName().toString().matches("v\\d+\\.json"))
                    .sorted()
                    .toList();
            for (Path p : versionFiles) {
                loaded.add(readVersion(p));
            }
        } catch (IOException e) {
            throw new UncheckedIOException("failed to list model versions in " + dir, e);
        }
        loaded.sort(Comparator.comparingLong(ModelVersion::version));
        return List.copyOf(loaded);
    }

    /** Loads the persisted active-version pointer, if any. */
    public Optional<Long> loadActiveVersion() {
        Path file = dataDir.resolve(REGISTRY_FILE);
        if (!Files.exists(file)) {
            return Optional.empty();
        }
        try {
            RegistryFile registry = objectMapper.readValue(file.toFile(), RegistryFile.class);
            return Optional.ofNullable(registry.activeVersion());
        } catch (IOException e) {
            throw new UncheckedIOException("failed to read registry state from " + file, e);
        }
    }

    // ------------------------------------------------------------------
    // internals
    // ------------------------------------------------------------------

    private Path modelsDir() {
        return dataDir.resolve(MODELS_DIR);
    }

    private static String fileName(long version) {
        return "v" + version + ".json";
    }

    private ModelVersion readVersion(Path file) {
        try {
            return objectMapper.readValue(file.toFile(), ModelFile.class).toModelVersion();
        } catch (IOException e) {
            throw new UncheckedIOException("corrupt or unreadable model version file: " + file, e);
        }
    }

    private void writeAtomic(Path target, Object payload) {
        try {
            Files.createDirectories(target.getParent());
            Path tmp = target.resolveSibling(target.getFileName() + ".tmp");
            objectMapper.writeValue(tmp.toFile(), payload);
            Files.move(tmp, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException e) {
            throw new UncheckedIOException("failed to persist " + target, e);
        }
    }

    /** JSON representation of the registry pointer file. */
    record RegistryFile(Long activeVersion, List<Long> versions) {
    }

    /** JSON representation of one model version file. */
    record ModelFile(
            long version,
            long createdAtEpochMilli,
            TrainConfig config,
            int sampleCount,
            int iterationsRun,
            double finalLogLikelihood,
            List<String> stateLabels,
            List<String> vocabulary,
            double[] logPi,
            double[][] logA,
            double[][] logB) {

        static ModelFile from(ModelVersion v) {
            HmmModel m = v.model();
            return new ModelFile(
                    v.version(),
                    v.createdAt().toEpochMilli(),
                    v.config(),
                    v.sampleCount(),
                    v.iterationsRun(),
                    v.finalLogLikelihood(),
                    m.stateLabels(),
                    m.vocabulary(),
                    m.logPi(),
                    m.logA(),
                    m.logB());
        }

        ModelVersion toModelVersion() {
            HmmModel model = new HmmModel(stateLabels, vocabulary, logPi, logA, logB);
            return new ModelVersion(
                    version,
                    Instant.ofEpochMilli(createdAtEpochMilli),
                    config,
                    model,
                    sampleCount,
                    iterationsRun,
                    finalLogLikelihood);
        }
    }
}
