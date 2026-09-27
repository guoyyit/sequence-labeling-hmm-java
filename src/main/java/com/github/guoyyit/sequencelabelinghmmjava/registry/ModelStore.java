package com.github.guoyyit.sequencelabelinghmmjava.registry;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.github.guoyyit.sequencelabelinghmmjava.domain.ModelVersion;
import com.github.guoyyit.sequencelabelinghmmjava.error.ErrorCode;
import com.github.guoyyit.sequencelabelinghmmjava.error.ServiceException;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * JSON file persistence for model versions and the registry index.
 *
 * <p>Layout under the data directory:</p>
 * <pre>
 *   registry.json          registry index (version ids + active pointer)
 *   versions/&lt;id&gt;.json     one file per committed model version
 * </pre>
 *
 * <p>Every write goes to a temp file first and is then moved atomically, so a
 * crash can never leave a half-written version or index visible.</p>
 */
public class ModelStore {

    private static final Pattern SAFE_ID = Pattern.compile("[A-Za-z0-9_-]+");
    private static final String REGISTRY_FILE = "registry.json";
    private static final String VERSIONS_DIR = "versions";

    private final Path dataDir;
    private final Path versionsDir;
    private final ObjectMapper mapper;

    public ModelStore(Path dataDir) {
        this.dataDir = dataDir;
        this.versionsDir = dataDir.resolve(VERSIONS_DIR);
        this.mapper = new ObjectMapper()
                .registerModule(new JavaTimeModule())
                .disable(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS);
        try {
            Files.createDirectories(this.versionsDir);
        } catch (IOException e) {
            throw new ServiceException(ErrorCode.PERSISTENCE_FAILURE,
                    "cannot create data directory " + dataDir, e);
        }
    }

    /** Persists a committed version (idempotent per version id). */
    public void saveVersion(ModelVersion version) {
        checkId(version.versionId());
        writeAtomic(versionFile(version.versionId()), version);
    }

    /** Persists the registry index (version ids + active pointer). */
    public void saveState(RegistryState state) {
        writeAtomic(dataDir.resolve(REGISTRY_FILE), state);
    }

    public Optional<ModelVersion> loadVersion(String versionId) {
        checkId(versionId);
        Path file = versionFile(versionId);
        if (!Files.exists(file)) {
            return Optional.empty();
        }
        try {
            return Optional.of(mapper.readValue(file.toFile(), ModelVersion.class));
        } catch (IOException e) {
            throw new ServiceException(ErrorCode.PERSISTENCE_FAILURE,
                    "cannot read model version " + versionId, e);
        }
    }

    /** Loads every persisted version, ordered by numeric version id. */
    public List<ModelVersion> loadAllVersions() {
        List<ModelVersion> versions = new ArrayList<>();
        try (Stream<Path> files = Files.list(versionsDir)) {
            List<Path> sorted = files
                    .filter(f -> f.getFileName().toString().endsWith(".json"))
                    .sorted(Comparator.comparingLong(this::versionNumber))
                    .toList();
            for (Path file : sorted) {
                versions.add(mapper.readValue(file.toFile(), ModelVersion.class));
            }
        } catch (IOException | UncheckedIOException e) {
            throw new ServiceException(ErrorCode.PERSISTENCE_FAILURE,
                    "cannot load persisted model versions", e);
        }
        return versions;
    }

    public Optional<RegistryState> loadState() {
        Path file = dataDir.resolve(REGISTRY_FILE);
        if (!Files.exists(file)) {
            return Optional.empty();
        }
        try {
            return Optional.of(mapper.readValue(file.toFile(), RegistryState.class));
        } catch (IOException e) {
            throw new ServiceException(ErrorCode.PERSISTENCE_FAILURE,
                    "cannot read registry index", e);
        }
    }

    private long versionNumber(Path file) {
        String name = file.getFileName().toString();
        String digits = name.replaceAll("[^0-9]", "");
        return digits.isEmpty() ? Long.MAX_VALUE : Long.parseLong(digits);
    }

    private Path versionFile(String versionId) {
        return versionsDir.resolve(versionId + ".json");
    }

    private void checkId(String versionId) {
        if (versionId == null || !SAFE_ID.matcher(versionId).matches()) {
            throw new ServiceException(ErrorCode.MODEL_VERSION_NOT_FOUND,
                    "invalid version id: " + versionId);
        }
    }

    private void writeAtomic(Path target, Object value) {
        Path temp = target.resolveSibling(target.getFileName() + ".tmp");
        try {
            mapper.writerWithDefaultPrettyPrinter().writeValue(temp.toFile(), value);
            Files.move(temp, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException e) {
            throw new ServiceException(ErrorCode.PERSISTENCE_FAILURE,
                    "cannot persist " + target.getFileName(), e);
        }
    }

    /** Registry index: committed version ids and the active version pointer. */
    public record RegistryState(List<String> versionIds, String activeVersionId) {
    }
}
