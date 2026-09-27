# sequence-labeler: Local Sequence Labeling & Model Management Service

A local Spring Boot service that trains Hidden Markov Models (HMM) from labeled
sequences, registers them as immutable versions, and serves numerically stable
decoding (posterior probabilities + Viterbi best path) with atomic version
switching and disk persistence.

## Quick start

```bash
mvn spring-boot:run          # serves on http://localhost:8080
mvn test                     # runs the full test suite
```

## Probability model

The model is a first-order HMM with discrete emissions:

- **States** `S = {s0..sN-1}` — labels are supplied in training (`stateLabels`)
  or default to `S0..S(n-1)`.
- **Parameters** (all stored and computed in **log space**):
  - `pi[s]`  initial state probability
  - `A[a][b]` transition probability `P(s_b | s_a)`
  - `B[s][x]` emission probability `P(x | s)` over a fixed vocabulary built from
    the training samples (sorted lexicographically for determinism).

### Training pipeline (`HmmTrainer`)

1. **Canonicalization** — samples are sorted by an unambiguous
   length-prefixed key, so repeated or out-of-order arrival yields identical
   parameters. Counts are accumulated as integers (exact).
2. **Supervised MLE initialization** — initial/transition/emission counts from
   the gold labels, add-one smoothed so every parameter is strictly positive.
3. **Seeded jitter** (only when `maxIterations > 0`) — a `java.util.Random`
   seeded by `seed` perturbs the initial parameters (amplitude 1e-3) to break
   symmetry before EM.
4. **Baum-Welch EM** — unsupervised refinement on the observation sequences,
   at most `maxIterations` iterations, stopping early when the dataset
   log-likelihood gain drops below `tolerance`. A tiny pseudo-count floor
   (1e-9) keeps parameters positive.

Training is a **pure function**: all samples are validated before anything is
computed, so an invalid sample can never leave a partial update behind. Same
input + same configuration ⇒ bit-identical parameters.

### Decoding (`HmmDecoder`)

All recursions run in log space with max-shifted `logSumExp` accumulation, so
sequences of any length (tested at 100,000 symbols) neither underflow nor
overflow. Decoding returns:

- `logLikelihood` — `log P(observations)` from the forward algorithm,
- `posteriors[t][s]` — `P(state_t = s | observations)` (forward-backward),
  each row sums to 1,
- `bestPath` / `bestPathLogProbability` — Viterbi decoding; ties resolve to
  the lowest state index for determinism.

## HTTP API

Base path: `/api/models`

| Method | Path                    | Description                                        |
|--------|-------------------------|----------------------------------------------------|
| POST   | `/train`                | Train, register and activate a new version (201).  |
| POST   | `/decode`               | Decode with `modelVersion` or the active version.  |
| GET    | `/`                     | List all committed versions.                       |
| POST   | `/{version}/activate`   | Atomically switch the active version.              |

### Train request

```json
{
  "numStates": 2,
  "maxIterations": 20,
  "tolerance": 0.0001,
  "seed": 42,
  "stateLabels": ["N", "V"],
  "samples": [
    {"observations": ["dogs", "bark"], "labels": ["N", "V"]},
    {"observations": ["cats", "sleep"], "labels": ["N", "V"]}
  ]
}
```

`maxIterations` (default 20), `tolerance` (default 1e-4), `seed` (default 0)
and `stateLabels` (default `S0..S(n-1)`) are optional; `numStates` and
`samples` are required. The response is the committed `ModelVersionView`
including `version`, `iterationsRun` and `finalLogLikelihood`.

### Decode request

```json
{"modelVersion": 1, "observations": ["dogs", "bark"]}
```

`modelVersion` is optional (defaults to the active version). The response
carries `modelVersion`, `logLikelihood`, `posteriors`, `bestPath` and
`bestPathLogProbability`.

### Error contract

Failures return `{ "code": "...", "message": "..." }` with a
machine-distinguishable code — invalid input is never silently defaulted:

| Code                  | HTTP | Meaning                                            |
|-----------------------|------|----------------------------------------------------|
| `INVALID_LABEL`       | 400  | Unknown label or label/observation length mismatch.|
| `MISSING_OBSERVATION` | 400  | Observations field missing or contains null/blank. |
| `EMPTY_SEQUENCE`      | 400  | Empty observation sequence.                        |
| `UNKNOWN_SYMBOL`      | 400  | Decode-time symbol outside the model vocabulary.   |
| `INVALID_CONFIG`      | 400  | Bad training configuration or malformed body.      |
| `MODEL_NOT_FOUND`     | 404  | Requested version does not exist.                  |
| `NO_ACTIVE_MODEL`     | 409  | Decode requested before any model was trained.     |
| `INTERNAL_ERROR`      | 500  | Unexpected failure.                                |

## Versioning & concurrency

- Committed versions are **immutable**; readers take one atomic snapshot of
  the active-version reference per request, so a request always uses exactly
  one whole version — never a mix of old and new parameters.
- Training commits are serialized: the version file is persisted **before**
  the version becomes visible, then the registry pointer is updated and
  persisted. Historical versions stay decodable forever.
- Newly trained versions are auto-activated; `POST /{version}/activate`
  switches back atomically.

## Persistence format

Everything lives under `hmm.data-dir` (default `./data`):

```
data/
  models/v1.json      # one committed version per file
  models/v2.json
  registry.json       # {"activeVersion": 2, "versions": [1, 2]}
```

`v<N>.json` contains `version`, `createdAtEpochMilli`, the full training
`config`, `sampleCount`, `iterationsRun`, `finalLogLikelihood`, and the model
itself: `stateLabels`, `vocabulary`, `logPi`, `logA`, `logB` (log-space
parameter arrays). Every write goes to a `*.tmp` file followed by an atomic
move, so a crash never leaves a half-written version. On startup all version
files are reloaded and the persisted active pointer is restored (falling back
to the latest version if the pointer is missing).

## Configuration

| Property       | Default  | Description                                  |
|----------------|----------|----------------------------------------------|
| `hmm.data-dir` | `./data` | Directory for version and registry files.    |

Training-time configuration (per request): `numStates`, `maxIterations`,
`tolerance`, `seed`, `stateLabels` — see *Train request* above.

## Tests

`mvn test` covers: numerical stability on 100k-symbol sequences, forward
likelihood vs. brute-force enumeration, hand-verified Viterbi paths,
deterministic training under shuffled/duplicated input, seed reproducibility,
concurrent version switching and train/decode races, restart recovery,
rejection of invalid labels / missing observations / empty sequences /
unknown symbols, and bit-identical reproduction of historical results.
