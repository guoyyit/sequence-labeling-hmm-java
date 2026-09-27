# sequence-labeling-hmm-java

本地序列标注与模型管理服务。基于一阶隐马尔可夫模型（HMM），支持从带标签序列
训练模型、版本化注册与原子切换、并发解码，并将模型与运行状态持久化到本地磁盘。

## 构建与运行

```bash
mvn package                              # 构建并运行全部测试
java -jar target/sequence-labeling-hmm-java-1.0-SNAPSHOT.jar
```

服务默认监听 8080 端口，数据目录默认 `./data`（见下文配置）。

## 概率模型

模型为一阶 HMM，全部参数与计算均在**对数空间**进行：

- `π_i = P(state_0 = i)`：初始状态分布，存 `logInitial[i]`
- `A_ij = P(state_{t+1} = j | state_t = i)`：转移矩阵，存 `logTransition[i][j]`
- `B_i(k) = P(obs_t = k | state_t = i)`：发射矩阵，存 `logEmission[i][k]`

状态 `i` 对应标签 `labels[i]`，符号 `k` 对应词表 `vocabulary[k]`。

### 训练

1. **整批校验**：任一非法样本（空序列、缺失观测、标签数量不符、空白标签）
   都会使整批训练失败，不会留下任何部分更新。
2. **规范化**：完全相同的样本去重，剩余样本按内容排序为规范顺序。因此
   重复或乱序到达的相同样本集合产生**逐位一致**的参数。
3. **监督 MLE 初始化**：对初始/转移/发射计数做加 α 平滑（`smoothingAlpha`）
   后归一化。计数为零的行回退为均匀分布。
4. **Baum-Welch 精调**：在观测序列上执行 EM，最多 `maxIterations` 轮，
   对数似然提升小于 `convergenceThreshold` 时提前收敛。
5. **未观测状态**：当 `stateCount` 大于不同标签数时，多余状态由
   `seed` 播种的随机数初始化，相同种子结果可复现。

所有浮点运算使用 `StrictMath`，保证跨平台逐位可复现。

### 推理

- **解码**：对数空间 Viterbi，返回最优标签路径及其对数概率。
- **打分**：对数空间前向-后向算法，返回每个位置上各状态的后验概率
  （`logSpace=true` 时返回对数值）与序列对数似然。
- 长序列不会上溢/下溢：所有求和经 log-sum-exp 完成，10 万长度序列的
  对数似然仍为有限值（有测试覆盖）。

## 版本化与并发

- 每次训练提交一个**不可变**版本（`v1`、`v2`…），提交即持久化。
- 激活指针是原子引用：切换一步到位，读取方只会看到某个已提交的完整
  版本，绝不会观察到新旧参数混合。
- 训练经锁串行化；解码/打分无锁，可安全并发。
- 重启后从数据目录恢复全部版本与激活指针，版本号继续递增。

## 持久化格式

数据目录（`hmm.data-dir`）布局：

```
<data-dir>/
├── registry.json        # 注册表索引：{"versionIds":["v1","v2"],"activeVersionId":"v2"}
└── versions/
    ├── v1.json          # 单个模型版本
    └── v2.json
```

版本文件为 JSON：

```json
{
  "versionId": "v1",
  "createdAt": "2026-09-28T03:00:00Z",
  "config": {"stateCount": null, "maxIterations": 20,
             "convergenceThreshold": 1.0E-6, "seed": 42, "smoothingAlpha": 0.01},
  "parameters": {
    "stateCount": 3,
    "labels": ["D", "N", "V"],
    "vocabulary": ["a", "cat", "ran", "the"],
    "logInitial": [...],
    "logTransition": [[...], ...],
    "logEmission": [[...], ...]
  },
  "metrics": {"sampleCount": 8, "uniqueSampleCount": 8,
              "finalLogLikelihood": -31.4, "iterations": 6, "converged": true}
}
```

所有写入先写临时文件再原子移动（`ATOMIC_MOVE`），崩溃不会留下半写的
版本或索引。

## 配置项

### 服务配置（`application.properties`）

| 键 | 默认值 | 说明 |
|---|---|---|
| `hmm.data-dir` | `./data` | 模型与注册表持久化目录 |

### 训练配置（`POST /api/v1/models/train` 的 `config`，均可缺省）

| 字段 | 默认值 | 约束 | 说明 |
|---|---|---|---|
| `stateCount` | 不同标签数 | ≥ 1 且 ≥ 不同标签数 | 隐状态数；超出标签数的状态由种子随机初始化 |
| `maxIterations` | 20 | ≥ 0 | Baum-Welch 迭代上限；0 表示仅监督 MLE |
| `convergenceThreshold` | 1e-6 | ≥ 0 | 对数似然提升小于该值即收敛 |
| `seed` | 42 | 任意长整型 | 随机种子；相同输入+配置结果逐条一致 |
| `smoothingAlpha` | 0.01 | ≥ 0 | 加 α 平滑系数 |

## REST API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/v1/models/train` | 训练并注册新版本（不自动激活） |
| POST | `/api/v1/models/{versionId}/activate` | 原子切换激活版本 |
| GET | `/api/v1/models` | 列出全部版本（含 `active` 标记） |
| GET | `/api/v1/models/active` | 当前激活版本 |
| POST | `/api/v1/decode` | Viterbi 解码；`versionId` 缺省用激活版本 |
| POST | `/api/v1/score` | 各状态后验概率 + 对数似然；`logSpace` 返回对数值 |

示例：

```bash
curl -X POST localhost:8080/api/v1/models/train -H 'Content-Type: application/json' -d '{
  "samples": [
    {"observations": ["the","cat","sat"], "labels": ["D","N","V"]},
    {"observations": ["the","dog","ran"], "labels": ["D","N","V"]}
  ],
  "config": {"maxIterations": 10, "seed": 7}
}'
curl -X POST localhost:8080/api/v1/models/v1/activate
curl -X POST localhost:8080/api/v1/decode -H 'Content-Type: application/json' \
  -d '{"observations": ["the","cat","sat"]}'
```

## 错误契约

错误响应体统一为 `{"code": ..., "message": ..., "timestamp": ...}`，
`code` 可机器区分：

| code | HTTP | 触发场景 |
|---|---|---|
| `EMPTY_SEQUENCE` | 400 | 空观测序列 / 空训练批次 |
| `MISSING_OBSERVATION` | 400 | 观测序列缺失或含空元素 |
| `INVALID_LABEL` | 400 | 标签缺失、空白或与观测数量不符 |
| `UNKNOWN_SYMBOL` | 400 | 解码/打分出现词表外符号 |
| `INVALID_CONFIG` | 400 | 非法训练配置（如 `stateCount` 小于标签数） |
| `MODEL_VERSION_NOT_FOUND` | 404 | 未知版本号 |
| `NO_ACTIVE_MODEL` | 409 | 尚未激活任何版本 |
| `PERSISTENCE_FAILURE` | 500 | 磁盘读写失败 |

非法输入一律拒绝，不会以默认值静默替代。

## 测试

`src/test/java/.../tests/` 覆盖：

- `HmmInferenceTest`：长序列数值稳定性（10 万长度无下溢）、极小概率不溢出、
  后验概率归一、单点似然精确值。
- `HmmTrainerDeterminismTest`：乱序/重复样本逐位一致、同种子复现、
  异种子分叉、非法样本整批拒绝。
- `ConcurrencyTest`：并发解码 + 原子切换不观察混合参数、并发训练版本号唯一、
  训练与历史查询并发自洽。
- `PersistenceTest`：重启后版本与激活指针恢复、参数逐位一致、版本号延续。
- `ValidationTest`：各类非法输入的错误码、失败批次无部分更新。
- `ApiContractTest`：HTTP 层训练→激活→解码流程与错误响应契约。
