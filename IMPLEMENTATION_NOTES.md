# 实现说明

## 阶段 B、C 与交付更新

阶段 B 增加 Streamlit 界面、JSON/CSV 解析和人工事件核对。用户可修正每条攻击发现的发起者、受影响对象和事件类型，原始消息及核对来源随报告保存。SQLite 将消息、事件、候选回应、特征、模型版本与支持结果按键保存。重复提交同一事件更新其计算记录，人工跟进单独保留。历史快照带修订号，若并发运行已更新历史，旧快照的写入会整体拒绝。

阶段 C 使用配置指定的真实 Sentence-BERT 构建关联语义余弦和八类分类的联合向量。已训练线性头用 JSON 保存 W、b 和训练来源。训练脚本学习时使用 BCEWithLogitsLoss，固定训练轮次后只用留出场景测预测，概率由同一线性层公式和 Sigmoid 得出。训练和留出数据按情境组划分，分别有 64 和 32 条。生成脚本检查跨集同组、相同摘要、相同回复和字符序列近重复。阈值 0.5 仅用于计算合成留出集指标。训练配置、证据来源和每条候选的模型状态都随 JSON 导出。

Qwen 可用性检查、关联一次请求、支持两次请求与结构化输出校验在 `llm_adapter.py` 和 `support.py`。模型不可用时支持页标记 `safe_template`。配置已启用 Qwen 但响应不可用时，相应 D 或 G 取 0 并附原因。显式选择不调用 Qwen 的演示路径可使用原来的启发式 D/G。S7 写入 SQLite 的后续支持记录由人工填写。

需求书提供的 b0 至 b6 仍作为工程演示参数。可用 `python -m scripts.calibrate_association --labels <CSV> --epochs <轮次> --learning-rate <学习率>` 读取人工关联标签的 `L,Q,S,T,D,G,label` 六特征数据，输出候选系数。该命令只输出建议参数，由使用者复核后再更新配置。

阶段 D 按本轮授权保留 `MediaFeatureExtractor` 接口和许可目录校验，结构化模拟观测继续用于验证非语言计算。真实媒体提取与验证待提供授权素材后开展。录屏脚本和 S1–S7 的源码、测试、界面、录像片段对应关系在 [录屏脚本](docs/DEMO_SCRIPT.md)。

## 阶段 A 公式与工程口径

公式以用户提供的 DOCX 权利要求和「具体实施方式」为依据。正文有七步，附图 1 将局部流程细分至 S9，其中图示的 S9 对应正文「步骤七、记录后续交互和支持结果」。程序按需求书的 S1–S7 编号，其中 S3 同时覆盖掩饰分类和同步非语言证据。

## 专利步骤和代码对应

| 步骤 | 源码与核心函数 | 验证入口 |
| --- | --- | --- |
| S1 事件与严重程度 | `event_builder.build_events`，`severity.evaluate_severity`，`severity.severity_score` 计算加权 (B) | `test_event_builder.py`，`test_formulas.py` 的表一和权重测试 |
| S2 回应关联 | `reply_linker.evaluate_reply`，`association_logit` 计算六特征加权和，`math_utils.sigmoid` 求 (A)，`is_effective` 比较阈值 | `test_formulas.py` 与 `test_pipeline.py` |
| S3 八类输出与非语言 | `masking_classifier.linear_probabilities` 计算 (p_k)，`masking_score` 计算 (M)，`nonverbal.evaluate_nonverbal` 计算 μ、σ、z、u、q、N | `test_formulas.py`、`test_missing_modality.py` |
| S4 事件条件掩饰 | `event_masking.event_masking_score` 计算联合分支与纯文本分支，`representative_reply` 取最大 (H) | `test_formulas.py`、`test_pipeline.py` |
| S5 跨事件分析 | `cross_event.compare_patterns` 比较 8 或 9 维向量，`cross_event_recurrence` 计算 (P)，`event_recurrence` 计算 (F) | `test_formulas.py`、`test_pipeline.py` |
| S6 分级支持 | `risk_support.risk_score` 计算 (U)，`risk_level_index` 分级，`check_safety` 独立检查并触发人工复核 | `test_formulas.py`、`test_safety.py` |
| S7 记录更新 | `pipeline.run_pipeline` 把本次评分按时间追加到本次运行的内存历史，用于同次运行的后续事件 | `test_pipeline.py` 校验历史状态和重放，SQLite 留到阶段 B |

## 工程取值与数据处理

**S1。** 按 UTC 时间排序，在同平台、同会话、同一受影响对象内聚合。事件窗口以首条攻击为起点，主题条件采用消息与组内攻击文本的最大字符一元和二元计数余弦。报告记录每次合并命中的已有消息和余弦值。关键词规则列于 `event_builder.py`，攻击性与伤害性按表一评分，事件取各消息在该维度的最大值。明确指向取 `g=1`，经同会话回复或引用解析取 `g=0.5`，规则无法定位对象时在输入质量中列出该消息。摘要按正文固定模板生成，以 UTC+08:00 显示时间、以顿号连接发起者和类型。事件 ID 是平台、会话、对象与消息 ID 集合的稳定哈希。

**S1 重复性。** 默认 `within_event` 使用事件内攻击次数。可选 `similar_attacks` 在配置的历史时间范围内，按同平台、同发起者、同受影响对象及文本余弦达到 `tau_s` 匹配历史攻击，以一个当前攻击为起点取证据数最多的结果。两种口径分别计算，报告记录选用口径、计数和证据消息。0.7 的行为相似阈值来自专利实施例，0.65 的事件主题阈值来自工程配置。

**S2。** 回应发送者须为本事件受影响对象，且与事件共享平台和会话。以事件结束时间起算，默认取 30 分钟内的发言，进行中发言单独列出。 `S` 在演示中使用事件摘要与回应的字符余弦，通用余弦函数保留正负号。词向量为空时 `A=null`，即使显式回复关系存在也需要完整六特征参与打分。 `D/G` 由 `reply_linker.py` 的指代线索、元数据及事件正文的主题余弦产生，统一标为 `heuristic_fallback`。工程系数 `b0…b6` 可在 YAML 调整。

**S3 文本。** 八类标签采用需求书顺序。 `joint_representation` 实现 `concat(e,r,abs(e-r))`，其输出长度取决于输入向量维度。 `linear_probabilities` 在独立算例中使用测试用 W 和 b，阶段 A 的入口计算使用触发词产生的 0/1 规则分数。JSON 中的 `p` 保留为 `null`，规则分数、触发词与 (M) 的输入权重可查。

**S3 非语言。** 当前观测的时间窗需覆盖回应发送时刻，参考观测需处于事件开始前的局部窗口。默认采用事件前 60 分钟，至少 3 个非重叠的有效观测，使用总体标准差（分母为 (n)）。本阶段把标准差低于配置下限的特征列为不可用。有效特征权重总和为零时返回 `N=null`。内置当前值 45 和参考值 50、55、60 为合成幅值，`simulated_no_face` 是测试输入中的缺失理由。所有实际媒体读取和长度、质量核验属于阶段 D。

**S4–S5。** 代表回应按 (H) 最大选择，同分时优先时间较早、再按作用域 ID 顺序。 `tau_p=30` 来自实施例，原文未固定时间单位，程序以天计算并以事件开始时刻做差。 (F) 统计第二窗口内同平台、同对象、当前之前已结束的事件，排除当前事件。 (P) 比较以前已结束且代表回应也发生在当前事件开始前的同源模式，两边均有可用 (N) 时用 9 维，其余使用 8 维。历史 (H_m) 未达阈值时只参与分母。累计可比较模式未达三个时以 `P=0` 参与 (U) 并保留 `insufficient_history`，零范数比较记录 `Sim=null` 和零贡献。没有有效回应时 (H)、(P) 和 (U) 保留为 `null`。

**S6–S7。** 风险计算和独立安全检查分别执行。阶段 A 的安全词表是工程演示规则，检查事件攻击消息以及该对象在事件期间或候选时间窗内的消息。触发时记录命中项，最终等级标为高风险，保留原计算等级和 (U)。程序输出静态许可策略映射及人工复核标记。SQLite、支持回复与模型推理介入在后续阶段完成。
