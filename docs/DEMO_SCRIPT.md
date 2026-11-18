# 录屏演示脚本

建议录制 6 至 8 分钟。从项目目录启动 `streamlit run app.py --server.address 127.0.0.1`，打开终端显示的地址。为便于复现，录制用新的 SQLite 文件，或在「历史记录」清空已有合成记录。使用 `data/synthetic_messages.json` 的默认三期场景，前半段选「规则演示」，全片只展示虚构对象。末段用真实 Sentence-BERT 和合成训练分类头展示概率来源。

| 时间 | 页面操作 | 预期内容 |
| --- | --- | --- |
| 00:00–00:35 | 展示标题，选择「加载合成示例」和「三期合成事件（同一对象）」。保持模拟特征复选框关闭。 | 10 条消息的时间、发送者、目标、回复和引用 ID。说明模式与数据来源。 |
| 00:35–01:20 | 展开「人工核对事件类别与参与者」查看预览。点击「运行 S1–S7」，选 10 月 1 日，打开「S1 事件」。 | 两条攻击合并、五字段摘要、规则来源，g=1、a=1、r=0.5、h=0.5，B=0.75。 |
| 01:20–02:10 | 打开「S2 回应关联」。再切到 10 月 11 日继续看 S2。 | 阈值 0.7；首个事件两个通过回应，数学作业消息 A≈0.172607；10 月 11 日无引用 ID 的回应 A≈0.730927。 |
| 02:10–03:05 | 回到 10 月 1 日，打开「S3 八类掩饰」和「S3 非语言」，再打开「S4–S5」。 | DEMO 显示八项规则值。纯文本下 N 未计算，代表回应 oct01-reply-1 使用文本分支，H≈0.288271。 |
| 03:05–04:10 | 勾选「加入结构化模拟非语言特征」并重跑。查看 10 月 1 日 S3 非语言，再查看 10 月 21 日 S4–S5。 | 结构化模拟观测来源。RMS 参考 50、55、60，μ=55，σ≈4.082483，q=1，N≈0.913662，视频 q=0；第 3 起事件有两条历史比较，P≈0.373361，F≈0.486583。 |
| 04:10–05:05 | 选择 10 月 21 日，打开「S6 风险与支持」「S7 保存状态」，下载 JSON。 | U≈0.459682，分数等级重点关注，最终高风险并人工复核。支持来源 safe_template、SQLite 保存成功。 |
| 05:05–05:55 | 侧栏进入「历史记录」，筛选同学丙，打开一条保存记录。返回分析页并重复运行。 | 同一对象三条历史、历史跟进入口和重复执行后事件数仍为三。 |
| 05:55–07:00 | 选「Sentence-BERT + 已训练分类头」重新运行，打开 S3、参数与诊断。 | HYBRID 模式下显示真实八类 Sigmoid 概率、`synthetic_training_demo`、模型版本与 S 的 Sentence-BERT 来源。Qwen 的来源和状态按当次服务配置展示。 |
| 07:00–07:20 | 打开训练报告与当前实现范围。 | 区分留出合成集结果、原型可追溯计算、真实音视频接口状态和人工复核要求。 |

若已配置真实 Qwen 服务，可另录一次启用 Qwen 的 S2 和 S6。检查参数页的模型 ID、各 D/G 字段来源、策略和回复状态，并按实际结果保留可用或回退标志。当前音视频步骤用结构化模拟观测录制。真实媒体验证在授权素材和实际特征提取器就绪后单独录制。

| 专利步骤 | 源码入口 | 代表测试 | 页面与片段 |
| --- | --- | --- | --- |
| S1 | `event_builder.build_events`，`severity.evaluate_severity` | `test_aggregation_and_exact_summary_template` | S1 事件，00:35 |
| S2 | `reply_linker.evaluate_reply`，`QwenAdapter.judge_relations` | `test_effective_explicit_and_implicit_replies_and_unrelated_message` | S2 回应关联，01:20 |
| S3 | `MaskHead.predict`，`nonverbal.evaluate_nonverbal` | `test_saved_classifier_reproduces_real_holdout_probabilities`，`test_reference_mean_population_std_and_n_with_missing_video` | S3 两个标签页，02:10 / 03:05 / 05:55 |
| S4 | `event_masking.event_masking_score`，`representative_reply` | `test_multimodal_and_text_h_from_requirements` | S4–S5，02:10 / 03:05 |
| S5 | `cross_event.cross_event_recurrence` | `test_p_f_u_independent_values_from_requirements` | S4–S5，03:05 |
| S6 | `risk_support.evaluate_risk`，`support.generate_support` | `test_safety_override_works_without_associated_reply_or_u`，`test_two_support_calls_and_allowlist_validation` | S6 风险与支持，04:10 |
| S7 | `HistoryStore.analyze`，`save_follow_up` | `test_separate_submissions_use_recent_persisted_events_and_deduplicate` | S7 保存状态与历史记录，04:10 / 05:05 |
