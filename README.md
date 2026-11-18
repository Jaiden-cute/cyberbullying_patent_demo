# 青少年网络欺凌回应关联原型

本地 Python 原型按说明书具体实施方式处理事件、回应、掩饰表达、跨事件历史、风险和支持。用户界面提供聊天输入、对象选择、支持回复及历史回看。示例事件和训练样本均为合成数据。

## Windows PowerShell 启动

```powershell
# 在克隆后的仓库根目录执行
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-models.txt
.\.venv\Scripts\python.exe -X utf8 -m scripts.fetch_sbert
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

打开终端给出的本机 URL，默认端口 8501。首页为「用户交互」，默认通过规则演示给出支持回复。只录制 A、B、C 文本流程时可改装 `requirements.txt`，并跳过 `scripts.fetch_sbert`。如需真实分类器，在「分析设置」选择「模型辅助分析」，使用随附分类头和下载的编码器。

## 用户端操作

选择一个虚构场景、粘贴对话或导入 UTF-8 的 JSON / CSV，再选择「当前受影响对象」并点击「开始智能分析」。首页展示本次的共情回复、支持建议、来源及人工复核提示。既往共情回复在侧栏「历史记录」中查看。完整计算记录可用原有命令行导出。

粘贴格式为「带时区的时间 | 发言人 | 消息内容」，一行一条，也支持原消息结构或简化的 JSON 数组。简化的 JSON / CSV 使用 `timestamp`、`sender`、`text`，可附 `message_id`、`reply_to_id`、`quoted_message_id`、`mentions`、`platform`、`conversation_id`、`target` 或对应的原字段。缺少 ID 时按消息内容和会话生成。消息确实指向所选对象但原文没有标注时，可展开「补充缺失的指向信息」逐条确认。

录制 A / B 时，依次单独提交 `synthetic_scene_A_B01.json`、`synthetic_scene_B02.json`、`synthetic_scene_B03.json`，保留同一个 SQLite 路径。场景 C 使用 `synthetic_scene_C.json`。新一轮录屏可以给 `PATENT_HISTORY_DB` 指定新文件，原本的历史仍可以用旧路径打开。[逐镜头操作](docs/DEMO_SCRIPT.md)

## 数据、训练与测试

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.generate_training_data
.\.venv\Scripts\python.exe -X utf8 -m scripts.train_mask_head
.\.venv\Scripts\python.exe -X utf8 -m pytest -vv
```

数据生成脚本根据 `data/synthetic_annotation_source.json` 的逐条标注写出训练和留出 CSV。CSV 按事件情境分组，两个分组集合无交集，脚本另外验证事件摘要、回复重复及回复近重复。训练脚本使用实际 Sentence-BERT 嵌入和 `BCEWithLogitsLoss` 学习 W、b，按同一公式用重载权重对留出集预测，写出模型和 `reports/mask_head_training.json`。数据、权重、报告和界面均标记 `synthetic_training_demo`。F1 只使用独立合成场景计算，项目未进行真实青少年数据验证。可以用 `pytest -m "not model"` 运行无需权重的测试。

## 命令行回放

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.cli --mode demo --text-only --output reports/demo_text.json
.\.venv\Scripts\python.exe -X utf8 -m src.cli --mode trained --no-qwen --history state/cli_history.sqlite3 --output reports/hybrid_run.json
```

`--messages` 接受 UTF-8 JSON 数组或 CSV。CSV 和原消息规范的字段相同，`mention_ids`、`target_user_ids` 写成 JSON 数组，时间使用带时区 ISO 8601，`source_type` 和 `consent_status` 标注合成或授权本地数据。使用 `--features` 导入结构化模拟观测，用 `--reviews` 导入人工核对 JSON，格式见 `schemas.py`。运行默认的合成消息时自动使用合成特征文件，`--text-only` 显式关闭。导出完整 JSON 可检查候选的六项关联特征、八类 p 或规则分数、特征质量、代表回应、历史衰减和风险支持来源。

SQLite 默认由界面写入 `state/history.sqlite3`，可用 `PATENT_HISTORY_DB` 为界面选择另一个本地文件。CLI 由 `--history` 选择数据库，省略时只用本次运行内历史。重复提交同一事件更新其计算记录，已写的人工跟进保留。历史页可按对象查询和回看，合成记录清理需先勾选确认。

## 推理来源

| 模式 | 运行条件与结果 |
| --- | --- |
| DEMO | 字符余弦、启发式 D/G、规则八类分数与安全模板。规则值写入 `non_model_score`，`p` 为 null。 |
| HYBRID | 真实模型与演示或回退组件同时使用。逐字段记录 Sentence-BERT、分类头、Qwen、规则和模板来源。 |
| MODEL | 实际 Sentence-BERT 和分类头输出八类概率，且所需 Qwen 关联判断及两阶段支持输出通过校验。训练来源仍显示为 `synthetic_training_demo`。 |
| UNAVAILABLE | 显式要求模型且所需编码能力不可用。受影响公式保留未计算状态。 |

Qwen 服务可选。复制 `.env.example` 为 `.env` 后配置服务地址和模型 ID，再在「分析设置」选择「模型辅助分析」并勾选「使用 AI 生成回复」，或以 `--mode trained` 运行 CLI。D/G 由同一次调用返回，字段无效时相应项按说明书取 0 并附理由。支持由策略和回复两次调用生成，输出校验失败时展示安全模板。命令行报告保留组件状态和原因。

验收使用官方 Qwen2.5-7B-Instruct Q4_K_M 两片 GGUF、llama.cpp Windows x64 CPU 版和 `http://127.0.0.1:8765/v1`。在新设备使用相同部署时，先取得官方权重和 CPU 运行程序，将权重放入 `models/`，将程序解压到 `state/llama-cpp/`。将自己生成的本机服务密钥保存到 `state/qwen-api-key.txt`，然后在 `.env` 中填入同一个 `QWEN_API_KEY` 和上面的 `QWEN_BASE_URL`。权重、运行程序与密钥路径的说明见 [模型文件说明](models/README.md)。

```powershell
.\state\llama-cpp\llama-server.exe -m models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf --host 127.0.0.1 --port 8765 --alias Qwen2.5-7B-Instruct --api-key-file state/qwen-api-key.txt -ngl 0 -t 8 -c 3072 -np 1
```

启动本机服务后，在界面勾选「使用 AI 生成回复」。首次拉取的仓库包含源码、合成数据、训练好的八类分类头和合成验收报告。公开基础模型按需下载，虚拟环境和运行数据在各自设备生成。

## 音视频与录屏

用户端仅分析文本。结构化模拟观测可用原有命令行进行公式测试。真实音视频特征提取和实际素材验证尚未完成。

## 启动排错

```powershell
py -3.12 --version
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -X utf8 -m pytest -q
```

如端口 8501 已占用，启动时可指定 `--server.port 8502`。模型不可用时，纯文本录屏可以选择「规则演示」。

录屏步骤含点击位置、界面预期结果和逐段验收点。[录屏脚本](docs/DEMO_SCRIPT.md) · [实际验收记录](docs/VALIDATION.md) · [实现及工程口径](IMPLEMENTATION_NOTES.md) · [模型文件说明](models/README.md)
