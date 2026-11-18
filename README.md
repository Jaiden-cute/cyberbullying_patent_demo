# 青少年网络欺凌回应关联原型

本地 Python 原型按说明书具体实施方式处理事件、回应、八类掩饰表达、非语言证据、跨事件重复、风险和支持，保存可追溯的 SQLite 历史。界面可导入消息、人工修正事件类别和参与者、查看逐步计算、导出报告。内置三期事件和训练样本均为合成数据。

## Windows PowerShell 启动

```powershell
# 在克隆后的仓库根目录执行
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-models.txt
.\.venv\Scripts\python.exe -X utf8 -m scripts.fetch_sbert
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

打开终端给出的本机 URL，默认端口 8501。项目已有训练分类头，首次部署可直接使用随附权重。仅运行规则演示时，安装 `requirements.txt` 即可，在界面选择「规则演示」。

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

Qwen 服务可选。复制 `.env.example` 为 `.env` 后配置服务地址和模型 ID，再在界面勾选「使用已配置的 Qwen 推理服务」，或以 `--mode trained` 运行 CLI。关联的 D/G 由同一次调用返回；请求失败或字段无效时相应项按说明书取 0 并附理由。支持由策略和回复两次调用生成，输出校验失败时展示安全模板。运行页的「参数与诊断」列出实际组件状态。

验收使用官方 Qwen2.5-7B-Instruct Q4_K_M 两片 GGUF、llama.cpp Windows x64 CPU 版和 `http://127.0.0.1:8765/v1`。在新设备使用相同部署时，先取得官方权重和 CPU 运行程序，将权重放入 `models/`，将程序解压到 `state/llama-cpp/`。将自己生成的本机服务密钥保存到 `state/qwen-api-key.txt`，然后在 `.env` 中填入同一个 `QWEN_API_KEY` 和上面的 `QWEN_BASE_URL`。权重、运行程序与密钥路径的说明见 [模型文件说明](models/README.md)。

```powershell
.\state\llama-cpp\llama-server.exe -m models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf --host 127.0.0.1 --port 8765 --alias Qwen2.5-7B-Instruct --api-key-file state/qwen-api-key.txt -ngl 0 -t 8 -c 3072 -np 1
```

启动本机服务后，在界面勾选 Qwen。首次拉取的仓库包含源码、合成数据、训练好的八类分类头和合成验收报告。公开基础模型按需下载，虚拟环境和运行数据在各自设备生成。

## 音视频与录屏

当前支持纯文本及结构化模拟观测，后者会按回应对象和时间窗验证，并逐特征展示 μ、σ、z、u、q 和 N。产品代码提供 `MediaFeatureExtractor` 接口与许可目录内路径校验。真实音视频提取器和授权素材联调待实施。展示模拟输入时，界面和报告均显示 `structured_test_features`。

录屏步骤含点击位置、界面预期结果和逐段验收点。[录屏脚本](docs/DEMO_SCRIPT.md) · [实际验收记录](docs/VALIDATION.md) · [实现及工程口径](IMPLEMENTATION_NOTES.md) · [模型文件说明](models/README.md)
