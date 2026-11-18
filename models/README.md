# 模型文件

`mask_head.synthetic_training_demo.json` 保存实际训练得到的 8×1152 权重矩阵和八个偏置，包含标签顺序、Sentence-BERT 版本、输入维度、训练 CSV 的 SHA-256、训练参数和模型版本。加载时核对编码器身份、维度及分类头指纹。文件以 JSON 保存数值权重。

Sentence-BERT 采用需求书配置的 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`。本原型使用官方仓库提交 `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`，放置在 `sbert-e8f8c211/`。运行 `python -m scripts.fetch_sbert` 可从该提交下载必要的权重、Tokenizer 和池化配置。编码器报告的实际维度为 384，联合向量维度为 1152。[Sentence-BERT 官方模型卡](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)

分类头的训练来源为 `synthetic_training_demo`，用于验证原型训练、保存与推理流程。独立合成场景的输出见 `reports/mask_head_training.json`。训练和留出样本都由合成数据构成，报告里的 F1 仅对应留出合成集。

Qwen 可另部署为 OpenAI 兼容服务。服务需提供 `/v1/models` 和 `/v1/chat/completions`，模型 ID 应为 `Qwen2.5-7B-Instruct` 或 `Qwen/Qwen2.5-7B-Instruct`。在项目的 `.env` 填写 `QWEN_BASE_URL` 和服务需要的配置。官方 GGUF 版本可供本地 CPU 服务部署。[Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) · [Qwen 官方 GGUF](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF)

本机真实接口验收使用官方 GGUF 提交 `bb5d59e06d9551d752d08b292a50eb208b07ab1f` 和 llama.cpp `b12499`。两片文件都已按发布仓库的 SHA-256 校验。

| GGUF 文件 | SHA-256 |
| --- | --- |
| `qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf` | `dfce12e3862a5283ccfb88221b48480e58745165de856439950d0f22590580db` |
| `qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf` | `539cf93f78e887edea1c04e2d7d8cdaca9d01dae9c9025bcb8accbe29df3d72a` |

在其他设备上可从发布页取得上述两个分片及 Windows x64 CPU 包。[固定版本 GGUF 文件](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/tree/bb5d59e06d9551d752d08b292a50eb208b07ab1f) · [llama.cpp b12499](https://github.com/ggml-org/llama.cpp/releases/tag/b12499)
