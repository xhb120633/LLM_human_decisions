# 用大语言模型理解人类决策：课前准备

本仓库用于 **Using Large Language Models to Understand Human Decision-Making** tutorial。

这份页面目前只完成两件事：

1. 帮助学员在开课前把 Python 环境安装好；
2. 提供一份由浅入深的课前阅读清单。

课程面向心理学、神经科学、行为经济学和管理学等背景的研究者。**不要求机器学习经验，也不要求独立显卡。**

## 课程材料

当前公开材料均为最新版，并按照“科学问题 → 技术实现 → 证据边界”的主线整合：

- [课程 PPT](slides/llm_human_decisions_tutorial.pptx)
- [Notebook 使用顺序与数据说明](notebooks/README.md)
- [Notebook 1：从单次预测到个体历史学习曲线](notebooks/01_prediction_from_zero_shot_to_icl.ipynb)
- [Notebook 2：从隐藏状态到推理轨迹](notebooks/02_representation_from_hidden_states_to_reasoning_trajectories.ipynb)
- [Notebook 3：从解释标注到模型发现](notebooks/03_explanation_annotation_to_model_discovery.ipynb)
- [Notebook 4：本地模型与参数适配（可选）](notebooks/04_optional_local_models_and_adaptation.ipynb)

Notebook 默认使用随仓库发布的小型公开教学切片和缓存模型输出，但 MDS、persona trajectory、downstream decoding 等轻量分析会现场重算。API 调用、Qwen hidden-state extraction、layerwise probe、Text2Decision 训练、model-discovery iteration 和 LoRA 训练的原始代码均保留，并通过显式开关选择现场运行或缓存 fallback。仓库不包含原始参与者数据、模型权重或完整 activation matrices。

## 你需要准备什么

最低要求：

- 一台可以正常使用浏览器的电脑；
- 建议至少 8 GB 内存，16 GB 会更顺畅；
- 能够安装软件，或者能够使用 Google Colab；
- 预留约 30 分钟完成安装和测试。

不需要提前准备：

- NVIDIA 显卡或 CUDA；
- 本地大语言模型；
- 深度学习训练经验；
- 付费 API key。

## 推荐方案：Google Colab

如果你不熟悉 Python 环境配置，建议优先使用 [Google Colab](https://colab.research.google.com/)。它直接在浏览器中运行，不需要在本地安装 Python，也不需要独立显卡。

开课前只需确认：

1. 可以正常登录和打开 Colab；
2. 可以新建一个 notebook；
3. 在代码单元格中运行下面的内容：

```python
print("Colab is ready")
```

看到 `Colab is ready` 即可。课程核心内容使用 CPU 环境就能完成。

Colab 是临时运行环境，断开连接后可能需要重新安装依赖。具体规则见 [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)。

## 本地安装方案

如果希望在自己的电脑上运行，推荐使用 **Python 3.11** 和项目独立的虚拟环境。

### 第一步：安装必要软件

请先安装：

- [Python 3.11](https://www.python.org/downloads/)
- [Git](https://git-scm.com/downloads)

Windows 安装 Python 时，请勾选 **Add Python to PATH**。

### 第二步：下载仓库

打开 PowerShell、Windows Terminal 或 macOS/Linux Terminal：

```bash
git clone https://github.com/xhb120633/LLM_human_decisions.git
cd LLM_human_decisions
```

如果不熟悉 Git，也可以在 GitHub 页面选择 **Code → Download ZIP**，解压后进入项目文件夹。

### 第三步：创建独立环境

#### Windows

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

成功激活后，终端开头通常会出现 `(.venv)`。

如果 PowerShell 阻止激活脚本，可以只对当前窗口临时放宽限制：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 第四步：安装课程依赖

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` 是标准课程环境，包括：

- JupyterLab；
- NumPy、pandas、Matplotlib、Seaborn 和 scikit-learn；
- PyTorch；
- Hugging Face Transformers、Accelerate、SafeTensors 和 SentencePiece；
- OpenAI-compatible API 客户端；
- `.env` 配置支持。

PyTorch 和 Transformers 可以在 CPU 上运行，不要求独立显卡。安装这些包不会自动下载任何模型权重，也不会安装 CUDA、vLLM 或 Unsloth。

如果电脑空间或网络非常受限，并且只准备使用 API 与缓存结果，可以改装轻量备用环境：

```bash
python -m pip install -r requirements-light.txt
```

### 第五步：检查环境

运行：

```bash
python -c "import torch, transformers, numpy, pandas, sklearn, matplotlib, seaborn, openai; print('Environment OK | torch', torch.__version__, '| device', 'cuda' if torch.cuda.is_available() else 'cpu')"
```

如果看到：

```text
Environment OK
```

说明核心环境已经安装成功。

再启动 JupyterLab：

```bash
python -m jupyter lab
```

浏览器中能够看到 JupyterLab 文件页面，即完成全部课前环境准备。

## API key：暂时不是必需的

课程会介绍如何调用 DeepSeek 或智谱 GLM 等 OpenAI-compatible API，但没有 key 也可以完成核心教学内容。

如果之后需要尝试实时 API，可以复制配置模板：

### Windows

```powershell
Copy-Item .env.example .env
```

### macOS / Linux

```bash
cp .env.example .env
```

然后填写 `.env`：

```text
DEEPSEEK_API_KEY=
ZAI_API_KEY=
```

不要把 key 写进公开 notebook、截图或 Git commit。模型名称和接口能力会持续更新，使用前请查看官方文档：

- [DeepSeek API](https://api-docs.deepseek.com/)
- [DeepSeek Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion)
- [GLM 的 OpenAI SDK 兼容接口](https://docs.bigmodel.cn/cn/guide/develop/openai/introduction)

## 没有独立显卡怎么办？

不影响课程核心学习。

| 电脑配置 | 推荐方式 |
|---|---|
| 无独显、8–16 GB 内存 | 安装标准环境；PyTorch 自动使用 CPU。课堂只运行小模型或缓存结果 |
| Apple Silicon | 安装标准环境；PyTorch 可使用 CPU 或 Apple Metal 后端 |
| NVIDIA GPU | 安装标准环境；是否使用 GPU 取决于本机 PyTorch 构建和驱动 |

不建议学员为了课程提前手动安装 CUDA、vLLM 或 Unsloth。标准 `requirements.txt` 已经具备 PyTorch 和 Transformers；没有独显时它们会在 CPU 上工作。需要匹配特定 NVIDIA CUDA 版本的学员，再通过 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 获取对应命令。

## 常见问题

### 找不到 `python` 或 `py`

重新打开终端。如果仍然找不到，请重新安装 Python，并在 Windows 安装界面勾选 **Add Python to PATH**。

### 出现 `ModuleNotFoundError`

通常是 Jupyter 和安装依赖时使用了不同的 Python 环境。请先激活 `.venv`，再从同一个终端运行：

```bash
python -m jupyter lab
```

### 安装速度很慢

优先尝试稳定网络；如果安装始终失败，直接使用 Google Colab，不需要在课前解决复杂的本地网络配置。

### 出现内存或显存不足

核心环境不应占用大量显存。请确认没有安装或运行本地大模型，并改用 Colab CPU 或关闭其他占用内存的软件。

### 不确定是否安装成功

把下面两项结果保留下来，课堂开始时可以快速定位问题：

```bash
python --version
python -m pip --version
```

## 课前阅读

以下书目信息已按论文原始页面核对。链接优先指向期刊、PMLR、OpenReview、会议官网或 arXiv 原始记录，并明确区分正式发表论文、会议 poster/workshop 和 preprint。

不要求逐篇精读。建议先看摘要、关键图和讨论部分，重点思考：研究者到底是在描述、预测、表征，还是解释决策行为？

### 最低预习：约 20–30 分钟

1. **Kahneman, D., & Tversky, A. (1979).** Prospect Theory: An Analysis of Decision under Risk. *Econometrica, 47*(2), 263–292. [DOI](https://doi.org/10.2307/1914185)
   目的：回顾经典风险决策研究如何从选择推断效用和概率权重。

2. **Vaswani, A., et al. (2017).** Attention Is All You Need. *Advances in Neural Information Processing Systems 30*. [arXiv](https://arxiv.org/abs/1706.03762)
   目的：只需形成一个直觉——token 被转成向量，经过多层上下文化计算，再用于预测后续 token；不要求掌握 Transformer 公式。

3. **Brown, T. B., et al. (2020).** Language Models are Few-Shot Learners. *Advances in Neural Information Processing Systems 33*. [arXiv](https://arxiv.org/abs/2005.14165)
   目的：理解为什么不更新模型参数，仅提供若干上下文示例也可能改变模型预测。

4. **Binz, M., & Schulz, E. (2023).** Using cognitive psychology to understand GPT-3. *Proceedings of the National Academy of Sciences, 120*(6), e2218523120. [PubMed](https://pubmed.ncbi.nlm.nih.gov/36730192/) · [DOI](https://doi.org/10.1073/pnas.2218523120)
   目的：思考把语言模型放进心理实验能够提供什么证据，以及不能支持什么结论。

### 按课程主题选读

#### 1. 把大语言模型作为行为模拟器或预测模型

- **Aher, G. V., Arriaga, R. I., & Kalai, A. T. (2023).** Using Large Language Models to Simulate Multiple Humans and Replicate Human Subject Studies. *Proceedings of the 40th International Conference on Machine Learning*, PMLR 202, 337–371. [PMLR](https://proceedings.mlr.press/v202/aher23a.html)
- **Binz, M., et al. (2025).** A foundation model to predict and capture human cognition. *Nature, 644*, 1002–1009. [Nature](https://www.nature.com/articles/s41586-025-09215-4)
- **Xie, H., Xiong, H., & Wilson, R. C. (2024).** Evaluating Predictive Performance and Learning Efficiency of Large Language Models with Think Aloud in Risky Decision Making. *Computational Cognitive Neuroscience 2024*, poster A88. [CCN abstract](https://2024.ccneuro.org/poster/?id=106) · [paper PDF](https://2024.ccneuro.org/pdf/67_Paper_authored_CCN_evaluating_efficiency_LLM_fina_published.pdf)

#### 2. 从语言和内部状态研究表征

- **Xie, H., Xiong, H., & Wilson, R. C. (2023).** Text2Decision: Decoding Latent Variables in Risky Decision Making from Think Aloud Text. *NeurIPS 2023 AI for Science Workshop*, poster. [OpenReview](https://openreview.net/forum?id=fEoemPDicz)
- **Xie, H., Xiong, H.-D., & Wilson, R. C. (2024).** From Strategic Narratives to Code-Like Cognitive Models: An LLM-Based Approach in A Sorting Task. *First Conference on Language Modeling (COLM 2024)*. [OpenReview](https://openreview.net/forum?id=1Tny4KgGO2)

#### 3. 从解释走向可执行模型与模型发现

- **Li, M. Y., Fox, E., & Goodman, N. (2024).** Automated Statistical Model Discovery with Language Models. *Proceedings of the 41st International Conference on Machine Learning*, PMLR 235, 27791–27807. [PMLR](https://proceedings.mlr.press/v235/li24v.html)
- **Rmus, M., Jagadish, A. K., Mathony, M., Ludwig, T., & Schulz, E. (2025).** Generating Computational Cognitive Models using Large Language Models. *Advances in Neural Information Processing Systems 38 (NeurIPS 2025)*, main conference track. The paper introduces GeCCo (Guided generation of Computational Cognitive Models). [NeurIPS proceedings](https://papers.nips.cc/paper_files/paper/2025/hash/7f14c9df045c5b58893a87079d16d2b3-Abstract-Conference.html) · [arXiv](https://arxiv.org/abs/2502.00879)
- **Zhu, J.-Q., Xie, H., Arumugam, D., Wilson, R. C., & Griffiths, T. L. (2026).** Using Reinforcement Learning to Train Large Language Models to Explain Human Decisions. *International Conference on Learning Representations (ICLR 2026)*. [OpenReview](https://openreview.net/forum?id=coJPBEZ9Te) · [arXiv](https://arxiv.org/abs/2505.11614)
- **Xie, H., Jagadish, A. K., Pan, L., & Wilson, R. C. (2026).** Think-Aloud Reshapes Automated Cognitive Model Discovery Beyond Behavior. *arXiv preprint arXiv:2605.05091*. [arXiv](https://arxiv.org/abs/2605.05091)

#### 4. 参数适配与人类反馈

- **Ouyang, L., et al. (2022).** Training language models to follow instructions with human feedback. *Advances in Neural Information Processing Systems 35*. [arXiv](https://arxiv.org/abs/2203.02155)

### 阅读时的三个问题

1. 这项研究实际观察了什么数据？
2. 它支持的是描述、预测、表征、解释，还是机制层面的主张？
3. 什么额外数据、干预或泛化测试能够区分竞争解释？

## 环境文件

```text
requirements.txt                 # 标准课程环境，包含 PyTorch 与 Transformers
requirements-light.txt           # API / 分析备用环境，不包含本地神经模型
requirements-local-models.txt    # 兼容入口，目前等同于标准环境
requirements-representation.txt  # 教师级表征复现环境；现在不用安装
.env.example                     # API key 模板
```

如果 `Environment OK` 检查通过，并且能够打开 JupyterLab 或 Google Colab，课前准备就完成了。

## License

除非文件中另有说明，本仓库由作者原创的源代码、notebook、文档、课件源文件和生成的教学材料采用 [Apache License 2.0](LICENSE) 授权：

```text
Copyright 2026 Hanbo Xie
SPDX-License-Identifier: Apache-2.0
```

以下内容不包含在本项目的 Apache 2.0 授权中：

- `notebooks/data/` 中来自或衍生自第三方数据集的文件；
- `notebooks/results/` 与 `artifacts/` 中受模型、API、数据集或其他上游条款约束的缓存结果；
- 课件或文档中明确引用的第三方论文、图片、商标及其他材料。

其中 `notebooks/data/c13k_tutorial_sample.csv` 是从 [choices13k](https://github.com/jcpeterson/choices13k) 制作的教学切片。使用或再分发这些材料前，请核对并遵守相应上游条款；本仓库的许可证不会替代或扩大任何第三方授权。使用 Choice13K 开展研究时，也请引用其原始论文。
