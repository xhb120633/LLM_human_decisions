# Tutorial notebooks

核心 notebook 默认可以离线运行。实时 API、模型下载和训练只有在教师或学员明确开启后才会执行。缓存用于降低课堂门槛，但不替代方法：关键结果均保留生成、训练和分析代码。

## 三种运行层级

1. **课堂离线层**：直接运行全部 notebook。API 和 GPU 开关保持 `False`；原始汇总分析会从公开的较低层数据重新计算，只有昂贵模型输出读取缓存。
2. **API 实验层**：填写 `.env` 后开启 `RUN_SINGLE_API`、`RUN_CURVE_API`、`RUN_BALANCED_ICL_API`、`RUN_ANNOTATION_API` 或 `RUN_DEEPSEEK_DISCOVERY`。Notebook 会生成新的原始响应和逐试次指标。
3. **本地模型复现层**：在有 NVIDIA GPU 的环境中开启 Notebook 2/4 的模型开关。公开脚本覆盖 Qwen choice recovery、sentence-boundary hidden-state extraction、layerwise probe、Text2Decision 数据构建与训练、reasoning inference，以及 LoRA 训练与 held-out evaluation。

每个昂贵单元均采用相同结构：先显示实际执行代码，再由布尔开关决定运行该代码还是读取随仓库发布的 fallback。这样无显卡学员可以完成课程，有资源的学员也能从上游重新生成结果。

## 推荐顺序

1. `01_prediction_from_zero_shot_to_icl.ipynb`
   从一个风险选择预测出发，理解 token log probability、重归一化、损失，以及同一参与者的历史示例能否改善后续预测。
2. `02_representation_from_hidden_states_to_reasoning_trajectories.ipynb`
   从开放权重模型的隐藏状态出发，进行逐层选择读取、降维和句子级推理轨迹分析。MDS、persona time course 和 downstream choice decoding 默认现场重算；Qwen forward pass 与 Text2Decision 训练可选现场复现。
3. `03_explanation_annotation_to_model_discovery.ipynb`
   对 masked reasoning 进行逐句 annotation，验证编码是否可靠、是否携带 held-out 行为信息，再将支持的计算成分转化为受限候选模型和诊断试次。
4. `04_optional_local_models_and_adaptation.ipynb`
   选修：本地模型、隐藏状态提取、监督微调和轻量参数适配。

## 数据角色

- `data/behavioral_expanded_public_slice.csv`：32 个匿名参与者，每人 40 个有序试次。前段试次作为同一参与者的可观察历史，后段试次用于测试个体内泛化。
- `data/c13k_tutorial_sample.csv`：Choice13K 的条件层面群体数据，用于群体基线和扩展练习，不用于个体化历史。
- `results/representation/`：公开的句子级中间表与少量注册报告。它们允许无 GPU 环境重算统计分析；多 GB 原始 activation matrices 和模型权重不发布。

## Notebook 2 的完整上游链

Notebook 2 中保留以下顺序的可执行命令：

```text
public masked reasoning table
  -> Qwen A/B logits and renormalized choice probabilities
  -> layer-by-layer sentence-boundary states
  -> grouped layerwise choice probes

Choice13K problem JSON
  -> option text + computed 12D targets
  -> Qwen layer-15 option-text states
  -> grouped Text2Decision training with early stopping
  -> frozen readout applied to reasoning states
  -> MDS, persona trajectories, and held-out choice decoding
```

复现完整 Text2Decision 训练时，通过环境变量提供 Choice13K problem 文件：

```powershell
$env:CHOICE13K_PROBLEMS_JSON="C:\path\to\c13k_problems.json"
```

仍然不会把完整 Choice13K、人类原始数据、activation matrices 或模型权重提交到仓库。

## 密钥

本地运行时，从仓库根目录复制 `.env.example` 为 `.env`，再填写：

```text
DEEPSEEK_API_KEY=
ZAI_API_KEY=
```

不要把密钥写进 notebook。Colab 用户应使用 Colab Secrets。模型名称和接口能力可能更新，运行前请查看服务商的官方文档。

更完整的安装说明、硬件路径、常见问题和课前阅读见仓库根目录的 `README.md`。
