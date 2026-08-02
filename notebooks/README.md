# Tutorial notebooks

核心 notebook 默认可以离线运行。实时 API、模型下载和训练只有在教师或学员明确开启后才会执行。

## 推荐顺序

1. `01_prediction_from_zero_shot_to_icl.ipynb`
   从一个风险选择预测出发，理解 token log probability、重归一化、损失，以及同一参与者的历史示例能否改善后续预测。
2. `02_representation_from_hidden_states_to_reasoning_trajectories.ipynb`
   从开放权重模型的隐藏状态出发，进行逐层选择读取、降维和句子级推理轨迹分析。高成本模型运行已缓存。
3. `03_explanation_annotation_to_model_discovery.ipynb`
   对 masked reasoning 进行逐句 annotation，验证编码是否可靠、是否携带 held-out 行为信息，再将支持的计算成分转化为受限候选模型和诊断试次。
4. `04_optional_local_models_and_adaptation.ipynb`
   选修：本地模型、隐藏状态提取、监督微调和轻量参数适配。

## 数据角色

- `data/behavioral_expanded_public_slice.csv`：32 个匿名参与者，每人 40 个有序试次。前段试次作为同一参与者的可观察历史，后段试次用于测试个体内泛化。
- `data/c13k_tutorial_sample.csv`：Choice13K 的条件层面群体数据，用于群体基线和扩展练习，不用于个体化历史。
- `results/representation/`：表征与 explanation 章节使用的缓存隐藏状态、句子文本、降维图和推理轨迹。Notebook 3 默认在 explicit preference claim 之前截断文本。

## 密钥

本地运行时，从仓库根目录复制 `.env.example` 为 `.env`，再填写：

```text
DEEPSEEK_API_KEY=
ZAI_API_KEY=
```

不要把密钥写进 notebook。Colab 用户应使用 Colab Secrets。模型名称和接口能力可能更新，运行前请查看服务商的官方文档。

更完整的安装说明、硬件路径、常见问题和课前阅读见仓库根目录的 `README.md`。
