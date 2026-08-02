from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from textwrap import dedent

import nbformat as nbf
import numpy as np
import pandas as pd

from build_participant_notebook import build_notebook as build_participant_notebook


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "notebooks"
DATA_DIR = OUT / "data"
DEFAULT_SOURCE_DIR = Path(
    r"C:\Users\Hanbo\Documents\GitHub\scaling_up_RL\human_data\choices13k\original"
)
SOURCE_DIR = Path(os.environ.get("CHOICE13K_SOURCE_DIR", DEFAULT_SOURCE_DIR))
SOURCE_CSV = SOURCE_DIR / "c13k_selections.csv"
SOURCE_JSON = SOURCE_DIR / "c13k_problems.json"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str, tags: list[str] | None = None):
    cell = nbf.v4.new_code_cell(dedent(text).strip())
    if tags:
        cell.metadata["tags"] = tags
    return cell


def notebook(cells: list, title: str):
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
            "colab": {"name": title, "provenance": []},
        }
    )
    return nb


def money(value: float) -> str:
    sign = "-$" if value < 0 else "$"
    value = abs(float(value))
    shown = f"{value:,.3f}".rstrip("0").rstrip(".")
    return f"{sign}{shown}"


def lottery_text(lottery: list[list[float]], ambiguous: bool = False) -> str:
    parts = []
    nonzero = [(float(p), float(v)) for p, v in lottery if float(p) > 1e-12]
    for probability, value in nonzero:
        if ambiguous:
            parts.append(f"an unknown chance of {money(value)}")
        elif abs(probability - 1.0) < 1e-10:
            parts.append(f"{money(value)} for sure")
        else:
            pct = f"{100 * probability:.3f}".rstrip("0").rstrip(".")
            parts.append(f"{pct}% chance of {money(value)}")
    return "; ".join(parts)


def lottery_stats(lottery: list[list[float]]) -> dict[str, float]:
    probabilities = np.asarray([float(x[0]) for x in lottery])
    outcomes = np.asarray([float(x[1]) for x in lottery])
    keep = probabilities > 1e-12
    probabilities = probabilities[keep]
    outcomes = outcomes[keep]
    probabilities = probabilities / probabilities.sum()
    expected_value = float(np.sum(probabilities * outcomes))
    variance = float(np.sum(probabilities * (outcomes - expected_value) ** 2))
    return {
        "ev": expected_value,
        "sd": variance**0.5,
        "min": float(outcomes.min()),
        "max": float(outcomes.max()),
        "n_outcomes": int(len(outcomes)),
    }


def make_teaching_sample() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    selections = pd.read_csv(SOURCE_CSV)
    lotteries = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))
    enriched = []
    for row_id, row in selections.iterrows():
        lottery_a = lotteries[str(row_id)]["A"]
        lottery_b = lotteries[str(row_id)]["B"]
        stats_a = lottery_stats(lottery_a)
        stats_b = lottery_stats(lottery_b)
        enriched.append(
            {
                "row_id": row_id,
                **row.to_dict(),
                "lottery_A": lottery_text(lottery_a, ambiguous=False),
                "lottery_B": lottery_text(lottery_b, ambiguous=bool(row["Amb"])),
                "ev_A": stats_a["ev"],
                "ev_B": stats_b["ev"],
                "sd_A": stats_a["sd"],
                "sd_B": stats_b["sd"],
                "min_A": stats_a["min"],
                "min_B": stats_b["min"],
                "max_A": stats_a["max"],
                "max_B": stats_b["max"],
                "n_outcomes_A": stats_a["n_outcomes"],
                "n_outcomes_B": stats_b["n_outcomes"],
                "majority_choice": "B" if row["bRate"] > 0.5 else "A",
            }
        )
    full = pd.DataFrame(enriched)
    # Clean core subset: explicit probabilities, no feedback history, one row per problem.
    sample = full.loc[
        (~full["Feedback"].astype(bool)) & (~full["Amb"].astype(bool))
    ].sort_values("row_id")
    path = DATA_DIR / "c13k_tutorial_sample.csv"
    sample.to_csv(path, index=False)
    return path


COMMON_SETUP = r'''
from pathlib import Path
import json
import os
import math
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

SEED = 3000
rng = np.random.default_rng(SEED)
pd.set_option("display.max_colwidth", 120)
plt.style.use("seaborn-v0_8-whitegrid")
'''


LOAD_DATA = r'''
def find_data_file():
    candidates = [
        Path("data/c13k_tutorial_sample.csv"),
        Path("c13k_tutorial_sample.csv"),
        Path("/content/data/c13k_tutorial_sample.csv"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find c13k_tutorial_sample.csv. Keep the data/ folder next to the notebook, "
        "or upload the CSV to Colab."
    )


DATA_PATH = find_data_file()
data = pd.read_csv(DATA_PATH)

# This tutorial predicts a binary majority label. Exact 50/50 rows have no majority.
data = data.loc[data["bRate"] != 0.5].copy()
data["majority_B"] = (data["bRate"] > 0.5).astype(int)
data["ev_diff"] = data["ev_B"] - data["ev_A"]
data["risk_diff"] = data["sd_B"] - data["sd_A"]
data["worst_diff"] = data["min_B"] - data["min_A"]
data["best_diff"] = data["max_B"] - data["max_A"]

print(f"Loaded {len(data):,} real Choice13K condition rows from {DATA_PATH}")
print(f"Unique problem IDs: {data['Problem'].nunique():,}")
data.head(3)
'''


PROMPT_FUNCTIONS = r'''
SYSTEM_PROMPT = (
    "You estimate aggregate human behavior in a risky-choice experiment. "
    "Do not choose for yourself."
)
LABEL_SYSTEM_PROMPT = (
    "You predict the human majority label in a risky-choice experiment. "
    "Respond with exactly one label: A or B."
)


def format_trial(row, include_answer=False):
    feedback = (
        "Outcome feedback is shown after each choice."
        if bool(row["Feedback"])
        else "No outcome feedback is shown."
    )
    text = (
        f"Option A: {row['lottery_A']}\n"
        f"Option B: {row['lottery_B']}\n"
        f"Condition: {feedback} Block {int(row['Block'])}."
    )
    if include_answer:
        text += f"\nObserved aggregate p_B: {row['bRate']:.3f}"
    return text


def zero_shot_messages(row):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                format_trial(row)
                + "\nParticipants faced this pair five times. Predict the average fraction "
                "of decisions allocated to Option B. Return JSON only: "
                '{"p_B": number between 0 and 1}.'
            ),
        },
    ]


def label_messages(row):
    return [
        {"role": "system", "content": LABEL_SYSTEM_PROMPT},
        {"role": "user", "content": format_trial(row) + "\nAnswer:"},
    ]
'''


def build_notebook_1() -> nbf.NotebookNode:
    cells = [
        md('''
        # Predicting Human Risky Choice with LLMs
        ## From zero-shot evaluation to in-context learning

        **Core hands-on notebook · 55–70 minutes**

        We will use real aggregate human choices from **Choice13K** to ask a deliberately narrow question:

        > Given two lotteries and an experimental condition, can a model predict the aggregate human B-choice rate (and, secondarily, its majority label)?

        By the end, you will be able to:

        1. construct a leakage-resistant train/test split;
        2. compare LLM predictions with simple behavioral baselines;
        3. inspect top-token log probabilities and re-normalize over valid labels;
        4. compare zero-shot, random few-shot, and retrieved few-shot prompts;
        5. state precisely what these results do—and do not—show.

        **Important data limitation.** `c13k_selections.csv` contains choice rates aggregated by problem and condition. It does **not** contain a longitudinal history for each participant. Therefore the ICL exercise below uses *other problems as population-level demonstrations*; it is not participant personalization.
        '''),
        md('''
        ## 0. Setup

        The offline path needs only `numpy`, `pandas`, `matplotlib`, and `scikit-learn`. API calls are opt-in, so the whole notebook runs without a key.

        In a fresh Colab runtime, uncomment and run:

        ```python
        %pip install -q "openai>=1.30" "pandas>=2.0" "scikit-learn>=1.3" "matplotlib>=3.7"
        ```
        '''),
        code(COMMON_SETUP),
        md('''
        ## 1. Load and understand Choice13K

        The bundled CSV is the full clean teaching subset (no feedback and no ambiguity) derived from the user-provided Choice13K files. Each row describes:

        - a lottery pair;
        - a no-feedback risky-choice condition with explicit probabilities;
        - `bRate`: the mean fraction of five repeated decisions allocated to Option B;
        - `n`: the number of participants contributing a five-choice proportion.

        We retain the continuous `bRate` for probabilistic evaluation and derive a majority label only when needed.
        '''),
        code(LOAD_DATA),
        code(r'''
        summary = pd.DataFrame({
            "rows": [len(data)],
            "unique_problems": [data["Problem"].nunique()],
            "mean_human_B_rate": [data["bRate"].mean()],
            "ambiguous_option_B": [data["Amb"].mean()],
            "feedback_condition": [data["Feedback"].mean()],
        })
        display(summary.round(3))

        ax = data["bRate"].hist(bins=20, figsize=(7, 3.2), color="#315b7d", edgecolor="white")
        ax.axvline(0.5, color="#995f16", linestyle="--", linewidth=2)
        ax.set(xlabel="Observed human B-choice rate", ylabel="Condition rows")
        plt.show()
        '''),
        code(r'''
        example = data.sample(1, random_state=SEED).iloc[0]
        print(format_trial(example) if "format_trial" in globals() else
              f"Option A: {example['lottery_A']}\nOption B: {example['lottery_B']}")
        print(f"\nHuman B-choice rate: {example['bRate']:.3f}")
        '''),
        md('''
        ### Why split by `Problem`?

        Some lottery pairs appear in more than one feedback/block condition. A row-level random split can therefore place nearly identical stimuli in train and test. We split by problem ID so the target lottery pair is genuinely held out.
        '''),
        code(r'''
        from sklearn.model_selection import GroupShuffleSplit

        # Ambiguous trials hide Option B probabilities from participants. Keep them for an extension,
        # but use non-ambiguous trials for the cleanest first comparison.
        analysis_data = data.loc[~data["Amb"].astype(bool)].reset_index(drop=True)

        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
        train_idx, test_idx = next(
            splitter.split(analysis_data, groups=analysis_data["Problem"])
        )
        train = analysis_data.iloc[train_idx].copy()
        test = analysis_data.iloc[test_idx].copy()

        assert set(train["Problem"]).isdisjoint(set(test["Problem"]))
        print(f"Train: {len(train):,} rows / {train['Problem'].nunique():,} problems")
        print(f"Test:  {len(test):,} rows / {test['Problem'].nunique():,} problems")
        '''),
        md('''
        ## 2. Behavioral baselines before an LLM

        An LLM result is hard to interpret without reference points. We compare:

        1. **Population rate:** always predict the training-set mean B rate.
        2. **EV Ridge:** learn a smooth B-rate prediction from expected-value difference.
        3. **Transparent Ridge:** learn from EV, risk, worst/best outcomes, and lottery complexity.

        Because the outcome is an observed aggregate rate rather than a single binary response, MAE and RMSE are primary. Majority accuracy and soft-label log loss are secondary views.
        '''),
        code(r'''
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge


        def soft_log_loss(observed_rate, predicted_rate):
            y = np.asarray(observed_rate, dtype=float)
            p = np.clip(np.asarray(predicted_rate, dtype=float), 1e-6, 1 - 1e-6)
            return np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))


        def metric_row(name, frame, predicted_rate):
            y_rate = frame["bRate"].to_numpy()
            p = np.clip(np.asarray(predicted_rate), 0, 1)
            return {
                "model": name,
                "mae": np.mean(np.abs(p - y_rate)),
                "rmse": np.sqrt(np.mean((p - y_rate) ** 2)),
                "soft_log_loss": soft_log_loss(y_rate, p),
                "majority_accuracy": np.mean((p >= 0.5) == (y_rate > 0.5)),
                "correlation_with_bRate": (
                    np.nan if np.std(p) < 1e-12 else np.corrcoef(p, y_rate)[0, 1]
                ),
            }


        population_p = np.repeat(train["bRate"].mean(), len(test))

        ev_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        ev_model.fit(train[["ev_diff"]], train["bRate"])
        ev_p = np.clip(ev_model.predict(test[["ev_diff"]]), 0, 1)

        rich_features = [
            "ev_A", "ev_B", "sd_A", "sd_B", "min_A", "min_B",
            "max_A", "max_B", "n_outcomes_A", "n_outcomes_B"
        ]
        rich_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        rich_model.fit(train[rich_features], train["bRate"])
        rich_p = np.clip(rich_model.predict(test[rich_features]), 0, 1)

        baseline_results = pd.DataFrame([
            metric_row("Population rate", test, population_p),
            metric_row("EV Ridge", test, ev_p),
            metric_row("Transparent Ridge", test, rich_p),
        ]).set_index("model")
        baseline_results.round(3)
        '''),
        code(PROMPT_FUNCTIONS),
        code(r'''
        target = test.sample(1, random_state=17).iloc[0]
        print("SYSTEM:\n" + zero_shot_messages(target)[0]["content"])
        print("\nUSER:\n" + zero_shot_messages(target)[1]["content"])
        print(f"\n[Hidden during prediction] Human B rate = {target['bRate']:.3f}")
        '''),
        md('''
        ## 3. One wrapper for DeepSeek, GLM, and a local server

        We use the OpenAI-compatible Python client for all backends:

        | Provider | Base URL | Teaching default | Best use here |
        |---|---|---|---|
        | GLM | `https://open.bigmodel.cn/api/paas/v4/` | `glm-4.7-flash` | ordinary generation / JSON |
        | DeepSeek | `https://api.deepseek.com` | `deepseek-v4-flash` | generation + token logprobs |
        | Ollama | `http://localhost:11434/v1` | `qwen3:0.6b` | optional local generation |

        API calls default to **off**. Put keys in environment variables or Colab Secrets; never paste keys into a shared notebook.
        '''),
        code(r'''
        PROVIDERS = {
            "glm": {
                "base_url": "https://open.bigmodel.cn/api/paas/v4/",
                "model": "glm-4.7-flash",
                "key_env": "ZAI_API_KEY",
                "supports_logprobs": False,
            },
            "deepseek": {
                "base_url": "https://api.deepseek.com",
                "model": "deepseek-v4-flash",
                "key_env": "DEEPSEEK_API_KEY",
                "supports_logprobs": True,
            },
            "ollama": {
                "base_url": "http://localhost:11434/v1",
                "model": "qwen3:0.6b",
                "key_env": None,
                "supports_logprobs": True,
            },
        }

        PROVIDER = "glm"       # change to "deepseek" for the logprob lab
        RUN_API = False         # opt in only after configuring a key
        N_API_TRIALS = 12       # keep the first classroom run small


        def read_secret(name):
            if not name:
                return "ollama"
            value = os.getenv(name)
            if value:
                return value
            try:
                from google.colab import userdata
                return userdata.get(name)
            except Exception:
                return None


        def make_client(provider=PROVIDER):
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError("Install the optional API dependency: pip install openai") from exc
            cfg = PROVIDERS[provider]
            key = read_secret(cfg["key_env"])
            if not key:
                raise RuntimeError(f"Set {cfg['key_env']} in your environment or Colab Secrets.")
            return OpenAI(api_key=key, base_url=cfg["base_url"])


        print("API calls enabled:", RUN_API)
        print("Selected provider:", PROVIDER, PROVIDERS[PROVIDER]["model"])
        '''),
        code(r'''
        def call_choice_json(messages, provider=PROVIDER):
            """Return the model's generated estimate of aggregate p_B.

            This number is generated text. It is not a next-token probability.
            """
            cfg = PROVIDERS[provider]
            client = make_client(provider)
            kwargs = {}
            if provider == "deepseek":
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            response = client.chat.completions.create(
                model=cfg["model"],
                messages=messages,
                temperature=0,
                max_tokens=80,
                response_format={"type": "json_object"},
                **kwargs,
            )
            obj = json.loads(response.choices[0].message.content)
            probability_b = float(obj["p_B"])
            if not 0 <= probability_b <= 1:
                raise ValueError(f"Invalid model output: {obj}")
            label = "B" if probability_b >= 0.5 else "A"
            return {"label": label, "probability_B": probability_b, "raw": obj}

        if RUN_API:
            print(call_choice_json(zero_shot_messages(target)))
        else:
            print("Skipped. Set RUN_API=True after configuring a key.")
        '''),
        md('''
        ## 4. Token logprobs → probability over valid labels

        A model usually returns **log probabilities**, not already normalized A/B choice probabilities.

        Suppose the first output position contains probability mass for many tokens:

        ```text
        " B"  logprob -0.51
        " A"  logprob -1.02
        "The" logprob -1.50
        "I"   logprob -2.10
        ...
        ```

        We exponentiate logprobs, select exact valid labels, and re-normalize only over A and B. This answers a conditional question:

        > If the next token must be A or B, how is probability divided between them?

        It is **not automatically equal** to the human choice rate.
        '''),
        code(r'''
        demo_top_logprobs = [
            {"token": " B", "logprob": -0.51},
            {"token": " A", "logprob": -1.02},
            {"token": "The", "logprob": -1.50},
            {"token": " I", "logprob": -2.10},
            {"token": "It", "logprob": -2.45},
        ]


        def renormalize_valid_labels(top_logprobs, valid_labels=("A", "B")):
            found = {}
            for item in top_logprobs:
                token = str(item["token"]).strip()
                if token in valid_labels and token not in found:
                    found[token] = float(item["logprob"])
            missing = [label for label in valid_labels if label not in found]
            if missing:
                raise ValueError(
                    f"Missing valid label(s) {missing} from top-k. Increase top_logprobs; "
                    "do not silently assign probability zero."
                )
            max_logprob = max(found.values())
            unnormalized = {
                label: math.exp(logprob - max_logprob)
                for label, logprob in found.items()
            }
            total = sum(unnormalized.values())
            return {label: value / total for label, value in unnormalized.items()}


        display(pd.DataFrame(demo_top_logprobs).assign(
            raw_probability=lambda x: np.exp(x["logprob"])
        ).round(3))
        print("Conditional distribution over valid labels:",
              renormalize_valid_labels(demo_top_logprobs))
        '''),
        code(r'''
        def call_choice_logprobs(messages, provider="deepseek", top_k=20):
            """Obtain first-position logprobs and re-normalize over A/B.

            If A or B is absent from top-k, return an explicit failure instead of zero.
            """
            cfg = PROVIDERS[provider]
            if not cfg["supports_logprobs"]:
                raise NotImplementedError(
                    f"Token logprobs are not documented for {provider}'s current chat API."
                )
            client = make_client(provider)
            response = client.chat.completions.create(
                model=cfg["model"],
                messages=messages,
                temperature=0,
                max_tokens=1,
                logprobs=True,
                top_logprobs=top_k,
                extra_body={"thinking": {"type": "disabled"}},
            )
            position = response.choices[0].logprobs.content[0]
            top = [
                {"token": item.token, "logprob": item.logprob}
                for item in position.top_logprobs
            ]
            conditional = renormalize_valid_labels(top)
            return {
                "generated_token": position.token,
                "top_logprobs": top,
                "probability_A_given_valid": conditional["A"],
                "probability_B_given_valid": conditional["B"],
            }


        if RUN_API and PROVIDER == "deepseek":
            logprob_result = call_choice_logprobs(label_messages(target))
            display(pd.DataFrame(logprob_result["top_logprobs"]).head())
            print({k: v for k, v in logprob_result.items() if k != "top_logprobs"})
        else:
            print("For the live logprob lab, select DeepSeek and enable API calls.")
        '''),
        md('''
        ## 5. In-context learning: what changes?

        The model weights stay fixed. We add demonstrations to the prompt.

        We compare:

        - **zero-shot:** target trial only;
        - **random few-shot:** randomly selected training problems;
        - **retrieved few-shot:** training problems close in engineered task features.

        The target problem and its human response remain held out in every condition.
        '''),
        code(r'''
        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors import NearestNeighbors

        RETRIEVAL_FEATURES = [
            "ev_diff", "risk_diff", "worst_diff", "best_diff",
            "n_outcomes_A", "n_outcomes_B", "Feedback", "Block"
        ]
        retrieval_scaler = StandardScaler().fit(train[RETRIEVAL_FEATURES])
        train_vectors = retrieval_scaler.transform(train[RETRIEVAL_FEATURES])
        neighbor_index = NearestNeighbors(metric="euclidean").fit(train_vectors)


        def random_demos(k=4, seed=SEED):
            return train.sample(k, random_state=seed)


        def retrieved_demos(target_row, k=4):
            target_vector = retrieval_scaler.transform(
                pd.DataFrame([target_row[RETRIEVAL_FEATURES]])
            )
            _, indices = neighbor_index.kneighbors(target_vector, n_neighbors=k)
            return train.iloc[indices[0]]


        def few_shot_messages(target_row, demonstrations):
            blocks = []
            for number, (_, demo) in enumerate(demonstrations.iterrows(), start=1):
                blocks.append(
                    f"Example {number}:\n{format_trial(demo, include_answer=True)}"
                )
            blocks.append(
                f'Target trial:\n{format_trial(target_row)}\n'
                'Predict aggregate behavior. Return JSON only: {"p_B": number between 0 and 1}.'
            )
            return [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "\n\n".join(blocks)},
            ]


        retrieved = retrieved_demos(target, k=4)
        print(few_shot_messages(target, retrieved)[1]["content"])
        '''),
        code(r'''
        prompt_conditions = {
            "zero_shot": zero_shot_messages(target),
            "random_4_shot": few_shot_messages(target, random_demos(4, seed=17)),
            "retrieved_4_shot": few_shot_messages(target, retrieved_demos(target, 4)),
        }
        pd.DataFrame({
            "condition": prompt_conditions.keys(),
            "characters_in_user_prompt": [len(x[-1]["content"]) for x in prompt_conditions.values()],
        })
        '''),
        md('''
        ### Small, cached-by-you evaluation

        The cell below deliberately limits calls and saves results locally. For a real study, increase the sample only after the prompt, parser, split, and cost estimate are fixed.
        '''),
        code(r'''
        def run_prompt_condition(frame, condition, provider=PROVIDER):
            rows = []
            for _, row in frame.iterrows():
                if condition == "zero_shot":
                    messages = zero_shot_messages(row)
                elif condition == "random_4_shot":
                    messages = few_shot_messages(row, random_demos(4, seed=int(row["row_id"])))
                elif condition == "retrieved_4_shot":
                    messages = few_shot_messages(row, retrieved_demos(row, 4))
                else:
                    raise ValueError(condition)

                result = call_choice_json(messages, provider=provider)
                p_b = result["probability_B"]
                label = result["label"]
                probability_type = "generated_aggregate_rate"

                rows.append({
                    "row_id": row["row_id"],
                    "condition": condition,
                    "provider": provider,
                    "prediction": label,
                    "p_B": p_b,
                    "probability_type": probability_type,
                    "human_bRate": row["bRate"],
                    "human_majority": row["majority_choice"],
                    "n": row["n"],
                })
            return pd.DataFrame(rows)


        if RUN_API:
            api_subset = test.sample(min(N_API_TRIALS, len(test)), random_state=SEED)
            cache_dir = Path("cache")
            cache_dir.mkdir(exist_ok=True)
            api_results = pd.concat([
                run_prompt_condition(api_subset, condition)
                for condition in ["zero_shot", "random_4_shot", "retrieved_4_shot"]
            ], ignore_index=True)
            cache_path = cache_dir / f"prediction_{PROVIDER}.csv"
            api_results.to_csv(cache_path, index=False)
            display(api_results.head())
            print("Saved:", cache_path)
        else:
            print("Dry run complete: prompts were built, but no external calls were made.")
        '''),
        code(r'''
        if RUN_API:
            comparison = []
            for condition, group in api_results.groupby("condition"):
                frame = group.rename(columns={"human_bRate": "bRate"}).copy()
                frame["n"] = group["n"]
                comparison.append(metric_row(condition, frame, group["p_B"]))
            display(pd.DataFrame(comparison).set_index("model").round(3))
        else:
            display(baseline_results.round(3))
            print("These are offline baselines, not cached LLM results.")
        '''),
        md('''
        ## 6. Robustness checks to assign in class

        Pick one manipulation at a time:

        1. swap the displayed A/B positions while preserving lotteries;
        2. shuffle demonstration labels across trials;
        3. use demonstrations from a different feedback condition;
        4. change example order;
        5. compare random versus retrieved demonstrations;
        6. repeat the split with another seed.

        Treat prompt variation as an experimental factor, not as an invisible tuning step.

        ## Take-home messages

        - **Prediction is graded:** accuracy, calibration, and correlation answer different questions.
        - **Token probability requires a contract:** top-k logprobs must contain both valid labels before re-normalization.
        - **ICL changes evidence, not weights.** A gain does not by itself identify what was learned.
        - **Choice13K here is population-level.** Do not call this participant personalization.
        - **Better prediction is useful evidence, not automatic evidence of a recovered cognitive mechanism.**
        '''),
    ]
    return notebook(cells, "01_prediction_from_zero_shot_to_icl.ipynb")


def build_notebook_2() -> nbf.NotebookNode:
    cells = [
        md('''
        # Representations, Explanations, and Executable Candidate Models

        **Core/advanced notebook · 40–55 minutes**

        Prediction asks whether a model can forecast behavior. This notebook asks three deeper—but distinct—questions:

        1. **Representation:** does a representation carry information useful for held-out behavior?
        2. **Explanation:** can a verbal account be translated into discriminative predictions?
        3. **Model discovery:** can candidate mechanisms be made executable, compared, and revised?

        We continue to use the bundled Choice13K teaching sample.
        '''),
        code(COMMON_SETUP),
        code(LOAD_DATA),
        code(PROMPT_FUNCTIONS),
        code(r'''
        from sklearn.model_selection import GroupShuffleSplit

        analysis_data = data.loc[~data["Amb"].astype(bool)].reset_index(drop=True)
        split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
        train_idx, test_idx = next(split.split(analysis_data, groups=analysis_data["Problem"]))
        train = analysis_data.iloc[train_idx].copy()
        test = analysis_data.iloc[test_idx].copy()
        assert set(train["Problem"]).isdisjoint(test["Problem"])

        train["trial_text"] = train.apply(format_trial, axis=1)
        test["trial_text"] = test.apply(format_trial, axis=1)
        print(len(train), len(test))
        '''),
        md('''
        ## 1. Start with a non-LLM text baseline

        Before interpreting an embedding space, ask whether a simple bag-of-words representation already solves the task. Here, TF–IDF is the baseline; GLM `embedding-3` is an optional API replacement.
        '''),
        code(r'''
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, log_loss

        tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=1200, min_df=2)
        X_train_tfidf = tfidf.fit_transform(train["trial_text"])
        X_test_tfidf = tfidf.transform(test["trial_text"])

        text_probe = LogisticRegression(max_iter=1000, random_state=SEED)
        text_probe.fit(X_train_tfidf, train["majority_B"], sample_weight=train["n"])
        tfidf_p = text_probe.predict_proba(X_test_tfidf)[:, 1]

        pd.DataFrame({
            "accuracy": [accuracy_score(test["majority_B"], tfidf_p >= 0.5)],
            "binary_log_loss": [log_loss(test["majority_B"], tfidf_p)],
            "corr_with_human_bRate": [np.corrcoef(tfidf_p, test["bRate"])[0, 1]],
        }).round(3)
        '''),
        md('''
        ## 2. Optional GLM embeddings

        `embedding-3` supports batched text embeddings. The API path is opt-in and cached. When disabled, subsequent cells use TF–IDF so the notebook remains runnable.

        An embedding can be scientifically useful without being a cognitive mechanism. We test it by held-out probing and controls—not by whether a 2-D plot “looks meaningful.”
        '''),
        code(r'''
        RUN_EMBEDDING_API = False
        EMBEDDING_MODEL = "embedding-3"
        EMBEDDING_DIMENSIONS = 256


        def read_zai_key():
            value = os.getenv("ZAI_API_KEY")
            if value:
                return value
            try:
                from google.colab import userdata
                return userdata.get("ZAI_API_KEY")
            except Exception:
                return None


        def glm_embeddings(texts, batch_size=64):
            from openai import OpenAI
            key = read_zai_key()
            if not key:
                raise RuntimeError("Set ZAI_API_KEY in the environment or Colab Secrets.")
            client = OpenAI(api_key=key, base_url="https://open.bigmodel.cn/api/paas/v4/")
            vectors = []
            texts = list(texts)
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                response = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch,
                    dimensions=EMBEDDING_DIMENSIONS,
                )
                vectors.extend(item.embedding for item in response.data)
            return np.asarray(vectors, dtype=np.float32)


        cache_dir = Path("cache")
        train_embedding_path = cache_dir / "glm_embedding_train.npy"
        test_embedding_path = cache_dir / "glm_embedding_test.npy"

        if RUN_EMBEDDING_API:
            cache_dir.mkdir(exist_ok=True)
            X_train_repr = glm_embeddings(train["trial_text"])
            X_test_repr = glm_embeddings(test["trial_text"])
            np.save(train_embedding_path, X_train_repr)
            np.save(test_embedding_path, X_test_repr)
            representation_name = "GLM embedding-3"
        elif train_embedding_path.exists() and test_embedding_path.exists():
            X_train_repr = np.load(train_embedding_path)
            X_test_repr = np.load(test_embedding_path)
            representation_name = "cached GLM embedding-3"
        else:
            X_train_repr = X_train_tfidf
            X_test_repr = X_test_tfidf
            representation_name = "TF-IDF baseline"

        print(representation_name, X_train_repr.shape, X_test_repr.shape)
        '''),
        code(r'''
        representation_probe = LogisticRegression(max_iter=1000, random_state=SEED)
        representation_probe.fit(
            X_train_repr, train["majority_B"], sample_weight=train["n"]
        )
        representation_p = representation_probe.predict_proba(X_test_repr)[:, 1]
        observed_accuracy = accuracy_score(test["majority_B"], representation_p >= 0.5)
        observed_correlation = np.corrcoef(representation_p, test["bRate"])[0, 1]

        # A small permutation control: refit the same probe after breaking
        # the mapping between representations and training labels.
        permutation_rows = []
        for permutation_seed in range(10):
            rng_control = np.random.default_rng(permutation_seed)
            shuffled_labels = rng_control.permutation(train["majority_B"].to_numpy())
            shuffled_probe = LogisticRegression(max_iter=1000, random_state=SEED)
            shuffled_probe.fit(X_train_repr, shuffled_labels)
            shuffled_p = shuffled_probe.predict_proba(X_test_repr)[:, 1]
            permutation_rows.append({
                "accuracy": accuracy_score(test["majority_B"], shuffled_p >= 0.5),
                "corr_with_bRate": np.corrcoef(shuffled_p, test["bRate"])[0, 1],
            })
        permutation_results = pd.DataFrame(permutation_rows)

        probe_results = pd.DataFrame([
            {
                "representation": representation_name,
                "accuracy": observed_accuracy,
                "corr_with_bRate": observed_correlation,
            },
            {
                "representation": "shuffled-label control (mean of 10)",
                "accuracy": permutation_results["accuracy"].mean(),
                "corr_with_bRate": permutation_results["corr_with_bRate"].mean(),
            },
        ]).set_index("representation")
        display(probe_results.round(3))
        print("Permutation accuracy range:",
              tuple(permutation_results["accuracy"].round(3).agg(["min", "max"])))
        '''),
        code(r'''
        from sklearn.decomposition import TruncatedSVD

        plot_n = min(400, len(test))
        plot_rows = test.sample(plot_n, random_state=SEED)
        plot_positions = test.index.get_indexer(plot_rows.index)
        plot_vectors = X_test_repr[plot_positions]
        coordinates = TruncatedSVD(n_components=2, random_state=SEED).fit_transform(plot_vectors)

        plt.figure(figsize=(7, 4.5))
        scatter = plt.scatter(
            coordinates[:, 0], coordinates[:, 1],
            c=plot_rows["bRate"], cmap="coolwarm", vmin=0, vmax=1,
            alpha=0.75, s=28,
        )
        plt.colorbar(scatter, label="Human B-choice rate")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.title(f"{representation_name}: visualization, not validation")
        plt.show()
        '''),
        md('''
        ## 3. Explanations: separate observation from hypothesis

        A useful structured explanation should distinguish:

        - **observation:** what is directly present in the trial or data;
        - **hypothesis:** a candidate latent strategy;
        - **prediction:** what the hypothesis implies on a new or counterfactual trial;
        - **falsifier:** evidence that would count against it.

        The prompt below asks an LLM for candidate explanations, not for privileged access to the participant’s true mental process.
        '''),
        code(r'''
        disagreement = test.loc[
            ((test["ev_diff"] > 0) & (test["bRate"] < 0.5))
            | ((test["ev_diff"] < 0) & (test["bRate"] > 0.5))
        ].copy()
        explanation_trial = disagreement.iloc[0] if len(disagreement) else test.iloc[0]
        print(format_trial(explanation_trial))
        print(f"Observed majority: {explanation_trial['majority_choice']} "
              f"(B rate={explanation_trial['bRate']:.2f})")
        '''),
        code(r'''
        EXPLANATION_SCHEMA = {
            "observation": "directly observed trial or behavioral fact",
            "candidate_mechanism": "short hypothesis name",
            "predicted_choice": "A or B",
            "counterfactual_prediction": "what should change under one manipulation",
            "falsifying_evidence": "result that would count against the mechanism",
        }


        def explanation_messages(row):
            return [
                {
                    "role": "system",
                    "content": (
                        "You are proposing a testable behavioral hypothesis. "
                        "Do not claim access to the participant's true mental state."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        format_trial(row, include_answer=True)
                        + "\nReturn JSON only with these fields:\n"
                        + json.dumps(EXPLANATION_SCHEMA, indent=2)
                    ),
                },
            ]


        print(explanation_messages(explanation_trial)[1]["content"])
        '''),
        code(r'''
        RUN_EXPLANATION_API = False
        EXPLANATION_PROVIDER = "glm"


        def call_structured_explanation(row, provider=EXPLANATION_PROVIDER):
            from openai import OpenAI
            if provider == "glm":
                key = read_zai_key()
                base_url = "https://open.bigmodel.cn/api/paas/v4/"
                model = "glm-4.7-flash"
                extra = {}
            else:
                key = os.getenv("DEEPSEEK_API_KEY")
                base_url = "https://api.deepseek.com"
                model = "deepseek-v4-flash"
                extra = {"extra_body": {"thinking": {"type": "disabled"}}}
            if not key:
                raise RuntimeError("Configure the provider API key first.")
            client = OpenAI(api_key=key, base_url=base_url)
            response = client.chat.completions.create(
                model=model,
                messages=explanation_messages(row),
                temperature=0,
                max_tokens=400,
                response_format={"type": "json_object"},
                **extra,
            )
            result = json.loads(response.choices[0].message.content)
            missing = set(EXPLANATION_SCHEMA) - set(result)
            if missing:
                raise ValueError(f"Missing fields: {sorted(missing)}")
            return result


        if RUN_EXPLANATION_API:
            display(call_structured_explanation(explanation_trial))
        else:
            print("Skipped external explanation generation. The prompt and validator are ready.")
        '''),
        md('''
        ## 4. Make candidate mechanisms executable

        Instead of rating explanations by fluency, implement simple candidate rules and ask where they disagree.

        These are intentionally simple *candidate models*, not claims about the true process.
        '''),
        code(r'''
        def stable_sigmoid(score, scale):
            scale = max(float(scale), 1e-6)
            return 1 / (1 + np.exp(-np.clip(score / scale, -30, 30)))


        scales = {
            "ev": max(train["ev_diff"].abs().median(), 1.0),
            "worst": max(train["worst_diff"].abs().median(), 1.0),
            "best": max(train["best_diff"].abs().median(), 1.0),
            "risk": max(train["risk_diff"].abs().median(), 1.0),
        }

        candidate_predictions = {
            "Expected value": stable_sigmoid(test["ev_diff"], scales["ev"]),
            "Maximin / worst outcome": stable_sigmoid(test["worst_diff"], scales["worst"]),
            "Best-outcome focus": stable_sigmoid(test["best_diff"], scales["best"]),
            "Risk seeking": stable_sigmoid(test["risk_diff"], scales["risk"]),
        }


        def soft_loss(y_rate, p):
            p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
            y_rate = np.asarray(y_rate)
            return np.mean(-(y_rate * np.log(p) + (1 - y_rate) * np.log(1 - p)))


        candidate_table = pd.DataFrame([
            {
                "candidate": name,
                "soft_log_loss": soft_loss(test["bRate"], prediction),
                "majority_accuracy": np.mean((prediction >= 0.5) == test["majority_B"]),
                "corr_with_bRate": np.corrcoef(prediction, test["bRate"])[0, 1],
            }
            for name, prediction in candidate_predictions.items()
        ]).sort_values("soft_log_loss")
        candidate_table.round(3)
        '''),
        code(r'''
        prediction_frame = pd.DataFrame(candidate_predictions, index=test.index)
        prediction_frame["spread"] = prediction_frame.max(axis=1) - prediction_frame.min(axis=1)
        diagnostic_indices = prediction_frame.nlargest(5, "spread").index

        for idx in diagnostic_indices[:3]:
            row = test.loc[idx]
            print("-" * 80)
            print(format_trial(row))
            print(f"Human B rate: {row['bRate']:.2f}")
            display(prediction_frame.loc[[idx]].drop(columns="spread").round(2))
        '''),
        md('''
        ## 5. A minimal closed-loop model-discovery scaffold

        A safe first implementation keeps model execution outside the LLM:

        ```text
        candidate feature sets
            → fit each candidate on train
            → evaluate on held-out problems
            → inspect residuals and disagreements
            → ask for a revised candidate specification
            → repeat
        ```

        The LLM may propose a **restricted JSON specification**. Trusted code then maps approved feature names to an estimator. We do not execute arbitrary generated Python.
        '''),
        code(r'''
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        candidate_specs = [
            {"name": "EV only", "features": ["ev_diff"]},
            {"name": "EV + risk", "features": ["ev_diff", "risk_diff"]},
            {"name": "EV + outcomes", "features": ["ev_diff", "worst_diff", "best_diff"]},
            {
                "name": "Expanded",
                "features": [
                    "ev_diff", "risk_diff", "worst_diff", "best_diff",
                    "n_outcomes_A", "n_outcomes_B", "Block", "Feedback"
                ],
            },
        ]


        def fit_and_score_spec(spec):
            features = spec["features"]
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, random_state=SEED),
            )
            model.fit(train[features], train["majority_B"],
                      logisticregression__sample_weight=train["n"])
            p = model.predict_proba(test[features])[:, 1]
            return {
                "candidate": spec["name"],
                "n_features": len(features),
                "soft_log_loss": soft_loss(test["bRate"], p),
                "majority_accuracy": np.mean((p >= 0.5) == test["majority_B"]),
                "model": model,
                "p_B": p,
            }


        discovery_runs = [fit_and_score_spec(spec) for spec in candidate_specs]
        discovery_table = pd.DataFrame([
            {k: v for k, v in run.items() if k not in {"model", "p_B"}}
            for run in discovery_runs
        ]).sort_values("soft_log_loss")
        discovery_table.round(3)
        '''),
        md('''
        ## Scientific stopping rule

        A predictive winner is not automatically a recovered mechanism. Before making a stronger claim, seek:

        - generalization to new task environments;
        - diagnostic trials where candidates make different predictions;
        - process data such as language, eye movements, RT, or confidence;
        - interventions that change the hypothesized variable;
        - explicit checks that the candidate space contains plausible alternatives.

        ## Take-home messages

        1. A 2-D embedding plot is an illustration; held-out probing and controls are evidence.
        2. A fluent explanation becomes scientific only when it yields discriminative predictions.
        3. LLMs can accelerate candidate generation, but trusted evaluation code should remain in control.
        4. Predictive model discovery, family recovery, and primitive/mechanistic recovery are different success criteria.
        '''),
    ]
    return notebook(cells, "legacy_representation_explanation_discovery.ipynb")


def build_notebook_3() -> nbf.NotebookNode:
    cells = [
        md('''
        # Optional Local Models, Hidden States, and Adaptation

        **Optional extension · 25–40 minutes**

        This notebook separates four engineering choices:

        - **Transformers:** best for teaching logits and hidden states;
        - **Ollama:** easiest lightweight local serving path;
        - **Unsloth:** optional Colab GPU route for LoRA/SFT;
        - **vLLM:** instructor/server route for high-throughput GPU inference.

        All model downloads and training cells default to off. The notebook runs end-to-end on an ordinary laptop without downloading a model.
        '''),
        code(COMMON_SETUP),
        code(LOAD_DATA),
        code(PROMPT_FUNCTIONS),
        md('''
        ## 1. Direct logits with Transformers

        Recommended teaching model: `Qwen/Qwen3-0.6B`.

        - small enough for a short Colab demonstration;
        - CPU is possible but slower;
        - use a pre-downloaded local directory when classroom network access is uncertain.

        For a fresh environment:

        ```python
        %pip install -q "transformers>=4.51" accelerate torch
        ```
        '''),
        code(r'''
        RUN_TRANSFORMERS = False
        MODEL_ID = "Qwen/Qwen3-0.6B"
        LOCAL_MODEL_DIR = None   # e.g., "/content/models/Qwen3-0.6B"
        MODEL_SOURCE = LOCAL_MODEL_DIR or MODEL_ID

        if RUN_TRANSFORMERS:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_SOURCE,
                torch_dtype="auto",
                device_map="auto",
            )
            model.eval()
            print("Loaded", MODEL_SOURCE)
        else:
            print("Skipped model download. Set RUN_TRANSFORMERS=True to run the local lab.")
        '''),
        md('''
        ### Candidate-sequence scoring

        Top-k output is convenient, but a valid label may be absent or may tokenize into multiple pieces. The function below scores every token in candidate completions `A` and `B`, then normalizes their sequence scores.
        '''),
        code(r'''
        local_trial = data.loc[~data["Amb"].astype(bool)].sample(1, random_state=SEED).iloc[0]
        plain_prompt = (
            LABEL_SYSTEM_PROMPT + "\n\n" + format_trial(local_trial)
            + "\nRespond with exactly one letter.\nAnswer:"
        )
        print(plain_prompt)
        '''),
        code(r'''
        def candidate_sequence_logprob(model, tokenizer, prompt, candidate):
            """Sum conditional log probabilities for every token in a candidate string."""
            import torch

            prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            candidate_ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
            input_ids = torch.tensor(
                [prompt_ids + candidate_ids], device=model.device, dtype=torch.long
            )
            with torch.no_grad():
                logits = model(input_ids=input_ids).logits[0]
            log_probs = torch.log_softmax(logits, dim=-1)

            start = len(prompt_ids) - 1
            score = 0.0
            for offset, token_id in enumerate(candidate_ids):
                score += float(log_probs[start + offset, token_id].cpu())
            return score, candidate_ids


        def score_a_vs_b(model, tokenizer, prompt):
            candidates = ["A", "B"]
            scored = {
                label: candidate_sequence_logprob(model, tokenizer, prompt, label)
                for label in candidates
            }
            log_scores = np.array([scored[label][0] for label in candidates])
            probabilities = np.exp(log_scores - log_scores.max())
            probabilities /= probabilities.sum()
            return pd.DataFrame({
                "label": candidates,
                "token_ids": [scored[label][1] for label in candidates],
                "sequence_logprob": log_scores,
                "renormalized_probability": probabilities,
            })


        if RUN_TRANSFORMERS:
            display(score_a_vs_b(model, tokenizer, plain_prompt))
        else:
            print("Function defined; execution skipped.")
        '''),
        md('''
        ### Inspect the actual top tokens

        Raw logits become log probabilities after `log_softmax`. Looking at the top tokens before restricting to A/B makes the re-normalization step explicit.
        '''),
        code(r'''
        if RUN_TRANSFORMERS:
            import torch
            encoded = tokenizer(plain_prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model(**encoded)
            final_logits = output.logits[0, -1]
            final_logprobs = torch.log_softmax(final_logits, dim=-1)
            values, token_ids = torch.topk(final_logprobs, k=8)
            top_tokens = pd.DataFrame({
                "token": [tokenizer.decode([int(i)]) for i in token_ids],
                "token_id": token_ids.cpu().numpy(),
                "logprob": values.cpu().numpy(),
                "probability": values.exp().cpu().numpy(),
            })
            display(top_tokens)
        else:
            print("Skipped forward pass.")
        '''),
        md('''
        ## 2. Hidden states: a representation, not a mechanism

        The next cell extracts the final-token hidden state from selected layers. A downstream probe can test whether these vectors carry EV, risk, or choice information. Similar geometry across model and brain data is evidence of correspondence—not proof of an identical computation.
        '''),
        code(r'''
        def extract_layer_vectors(model, tokenizer, prompts, layers=(-1, -7, -14)):
            import torch
            vectors = {layer: [] for layer in layers}
            for prompt in prompts:
                encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    output = model(**encoded, output_hidden_states=True)
                for layer in layers:
                    vector = output.hidden_states[layer][0, -1].float().cpu().numpy()
                    vectors[layer].append(vector)
            return {layer: np.vstack(items) for layer, items in vectors.items()}


        if RUN_TRANSFORMERS:
            representation_rows = data.loc[~data["Amb"].astype(bool)].sample(12, random_state=SEED)
            prompts = [SYSTEM_PROMPT + "\n" + format_trial(row) for _, row in representation_rows.iterrows()]
            layer_vectors = extract_layer_vectors(model, tokenizer, prompts)
            print({layer: matrix.shape for layer, matrix in layer_vectors.items()})
        else:
            print("Hidden-state extraction defined; execution skipped.")
        '''),
        md('''
        ## 3. Ollama: the easiest local serving path

        Install Ollama separately, then pull a small quantized model before class:

        ```bash
        ollama pull qwen3:0.6b
        ```

        Ollama exposes an OpenAI-compatible endpoint, so only the base URL and model change. It is excellent for local privacy and deployment demos, but it does not expose layer-wise hidden states.
        '''),
        code(r'''
        RUN_OLLAMA = False

        if RUN_OLLAMA:
            from openai import OpenAI
            ollama_client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
            response = ollama_client.chat.completions.create(
                model="qwen3:0.6b",
                messages=zero_shot_messages(local_trial),
                temperature=0,
                max_tokens=8,
            )
            print(response.choices[0].message.content)
        else:
            print("Skipped. Start Ollama locally and set RUN_OLLAMA=True.")
        '''),
        md('''
        ## 4. Prepare SFT data before training anything

        Supervised fine-tuning learns to imitate target outputs. Here the target is the *aggregate B-choice rate*, so improved performance would demonstrate behavioral imitation—not mechanistic recovery.

        Always split by problem ID before constructing training examples.
        '''),
        code(r'''
        from sklearn.model_selection import GroupShuffleSplit

        sft_source = data.loc[~data["Amb"].astype(bool)].copy()
        split = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
        train_idx, validation_idx = next(
            split.split(sft_source, groups=sft_source["Problem"])
        )
        sft_train = sft_source.iloc[train_idx]
        sft_validation = sft_source.iloc[validation_idx]
        assert set(sft_train["Problem"]).isdisjoint(sft_validation["Problem"])


        def to_chat_example(row):
            return {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": format_trial(row) + "\nAnswer:"},
                    {"role": "assistant", "content": json.dumps({"p_B": round(float(row["bRate"]), 3)})},
                ],
                "metadata": {
                    "problem_id": int(row["Problem"]),
                    "human_bRate": float(row["bRate"]),
                    "n": int(row["n"]),
                },
            }


        sft_examples = [to_chat_example(row) for _, row in sft_train.head(5).iterrows()]
        print(json.dumps(sft_examples[0], indent=2))
        '''),
        md('''
        ### Optional Unsloth/LoRA extension

        Use this only with a Colab GPU and a tested environment. The conceptual steps are:

        ```python
        # Pseudocode: pin versions after a classroom smoke test.
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="Qwen/Qwen3-0.6B",
            max_seq_length=1024,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(model, r=16)
        # Convert sft_train to a Dataset, format messages with the chat template,
        # train a small LoRA adapter, and evaluate only on held-out Problem IDs.
        ```

        Do not spend core tutorial time debugging CUDA, quantization, or package versions. Provide an instructor-tested Colab if you decide to run this live.
        '''),
        md('''
        ## 5. Where RL and vLLM belong

        **RL / preference optimization** is optional because the reward definition is the scientific commitment. Rewarding agreement with aggregate choice rates may improve imitation while reinforcing a shortcut. Show the reward and its failure modes before showing an optimizer.

        **vLLM** becomes useful when one GPU server must support many concurrent students or a large batch. It is not the simplest per-student laptop setup, and native Windows is not its main deployment path.

        ## Recommended teaching stack

        | Need | Default |
        |---|---|
        | Cloud generation / JSON in China | GLM via OpenAI-compatible client |
        | Token logprobs | DeepSeek non-thinking mode |
        | Local generation on a laptop | Ollama |
        | Raw logits and hidden states | Transformers |
        | Optional Colab LoRA/SFT | Unsloth |
        | Shared high-throughput GPU service | vLLM |

        The simplest path is intentionally not one package for everything: **Transformers teaches research signals; Ollama teaches deployment.**
        '''),
    ]
    return notebook(cells, "04_optional_local_models_and_adaptation.ipynb")


README = """Legacy builder note: the curated notebook index now lives in notebooks/README.md."""


def main():
    """Build only the prediction and optional extension notebooks.

    Notebook 2 (representation) and Notebook 3 (explanation to discovery) have
    dedicated builders because they depend on cached analysis artifacts.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    make_teaching_sample()
    notebooks = {
        "01_prediction_from_zero_shot_to_icl.ipynb": build_participant_notebook(),
        "04_optional_local_models_and_adaptation.ipynb": build_notebook_3(),
    }
    for filename, nb in notebooks.items():
        nbf.write(nb, OUT / filename)
    print(f"Wrote {len(notebooks)} notebooks to {OUT}; preserved the curated README.")


if __name__ == "__main__":
    main()
