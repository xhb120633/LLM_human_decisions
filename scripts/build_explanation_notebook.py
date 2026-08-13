from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


PROJECT_ROOT = Path(r"C:\Users\Hanbo\Documents\GitHub\LLM_human_decisions")
OUT = PROJECT_ROOT / "notebooks" / "03_explanation_annotation_to_model_discovery.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


def build_notebook():
    cells = [
        md(
            r'''
            # From explanations to executable model discovery

            **Core hands-on notebook · 35–45 minutes**

            A fluent explanation is not privileged access to a decision process. But it can become a useful **process-data channel** when we:

            1. annotate what information and operations appear in the text;
            2. test whether those annotations are reliable and behaviorally informative;
            3. translate supported annotations into restricted candidate models;
            4. compare those models on held-out decision problems.

            We reuse the synthetic GPT-4 choice-and-reasoning records from Notebook 2. The final A/B conclusion was masked before Qwen representation analysis. Here we also truncate each trace before its first explicit preference claim.

            > Annotation can constrain one model-discovery route; discovery does not require annotation, and annotation does not reveal the true mechanism by itself.
            '''
        ),
        md(
            r'''
            ## 0. Setup

            The default path is offline and lightweight. It reads the cached sentence-level table produced for Notebook 2. **All bundled results use the transparent lexical baseline.** Optional GLM or DeepSeek annotation is disabled unless you explicitly turn it on; DeepSeek V4 was not used to generate the reported numbers.
            '''
        ),
        code(
            r'''
            from pathlib import Path
            import json
            import os
            import re
            import sys

            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt

            from sklearn.compose import ColumnTransformer
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import balanced_accuracy_score, log_loss
            from sklearn.model_selection import GroupKFold, cross_val_predict
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            SEED = 3000
            pd.set_option("display.max_colwidth", 120)
            plt.style.use("seaborn-v0_8-whitegrid")
            '''
        ),
        code(
            r'''
            def find_sentence_table():
                relative = Path(
                    "results/representation/text2decision_multiscale_log_trajectories/"
                    "sentence_decision_states.csv"
                )
                candidates = [
                    relative,
                    Path("notebooks") / relative,
                    Path("/content/LLM_human_decisions/notebooks") / relative,
                ]
                for path in candidates:
                    if path.exists():
                        return path
                raise FileNotFoundError(
                    "Could not find sentence_decision_states.csv. Keep the notebooks/results folder "
                    "with the tutorial materials."
                )


            SENTENCE_PATH = find_sentence_table()
            sentences = pd.read_csv(SENTENCE_PATH)
            sentences["sentence_text"] = sentences["sentence_text"].fillna("")
            sentences["is_masked_conclusion"] = (
                sentences["is_masked_conclusion"].astype(str).str.lower().eq("true")
            )

            print(f"Sentence rows: {len(sentences):,}")
            print(f"Trials: {sentences['trial_row'].nunique():,}")
            print(f"Questions: {sentences['question_id'].nunique()}")
            print(f"Personas: {sentences['persona'].nunique()}")
            '''
        ),
        md(
            r'''
            ## 1. What exactly are we annotating?

            The unit is one sentence, but the trace remains ordered. We separate three axes:

            - **attended information:** expected value, probabilities, downside, upside, certainty, risk;
            - **operation:** describe, compute, compare, or consider a counterfactual;
            - **evidential role:** supports A, supports B, neither, or an explicit preference claim.

            The final axis is essential for leakage control: explicit preference statements are excluded before downstream evaluation.
            '''
        ),
        code(
            r'''
            OPTION = r"option\s+[ab]"
            DECISIVE = (
                r"(?:choose|chose|select|pick|lean(?:ing)?|inclined|opt(?:ing)?|"
                r"go with|would take|will take|my choice|my decision|prefer)"
            )
            EXPLICIT_PATTERN = re.compile(
                rf"(?i)(?:{OPTION}.{{0,120}}\b{DECISIVE}\b|"
                rf"\b{DECISIVE}\b.{{0,120}}{OPTION}|"
                rf"\b{OPTION}\b.{{0,80}}(?:appealing|attractive).{{0,40}}(?:to me|for me))"
            )


            def is_explicit_claim(text):
                return "Option X" in text or bool(EXPLICIT_PATTERN.search(str(text)))


            def retain_preference_free_prefix(frame):
                frame = frame.sort_values("sentence_index").copy()
                flags = frame["sentence_text"].map(is_explicit_claim).to_numpy()
                cutoff = int(np.flatnonzero(flags)[0]) if flags.any() else len(frame)
                return frame.iloc[:cutoff]


            retained = pd.concat(
                [
                    retain_preference_free_prefix(frame)
                    for _, frame in sentences.groupby("trial_row", sort=False)
                ],
                ignore_index=True,
            )

            audit = pd.DataFrame({
                "all_sentences": sentences.groupby("trial_row").size(),
                "retained_before_claim": retained.groupby("trial_row").size(),
            }).fillna(0)
            audit["removed"] = audit["all_sentences"] - audit["retained_before_claim"]
            display(audit.describe().round(2))
            '''
        ),
        code(
            r'''
            example_trial = int(
                retained.loc[
                    (retained["question_id"] == "Q01")
                    & (retained["persona"] == "Rational Decision-Maker"),
                    "trial_row",
                ].iloc[0]
            )
            example = retained.loc[retained["trial_row"] == example_trial].sort_values("sentence_index")
            print(example["question"].iloc[0])
            display(example[["sentence_index", "sentence_text"]])
            '''
        ),
        md(
            r'''
            ## 2. Begin with a transparent annotation baseline

            Before asking a large language model to code thousands of sentences, build a weak, auditable baseline. The lexicon below will miss paraphrases and context, but every assigned label can be traced to an explicit rule.

            This baseline teaches an important principle: **annotation quality is an empirical question, not a property of the annotator's fluency.**
            '''
        ),
        code(
            r'''
            CODEBOOK = {
                "expected_value": r"expected value|expected payoff|on average|average payoff",
                "probability": r"probab|chance|likely|unlikely|odds|likelihood",
                "probability_weighting": r"small chance|tiny chance|rare|0\.1\s*%|overweight|underweight",
                "downside": r"worst|lose|loss|nothing|zero|downside|negative outcome",
                "upside": r"best|highest|maximum|upside|win|large gain|big payoff",
                "certainty": r"sure|certain|certainty|guaranteed|safe|security",
                "risk": r"risk|risky|gamble|variance|uncertain|volatil",
            }

            OPERATION_PATTERNS = {
                "compute": r"calculate|compute|multiply|sum|equals|expected value is",
                "compare": r"compare|higher|lower|better|worse|more than|less than|versus|vs\.?",
                "counterfactual": r"\bif\b|would|could|otherwise|scenario",
            }


            def annotate_sentence(text):
                text = str(text)
                attended = [
                    label for label, pattern in CODEBOOK.items()
                    if re.search(pattern, text, flags=re.IGNORECASE)
                ]
                operations = [
                    label for label, pattern in OPERATION_PATTERNS.items()
                    if re.search(pattern, text, flags=re.IGNORECASE)
                ] or ["describe"]
                return {
                    "attended_information": attended,
                    "operations": operations,
                    "explicit_preference": is_explicit_claim(text),
                }


            annotated_example = example[["sentence_index", "sentence_text"]].copy()
            annotated_example["annotation"] = annotated_example["sentence_text"].map(annotate_sentence)
            display(annotated_example)
            '''
        ),
        md(
            r'''
            ### Optional large-language-model annotation

            A language model can replace or supplement the lexical baseline, but constrain its output to the same codebook. Never ask it simply to reveal the "true strategy." This optional comparison is not used for the bundled results.
            '''
        ),
        code(
            r'''
            RUN_ANNOTATION_API = False
            ANNOTATION_PROVIDER = "glm"  # or "deepseek"

            ANNOTATION_SCHEMA = {
                "attended_information": list(CODEBOOK),
                "operations": ["describe", "compute", "compare", "counterfactual"],
                "supports": ["A", "B", "neither"],
                "epistemic_role": ["observation", "inference", "hypothesis", "conclusion"],
            }


            def annotation_messages(text):
                return [
                    {
                        "role": "system",
                        "content": (
                            "Annotate the decision-relevant content of one reasoning sentence. "
                            "Use only the supplied labels. Do not infer a hidden true mechanism."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Sentence:\n{text}\n\nAllowed schema:\n"
                            + json.dumps(ANNOTATION_SCHEMA, indent=2)
                            + "\nReturn JSON only."
                        ),
                    },
                ]


            def call_annotation_api(text, provider=ANNOTATION_PROVIDER):
                from openai import OpenAI

                if provider == "glm":
                    key = os.getenv("ZAI_API_KEY")
                    base_url = "https://open.bigmodel.cn/api/paas/v4/"
                    model = "glm-4.7-flash"
                    extra = {}
                else:
                    key = os.getenv("DEEPSEEK_API_KEY")
                    base_url = "https://api.deepseek.com"
                    model = "deepseek-v4-flash"
                    extra = {"extra_body": {"thinking": {"type": "disabled"}}}
                if not key:
                    raise RuntimeError("Configure the selected provider key first.")
                client = OpenAI(api_key=key, base_url=base_url)
                response = client.chat.completions.create(
                    model=model,
                    messages=annotation_messages(text),
                    temperature=0,
                    max_tokens=220,
                    response_format={"type": "json_object"},
                    **extra,
                )
                return json.loads(response.choices[0].message.content)


            if RUN_ANNOTATION_API:
                display(call_annotation_api(example.iloc[0]["sentence_text"]))
            else:
                print("API annotation is off. The transparent lexical baseline remains active.")
            '''
        ),
        md(
            r'''
            ## 3. Does the annotation recover the controlled manipulation?

            These records are synthetic, so the prompted persona is known. We do **not** treat the persona as the annotation target. Instead, it provides a construct-validity check:

            > Do preference-free annotation profiles differ across the five controlled reasoning conditions, and do those differences generalize to held-out questions?

            Success would support sensitivity to the manipulation. It would not prove that the same labels describe human cognition.
            '''
        ),
        code(
            r'''
            annotation_rows = []
            for row in retained.itertuples():
                result = annotate_sentence(row.sentence_text)
                item = {
                    "trial_row": row.trial_row,
                    "question_id": row.question_id,
                    "persona": row.persona,
                    "choice": int(row.choice),
                }
                for label in CODEBOOK:
                    item[label] = int(label in result["attended_information"])
                annotation_rows.append(item)

            sentence_annotations = pd.DataFrame(annotation_rows)
            annotation_profiles = (
                sentence_annotations.groupby(["trial_row", "question_id", "persona", "choice"])[list(CODEBOOK)]
                .mean()
                .reset_index()
            )

            persona_profile = annotation_profiles.groupby("persona")[list(CODEBOOK)].mean()
            COLUMN_LABELS = {
                "expected_value": "Expected\nvalue",
                "probability": "Probability",
                "probability_weighting": "Probability\nweighting",
                "downside": "Downside",
                "upside": "Upside",
                "certainty": "Certainty",
                "risk": "Risk",
            }
            display(persona_profile.rename(columns=COLUMN_LABELS).round(3))

            fig, ax = plt.subplots(figsize=(11.8, 4.8), layout="constrained")
            image = ax.imshow(persona_profile.to_numpy(), aspect="auto", cmap="Blues")
            ax.set_xticks(np.arange(len(CODEBOOK)))
            ax.set_xticklabels([COLUMN_LABELS[name] for name in CODEBOOK], ha="center")
            ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, pad=8)
            ax.set_yticks(np.arange(len(persona_profile)))
            ax.set_yticklabels(persona_profile.index)
            ax.set_ylabel("Controlled persona")
            ax.set_title("Preference-free annotation profiles by controlled persona", loc="left", weight="bold")
            for row_index in range(persona_profile.shape[0]):
                for column_index in range(persona_profile.shape[1]):
                    value = persona_profile.iloc[row_index, column_index]
                    ax.text(
                        column_index,
                        row_index,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color="white" if value > 0.42 else "#1f2937",
                        fontsize=9,
                    )
            fig.colorbar(image, ax=ax, label="Fraction of retained sentences", pad=0.02)
            plt.show()
            '''
        ),
        code(
            r'''
            X_annotation = annotation_profiles[list(CODEBOOK)]
            y_persona = annotation_profiles["persona"]
            groups = annotation_profiles["question_id"]

            persona_model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
            )
            persona_prediction = cross_val_predict(
                persona_model,
                X_annotation,
                y_persona,
                groups=groups,
                cv=GroupKFold(n_splits=5),
            )
            persona_balanced_accuracy = balanced_accuracy_score(y_persona, persona_prediction)
            print(f"Held-question persona balanced accuracy: {persona_balanced_accuracy:.3f}")
            print(f"Five-class chance reference: {1 / y_persona.nunique():.3f}")
            '''
        ),
        md(
            r'''
            ## 4. Does the annotation add behavioral information?

            Now test a downstream claim. We compare:

            1. option features only;
            2. preference-free annotation profiles only;
            3. option features plus annotation profiles.

            Evaluation holds out entire questions. This asks whether the coded process report carries information about choice beyond the displayed lotteries.
            '''
        ),
        code(
            r'''
            LOTTERY_PATTERN = re.compile(
                r"(-?\d+(?:\.\d+)?) dollars with (\d+(?:\.\d+)?) % chance"
            )


            def parse_option(question, label):
                segment = question.split(f"Option {label}:", 1)[1]
                if label == "A":
                    segment = segment.split("Option B:", 1)[0]
                pairs = [(float(value), float(probability) / 100) for value, probability in LOTTERY_PATTERN.findall(segment)]
                probabilities = np.asarray([p for _, p in pairs], dtype=float)
                outcomes = np.asarray([v for v, _ in pairs], dtype=float)
                probabilities = probabilities / probabilities.sum()
                ev = float(np.sum(probabilities * outcomes))
                sd = float(np.sqrt(np.sum(probabilities * (outcomes - ev) ** 2)))
                best_index = int(np.argmax(outcomes))
                return {
                    "ev": ev,
                    "sd": sd,
                    "worst": float(outcomes.min()),
                    "best": float(outcomes.max()),
                    "p_best": float(probabilities[best_index]),
                    "p_zero": float(probabilities[np.isclose(outcomes, 0)].sum()),
                    "certain": float(len(outcomes) == 1 or np.isclose(probabilities.max(), 1)),
                }


            questions = sentences[["question_id", "question"]].drop_duplicates().copy()
            feature_rows = []
            for row in questions.itertuples():
                a = parse_option(row.question, "A")
                b = parse_option(row.question, "B")
                feature_rows.append({
                    "question_id": row.question_id,
                    **{f"{name}_diff": b[name] - a[name] for name in a},
                })
            option_features = pd.DataFrame(feature_rows)

            modeling = annotation_profiles.merge(option_features, on="question_id", how="left")
            OPTION_COLUMNS = [column for column in option_features if column != "question_id"]
            ANNOTATION_COLUMNS = list(CODEBOOK)


            def grouped_choice_predictions(feature_columns):
                x = modeling[feature_columns]
                y = modeling["choice"].to_numpy()
                groups = modeling["question_id"].to_numpy()
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
                )
                probability = cross_val_predict(
                    model,
                    x,
                    y,
                    groups=groups,
                    cv=GroupKFold(n_splits=5),
                    method="predict_proba",
                )[:, 1]
                return {
                    "balanced_accuracy": balanced_accuracy_score(y, probability >= 0.5),
                    "log_loss": log_loss(y, probability),
                    "probability": probability,
                }


            model_sets = {
                "option features": OPTION_COLUMNS,
                "annotation profile": ANNOTATION_COLUMNS,
                "options + annotation": OPTION_COLUMNS + ANNOTATION_COLUMNS,
            }
            choice_results = []
            prediction_cache = {}
            for name, columns in model_sets.items():
                result = grouped_choice_predictions(columns)
                prediction_cache[name] = result["probability"]
                choice_results.append({
                    "model": name,
                    "n_features": len(columns),
                    "balanced_accuracy": result["balanced_accuracy"],
                    "log_loss": result["log_loss"],
                })

            choice_table = pd.DataFrame(choice_results).sort_values("log_loss")
            display(choice_table.round(3))
            '''
        ),
        md(
            r'''
            ## 5. Translate annotations into restricted candidate models

            Annotation is most useful for discovery when it nominates **computational ingredients**, not when it directly writes unrestricted code.

            The mapping below is intentionally explicit. For example:

            - expected-value language nominates `ev_diff`;
            - downside and certainty language nominate `worst_diff`, `sd_diff`, and certainty;
            - upside language nominates `best_diff` and the probability of the best outcome.

            Trusted code maps approved ingredients to estimators. The language model never executes arbitrary Python.
            '''
        ),
        code(
            r'''
            ANNOTATION_TO_FEATURES = {
                "expected_value": ["ev_diff"],
                "probability": ["p_best_diff", "p_zero_diff"],
                "probability_weighting": ["p_best_diff", "p_zero_diff"],
                "downside": ["worst_diff", "sd_diff"],
                "upside": ["best_diff", "p_best_diff"],
                "certainty": ["certain_diff", "sd_diff"],
                "risk": ["sd_diff", "worst_diff"],
            }

            mean_annotation_rate = annotation_profiles[ANNOTATION_COLUMNS].mean().sort_values(ascending=False)
            supported_ingredients = mean_annotation_rate.head(4).index.tolist()
            selected_features = sorted({
                feature
                for ingredient in supported_ingredients
                for feature in ANNOTATION_TO_FEATURES[ingredient]
            })

            annotation_guided_spec = {
                "name": "annotation-guided candidate",
                "ingredients": supported_ingredients,
                "features": selected_features,
            }
            print(json.dumps(annotation_guided_spec, indent=2))
            '''
        ),
        code(
            r'''
            candidate_specs = [
                {"name": "expected value", "features": ["ev_diff"]},
                {"name": "safety / downside", "features": ["worst_diff", "sd_diff", "certain_diff"]},
                {"name": "upside focus", "features": ["best_diff", "p_best_diff"]},
                {"name": annotation_guided_spec["name"], "features": annotation_guided_spec["features"]},
                {"name": "expanded option model", "features": OPTION_COLUMNS},
            ]

            discovery_rows = []
            discovery_probabilities = {}
            for spec in candidate_specs:
                result = grouped_choice_predictions(spec["features"])
                discovery_probabilities[spec["name"]] = result["probability"]
                discovery_rows.append({
                    "candidate": spec["name"],
                    "n_features": len(spec["features"]),
                    "balanced_accuracy": result["balanced_accuracy"],
                    "log_loss": result["log_loss"],
                })

            discovery_table = pd.DataFrame(discovery_rows).sort_values("log_loss")
            display(discovery_table.round(3))
            '''
        ),
        md(
            r'''
            ## 6. Implement a minimal propose-search-evaluate loop

            The table above evaluates a hand-written candidate set. We now let **DeepSeek V4 Pro** propose and revise candidates, while keeping the scientific controls outside the language model.

            We use one synthetic participant (`69576`) to mirror the participant-level design in the CCN model-discovery study:

            - **training:** Q01-Q11; behavior and preference-free think-aloud are visible to DeepSeek;
            - **validation:** Q12-Q15; Python fits and ranks proposed candidates;
            - **test:** Q16-Q19; hidden until one candidate has been selected.

            DeepSeek can select only from the seven approved decision variables. It cannot execute arbitrary code, inspect the test set, or directly choose the winner.

            This compact design adapts the separation of proposal, fitting, held-out evaluation, and think-aloud-constrained search in [Xie et al., *Think-Aloud Reshapes Automated Cognitive Model Discovery Beyond Behavior* (CCN 2026)](https://arxiv.org/abs/2605.05091).
            '''
        ),
        code(
            r'''
            def find_project_root():
                candidates = [Path.cwd(), Path.cwd().parent, Path("/content/LLM_human_decisions")]
                for candidate in candidates:
                    if (candidate / "scripts/minimal_discovery_loop.py").exists():
                        return candidate.resolve()
                raise FileNotFoundError("Could not locate scripts/minimal_discovery_loop.py")


            PROJECT_ROOT = find_project_root()
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))

            from scripts.minimal_discovery_loop import (
                OUTPUT_DIR as DISCOVERY_OUTPUT_DIR,
                load_participant_trials,
                run_loop,
                split_trials,
            )

            discovery_trials = load_participant_trials()
            discovery_splits = split_trials(discovery_trials)
            split_summary = pd.DataFrame([
                {
                    "split": name,
                    "questions": ", ".join(frame["question_id"]),
                    "n_trials": len(frame),
                    "A choices": int((frame["choice_label"] == "A").sum()),
                    "B choices": int((frame["choice_label"] == "B").sum()),
                    "visible to DeepSeek": "behavior + think-aloud" if name == "train" else "no",
                }
                for name, frame in discovery_splits.items()
            ])
            display(split_summary)
            display(
                discovery_splits["train"][["question_id", "choice_label", "think_aloud"]]
                .head(3)
                .rename(columns={"choice_label": "observed_choice", "think_aloud": "preference_free_think_aloud"})
            )
            '''
        ),
        md(
            r'''
            ### Three search iterations, with the test set kept closed

            The live path makes three API calls:

            1. propose four candidate models from the training evidence;
            2. show validation performance and errors, then request three revisions;
            3. repeat the validation-guided revision once more to test whether search is
               still improving or has plateaued.

            Python validates every JSON specification, fits its coefficients on training
            behavior, and ranks it using validation log loss. Only after all three search
            iterations do we select one candidate, refit it on training + validation, and
            open the four test questions once. More rounds are possible, but every extra
            round consumes validation information and can overfit the validation set.

            Set `RUN_DEEPSEEK_DISCOVERY = True` to rerun the calls. The default uses the cached teaching run, so every student can inspect the complete loop without an API key.
            '''
        ),
        code(
            r'''
            RUN_DEEPSEEK_DISCOVERY = False
            DISCOVERY_CACHE = DISCOVERY_OUTPUT_DIR / "minimal_discovery_run.json"

            if RUN_DEEPSEEK_DISCOVERY:
                discovery_run = run_loop()
                DISCOVERY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                DISCOVERY_CACHE.write_text(
                    json.dumps(discovery_run, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            else:
                discovery_run = json.loads(DISCOVERY_CACHE.read_text(encoding="utf-8"))

            print("Search iterations:", len(discovery_run["search_iterations"]))
            for iteration in discovery_run["search_iterations"]:
                print(iteration["stage"], "->", iteration["response"]["model"])
            print("Test access:", discovery_run["design"]["test_access"])
            '''
        ),
        code(
            r'''
            validation_frames = []
            for iteration in discovery_run["search_iterations"]:
                validation_frames.append(
                    pd.DataFrame(iteration["validation"]).assign(
                        iteration=iteration["iteration"],
                        stage=iteration["stage"],
                    )
                )

            validation_view = pd.concat(validation_frames, ignore_index=True)
            validation_view["features"] = validation_view["features"].map(", ".join)
            validation_view = validation_view[
                ["iteration", "stage", "name", "features", "validation_balanced_accuracy", "validation_log_loss"]
            ].sort_values(["iteration", "validation_log_loss"])
            display(validation_view.round(3))

            print("Selected before opening test:")
            print(json.dumps(discovery_run["selected_model"], indent=2))
            '''
        ),
        code(
            r'''
            test_rows = [discovery_run["test_result"], *discovery_run["test_baselines"]]
            test_view = pd.DataFrame([
                {
                    "model": row["name"],
                    "features": ", ".join(row["features"]),
                    "parameters": row["n_parameters"],
                    "test_balanced_accuracy": row["balanced_accuracy"],
                    "test_log_loss": row["log_loss"],
                }
                for row in test_rows
            ]).sort_values("test_log_loss")
            display(test_view.round(3))

            print(
                "Teaching result: the DeepSeek-selected candidate wins validation but does not "
                "beat the comparison baselines on four test trials. Two revision rounds also "
                "plateau at the same validation loss. This is a runnable search "
                "loop, not evidence of mechanism recovery."
            )
            '''
        ),
        md(
            r'''
            ## 7. Use disagreement to design the next test

            A candidate model becomes scientifically useful when it produces predictions that can be separated from alternatives. We therefore search for questions where the candidate probabilities disagree most.
            '''
        ),
        code(
            r'''
            candidate_probability_frame = pd.DataFrame(discovery_probabilities)
            candidate_probability_frame["question_id"] = modeling["question_id"].to_numpy()
            candidate_probability_frame["spread"] = (
                candidate_probability_frame[list(discovery_probabilities)].max(axis=1)
                - candidate_probability_frame[list(discovery_probabilities)].min(axis=1)
            )

            diagnostic = (
                candidate_probability_frame.groupby("question_id")["spread"]
                .mean()
                .sort_values(ascending=False)
                .head(5)
                .rename("mean_candidate_disagreement")
                .reset_index()
                .merge(questions, on="question_id", how="left")
            )
            display(diagnostic)
            '''
        ),
        md(
            r'''
            ## 8. The integrated evidence loop

            The implemented search loop now has an auditable structure:

            ```text
            training behavior + masked reasoning traces
                -> DeepSeek proposes restricted candidate specifications
                -> trusted Python fits parameters on training behavior
                -> validation feedback guides one revision
                -> select one candidate before opening the test set
                -> evaluate once on frozen test questions
            ```

            The earlier annotation-to-candidate route is one possible subloop:

            ```text
            masked reasoning traces
                → annotate information and operations
                → validate annotation stability and construct sensitivity
                → nominate restricted computational ingredients
                → fit candidate models on behavior
                → evaluate on held-out questions
                → inspect model–annotation disagreements
                → design the next diagnostic trial
            ```

            A compact Bayesian statement is:

            \[
            p(M \mid C,E) \propto p(C \mid M)\,p(E \mid M)\,p(M),
            \]

            where \(C\) is choice evidence and \(E\) is process evidence. This factorization is a modeling assumption, not a free guarantee of independence.

            ## What this notebook establishes—and what it does not

            **Supported if the controls succeed**

            - reasoning annotations detect a controlled synthetic manipulation;
            - preference-free process annotations contain held-question choice information;
            - annotations can restrict a transparent candidate-model search;
            - a minimal proposal-revision-evaluation loop can be audited end to end;
            - candidate disagreement identifies useful diagnostic questions.

            **Not established**

            - that an annotation is the true causal process;
            - that a prompted GPT-4 persona is equivalent to a human strategy;
            - that the best predictive candidate recovered the generating primitives;
            - that adding process evidence removes model non-identifiability.

            ## Classroom exercises

            1. Add one annotation label and define its falsifier.
            2. Compare the lexical baseline with one language-model annotator on 30 sentences.
            3. Repeat the analysis after shuffling annotations across trials within each question.
            4. Remove one candidate family and observe how the proposed diagnostic questions change.
            '''
        ),
    ]

    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
            "colab": {
                "name": "03_explanation_annotation_to_model_discovery.ipynb",
                "provenance": [],
            },
        }
    )
    return notebook


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
