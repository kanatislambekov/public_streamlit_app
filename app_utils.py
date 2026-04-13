from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import chi2

from model_artifacts import ARTIFACTS


AGE4_ORDER = ARTIFACTS["constants"]["AGE4_ORDER"]
CHILD_ORDER = ARTIFACTS["constants"]["CHILD_ORDER"]
EDU_ORDER = ARTIFACTS["constants"]["EDU_ORDER"]
HOUSEHOLD_GROUP_ORDER = ARTIFACTS["constants"]["HOUSEHOLD_GROUP_ORDER"]
SETTLEMENT_ORDER = ARTIFACTS["constants"]["SETTLEMENT_ORDER"]
SEX_ORDER = ARTIFACTS["constants"]["SEX_ORDER"]
HARMONIZED_REGION_ORDER = ARTIFACTS["constants"]["HARMONIZED_REGION_ORDER"]
YEAR_ORDER = ARTIFACTS["constants"]["YEAR_ORDER"]

AGE_REFERENCE = AGE4_ORDER[0]
EDU_REFERENCE = "Higher+"

DISPLAY_LABELS = {
    "Higher+": "Higher education or above",
    "Primary or less": "Primary education or less",
    "Lower secondary": "Lower secondary education",
    "Secondary / technical": "Secondary or technical education",
    "No child household member": "No child household member",
    "Lives with child household member": "Lives with child household member",
}

PROFILE_COLUMNS = [
    "survey_year_factor",
    "age_group4",
    "sex_factor",
    "region_harmonized",
    "edu_group",
    "children_u18_label",
    "household_group",
    "settlement_type",
]

MLOGIT_SPECS = ARTIFACTS["multinomial"]


@dataclass(frozen=True)
class BinaryArtifact:
    design_columns: list[str]
    params: np.ndarray
    cov_matrix: np.ndarray


@dataclass(frozen=True)
class MultinomialArtifact:
    model_key: str
    model_title: str
    sample_note: str
    baseline: str
    categories: list[str]
    design_columns: list[str]
    beta_matrix: np.ndarray
    cov_matrix: np.ndarray


def _slugify(value: str) -> str:
    slug = value.replace("+", "plus").replace("-", "_").replace("/", "_").replace(" ", "_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def display_label(value: str) -> str:
    return DISPLAY_LABELS.get(value, value)


def default_profile() -> dict[str, str]:
    return {
        "survey_year_factor": YEAR_ORDER[-1],
        "age_group4": AGE_REFERENCE,
        "sex_factor": "Female",
        "region_harmonized": "Almaty city",
        "edu_group": EDU_REFERENCE,
        "children_u18_label": "No child household member",
        "household_group": "1 person",
        "settlement_type": "Urban",
    }


def make_profile_frame(profile: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame([{col: profile[col] for col in PROFILE_COLUMNS}])


def add_dummies(out: pd.DataFrame, series: pd.Series, prefix: str, reference: str) -> None:
    dummies = pd.get_dummies(series, prefix=prefix, dtype=float)
    reference_col = f"{prefix}_{reference}"
    for col in sorted(dummies.columns):
        if col != reference_col:
            out[col] = dummies[col].astype(float)


def build_design_matrix(df: pd.DataFrame, include_age_education_interactions: bool = True) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["Intercept"] = 1.0

    add_dummies(out, pd.Categorical(df["age_group4"], categories=AGE4_ORDER, ordered=True), "age", AGE_REFERENCE)
    add_dummies(out, pd.Categorical(df["sex_factor"], categories=SEX_ORDER, ordered=True), "sex", "Female")
    add_dummies(
        out,
        pd.Categorical(df["region_harmonized"], categories=HARMONIZED_REGION_ORDER, ordered=True),
        "region",
        "Almaty city",
    )
    add_dummies(out, pd.Categorical(df["edu_group"], categories=EDU_ORDER, ordered=True), "edu", EDU_REFERENCE)
    add_dummies(
        out,
        pd.Categorical(df["children_u18_label"], categories=CHILD_ORDER, ordered=True),
        "child",
        "No child household member",
    )
    add_dummies(
        out,
        pd.Categorical(df["household_group"], categories=HOUSEHOLD_GROUP_ORDER, ordered=True),
        "hh",
        "1 person",
    )
    add_dummies(
        out,
        pd.Categorical(df["settlement_type"], categories=SETTLEMENT_ORDER, ordered=True),
        "settle",
        "Urban",
    )
    add_dummies(
        out,
        pd.Categorical(df["survey_year_factor"], categories=YEAR_ORDER, ordered=True),
        "year",
        "2020",
    )

    if include_age_education_interactions:
        for age in AGE4_ORDER[1:]:
            age_mask = df["age_group4"].eq(age).astype(float)
            for edu in [level for level in EDU_ORDER if level != EDU_REFERENCE]:
                edu_mask = df["edu_group"].eq(edu).astype(float)
                out[f"age_{_slugify(age)}_x_edu_{_slugify(edu)}"] = age_mask * edu_mask

    return out


def get_binary_artifact() -> BinaryArtifact:
    artifact = ARTIFACTS["binary_age_education"]
    return BinaryArtifact(
        design_columns=list(artifact["design_columns"]),
        params=np.asarray(artifact["params"], dtype=float),
        cov_matrix=np.asarray(artifact["cov_matrix"], dtype=float),
    )


def get_multinomial_artifact(model_key: str) -> MultinomialArtifact:
    artifact = MLOGIT_SPECS[model_key]
    return MultinomialArtifact(
        model_key=model_key,
        model_title=artifact["title"],
        sample_note=artifact["sample_note"],
        baseline=artifact["baseline"],
        categories=list(artifact["categories"]),
        design_columns=list(artifact["design_columns"]),
        beta_matrix=np.asarray(artifact["beta_matrix"], dtype=float),
        cov_matrix=np.asarray(artifact["cov_matrix"], dtype=float),
    )


def aligned_binary_design(profile_df: pd.DataFrame) -> pd.DataFrame:
    artifact = get_binary_artifact()
    return build_design_matrix(profile_df, include_age_education_interactions=True).reindex(
        columns=artifact.design_columns,
        fill_value=0.0,
    )


def aligned_multinomial_design(profile_df: pd.DataFrame, model_key: str) -> pd.DataFrame:
    artifact = get_multinomial_artifact(model_key)
    return build_design_matrix(profile_df, include_age_education_interactions=False).reindex(
        columns=artifact.design_columns,
        fill_value=0.0,
    )


def probability_from_design_row(exog_row: pd.Series) -> dict[str, float]:
    artifact = get_binary_artifact()
    x = exog_row.to_numpy(dtype=float)
    eta = float(x @ artifact.params)
    probability = float(expit(np.clip(eta, -700, 700)))
    gradient = probability * (1.0 - probability) * x
    variance = float(gradient @ artifact.cov_matrix @ gradient)
    se = float(np.sqrt(max(variance, 0.0)))
    lower = max(0.0, probability - 1.96 * se)
    upper = min(1.0, probability + 1.96 * se)
    return {
        "probability": probability,
        "probability_pct": 100.0 * probability,
        "ci_lower_pct": 100.0 * lower,
        "ci_upper_pct": 100.0 * upper,
    }


def predict_profile(profile: dict[str, str]) -> dict[str, float]:
    profile_df = make_profile_frame(profile)
    exog = aligned_binary_design(profile_df)
    return probability_from_design_row(exog.iloc[0])


def probability_difference(left_profile: dict[str, str], right_profile: dict[str, str]) -> dict[str, float]:
    artifact = get_binary_artifact()
    profiles = pd.DataFrame([left_profile, right_profile])
    exog = aligned_binary_design(profiles)
    x_left = exog.iloc[0].to_numpy(dtype=float)
    x_right = exog.iloc[1].to_numpy(dtype=float)

    eta_left = float(x_left @ artifact.params)
    eta_right = float(x_right @ artifact.params)
    prob_left = float(expit(np.clip(eta_left, -700, 700)))
    prob_right = float(expit(np.clip(eta_right, -700, 700)))

    grad_left = prob_left * (1.0 - prob_left) * x_left
    grad_right = prob_right * (1.0 - prob_right) * x_right
    gradient = grad_left - grad_right
    variance = float(gradient @ artifact.cov_matrix @ gradient)
    se = float(np.sqrt(max(variance, 0.0)))
    diff = prob_left - prob_right
    return {
        "difference_pct_points": 100.0 * diff,
        "ci_lower_pct_points": 100.0 * (diff - 1.96 * se),
        "ci_upper_pct_points": 100.0 * (diff + 1.96 * se),
    }


def interaction_probability_grid(base_profile: dict[str, str]) -> pd.DataFrame:
    rows = []
    for age in AGE4_ORDER:
        for education in EDU_ORDER:
            profile = {**base_profile, "age_group4": age, "edu_group": education}
            pred = predict_profile(profile)
            rows.append(
                {
                    "age_group4": age,
                    "edu_group": education,
                    "predicted_probability_pct": pred["probability_pct"],
                    "ci_lower_pct": pred["ci_lower_pct"],
                    "ci_upper_pct": pred["ci_upper_pct"],
                }
            )
    return pd.DataFrame(rows)


def education_marginal_effects(base_profile: dict[str, str]) -> pd.DataFrame:
    rows = []
    for age in AGE4_ORDER:
        reference_profile = {**base_profile, "age_group4": age, "edu_group": EDU_REFERENCE}
        reference_prediction = predict_profile(reference_profile)
        for education in [level for level in EDU_ORDER if level != EDU_REFERENCE]:
            comparison_profile = {**reference_profile, "edu_group": education}
            diff = probability_difference(comparison_profile, reference_profile)
            rows.append(
                {
                    "age_group4": age,
                    "education_level": education,
                    "comparison": f"{display_label(education)} vs {display_label(EDU_REFERENCE)}",
                    "predicted_probability_pct": predict_profile(comparison_profile)["probability_pct"],
                    "reference_probability_pct": reference_prediction["probability_pct"],
                    "difference_pct_points": diff["difference_pct_points"],
                    "ci_lower_pct_points": diff["ci_lower_pct_points"],
                    "ci_upper_pct_points": diff["ci_upper_pct_points"],
                }
            )
    return pd.DataFrame(rows)


def age_marginal_effects(base_profile: dict[str, str]) -> pd.DataFrame:
    rows = []
    for education in EDU_ORDER:
        reference_profile = {**base_profile, "age_group4": AGE_REFERENCE, "edu_group": education}
        reference_prediction = predict_profile(reference_profile)
        for age in AGE4_ORDER[1:]:
            comparison_profile = {**reference_profile, "age_group4": age}
            diff = probability_difference(comparison_profile, reference_profile)
            rows.append(
                {
                    "edu_group": education,
                    "age_group4": age,
                    "comparison": f"{age} vs {AGE_REFERENCE}",
                    "predicted_probability_pct": predict_profile(comparison_profile)["probability_pct"],
                    "reference_probability_pct": reference_prediction["probability_pct"],
                    "difference_pct_points": diff["difference_pct_points"],
                    "ci_lower_pct_points": diff["ci_lower_pct_points"],
                    "ci_upper_pct_points": diff["ci_upper_pct_points"],
                }
            )
    return pd.DataFrame(rows)


def interaction_wald_test() -> dict[str, float]:
    artifact = get_binary_artifact()
    terms = [i for i, name in enumerate(artifact.design_columns) if "_x_edu_" in name]
    beta = artifact.params[terms]
    cov = artifact.cov_matrix[np.ix_(terms, terms)]
    stat = float(beta.T @ np.linalg.pinv(cov) @ beta)
    df = len(terms)
    p_value = float(1.0 - chi2.cdf(stat, df))
    return {"chi2": stat, "df": df, "p_value": p_value}


def multinomial_specs() -> dict[str, dict[str, object]]:
    return MLOGIT_SPECS


def multinomial_probabilities(model_key: str, profile: dict[str, str]) -> pd.DataFrame:
    artifact = get_multinomial_artifact(model_key)
    x = aligned_multinomial_design(make_profile_frame(profile), model_key).iloc[0].to_numpy(dtype=float)
    eta = np.clip(artifact.beta_matrix @ x, -700, 700)
    exp_eta = np.exp(eta)
    denom = 1.0 + exp_eta.sum()
    probs = np.concatenate([[1.0 / denom], exp_eta / denom])

    rows = []
    n_nonbase = len(artifact.categories) - 1
    p = len(artifact.design_columns)
    for j, category in enumerate(artifact.categories):
        gradient = np.zeros(n_nonbase * p, dtype=float)
        if j == 0:
            for l in range(n_nonbase):
                gradient[l * p : (l + 1) * p] = -probs[0] * probs[l + 1] * x
        else:
            for l in range(n_nonbase):
                multiplier = probs[j] * ((1.0 if l == (j - 1) else 0.0) - probs[l + 1])
                gradient[l * p : (l + 1) * p] = multiplier * x
        variance = float(gradient @ artifact.cov_matrix @ gradient)
        se = float(np.sqrt(max(variance, 0.0)))
        lower = max(0.0, probs[j] - 1.96 * se)
        upper = min(1.0, probs[j] + 1.96 * se)
        rows.append(
            {
                "category": category,
                "predicted_probability_pct": 100.0 * float(probs[j]),
                "ci_lower_pct": 100.0 * lower,
                "ci_upper_pct": 100.0 * upper,
            }
        )
    return pd.DataFrame(rows)
