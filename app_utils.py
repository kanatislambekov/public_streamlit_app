from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import product

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

PROFILE_VALUE_ORDERS = {
    "survey_year_factor": YEAR_ORDER,
    "age_group4": AGE4_ORDER,
    "sex_factor": SEX_ORDER,
    "region_harmonized": HARMONIZED_REGION_ORDER,
    "edu_group": EDU_ORDER,
    "children_u18_label": CHILD_ORDER,
    "household_group": HOUSEHOLD_GROUP_ORDER,
    "settlement_type": SETTLEMENT_ORDER,
}

PROFILE_FIELD_LABELS = {
    "survey_year_factor": "Survey year",
    "age_group4": "Age group",
    "sex_factor": "Sex",
    "region_harmonized": "Region",
    "edu_group": "Education",
    "children_u18_label": "Child in household",
    "household_group": "Household size",
    "settlement_type": "Settlement type",
}

MAX_PROFILE_COMBINATIONS = 128
VALIDATION_TOLERANCE = 1e-6

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


def _preview_values(values: Sequence[str], limit: int = 5) -> str:
    preview = ", ".join(values[:limit])
    if len(values) > limit:
        return f"{preview}, ..."
    return preview


def validate_profile(profile: Mapping[str, str]) -> None:
    missing_columns = [PROFILE_FIELD_LABELS[col] for col in PROFILE_COLUMNS if col not in profile]
    extra_columns = sorted(set(profile) - set(PROFILE_COLUMNS))
    issues = []

    if missing_columns:
        issues.append(f"Missing profile inputs: {', '.join(missing_columns)}.")
    if extra_columns:
        issues.append(f"Unexpected profile inputs: {', '.join(extra_columns)}.")

    for column in PROFILE_COLUMNS:
        if column not in profile:
            continue
        value = profile[column]
        if value not in PROFILE_VALUE_ORDERS[column]:
            issues.append(f"{PROFILE_FIELD_LABELS[column]} must be one of {', '.join(PROFILE_VALUE_ORDERS[column])}.")

    if issues:
        raise ValueError(" ".join(issues))


def generate_profile_combinations(
    selections: Mapping[str, Sequence[str]],
    max_profiles: int = MAX_PROFILE_COMBINATIONS,
) -> list[dict[str, str]]:
    normalized_selections: dict[str, list[str]] = {}
    combination_count = 1

    for column in PROFILE_COLUMNS:
        selected_values = list(dict.fromkeys(selections.get(column, [])))
        if not selected_values:
            raise ValueError(f"Select at least one value for {PROFILE_FIELD_LABELS[column]}.")

        invalid_values = [value for value in selected_values if value not in PROFILE_VALUE_ORDERS[column]]
        if invalid_values:
            raise ValueError(
                f"{PROFILE_FIELD_LABELS[column]} contains invalid selections: {_preview_values(invalid_values)}."
            )

        ordered_values = [value for value in PROFILE_VALUE_ORDERS[column] if value in set(selected_values)]
        normalized_selections[column] = ordered_values
        combination_count *= len(ordered_values)

    if combination_count > max_profiles:
        raise ValueError(
            f"Current selections create {combination_count} profile combinations. "
            f"Please narrow the filters to {max_profiles} combinations or fewer."
        )

    return [
        dict(zip(PROFILE_COLUMNS, values))
        for values in product(*(normalized_selections[column] for column in PROFILE_COLUMNS))
    ]


def profile_label(profile: Mapping[str, str]) -> str:
    validate_profile(profile)
    return " | ".join(
        [
            profile["survey_year_factor"],
            profile["age_group4"],
            profile["sex_factor"],
            profile["region_harmonized"],
            display_label(profile["edu_group"]),
            display_label(profile["children_u18_label"]),
            profile["household_group"],
            profile["settlement_type"],
        ]
    )


def make_profile_frame(profile: dict[str, str]) -> pd.DataFrame:
    validate_profile(profile)
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


@lru_cache(maxsize=1)
def _profile_universe() -> pd.DataFrame:
    return pd.DataFrame(
        product(
            YEAR_ORDER,
            AGE4_ORDER,
            SEX_ORDER,
            HARMONIZED_REGION_ORDER,
            EDU_ORDER,
            CHILD_ORDER,
            HOUSEHOLD_GROUP_ORDER,
            SETTLEMENT_ORDER,
        ),
        columns=PROFILE_COLUMNS,
    )


@lru_cache(maxsize=2)
def _available_design_columns(include_age_education_interactions: bool) -> tuple[str, ...]:
    return tuple(build_design_matrix(_profile_universe(), include_age_education_interactions).columns)


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


@lru_cache(maxsize=1)
def validate_artifacts() -> tuple[str, ...]:
    issues: list[str] = []

    binary_artifact = get_binary_artifact()
    binary_columns = len(binary_artifact.design_columns)
    available_binary_columns = set(_available_design_columns(include_age_education_interactions=True))

    if len(binary_artifact.params) != binary_columns:
        issues.append(
            "Binary model parameter count does not match the design matrix columns "
            f"({len(binary_artifact.params)} vs {binary_columns})."
        )
    if binary_artifact.cov_matrix.shape != (binary_columns, binary_columns):
        issues.append(
            "Binary model covariance matrix shape does not match the design matrix "
            f"({binary_artifact.cov_matrix.shape} vs {(binary_columns, binary_columns)})."
        )

    missing_binary_columns = sorted(set(binary_artifact.design_columns) - available_binary_columns)
    if missing_binary_columns:
        issues.append(
            "Binary model design columns are not generated by the app design matrix: "
            f"{_preview_values(missing_binary_columns)}."
        )

    available_multinomial_columns = set(_available_design_columns(include_age_education_interactions=False))
    for model_key in MLOGIT_SPECS:
        artifact = get_multinomial_artifact(model_key)
        design_columns = len(artifact.design_columns)
        n_nonbase = len(artifact.categories) - 1
        expected_beta_shape = (n_nonbase, design_columns)
        expected_cov_shape = (n_nonbase * design_columns, n_nonbase * design_columns)

        if len(set(artifact.categories)) != len(artifact.categories):
            issues.append(f"{model_key}: categories contain duplicates.")
        if artifact.categories[0] != artifact.baseline:
            issues.append(f"{model_key}: the baseline category is not first in the category list.")
        if artifact.beta_matrix.shape != expected_beta_shape:
            issues.append(
                f"{model_key}: beta matrix shape {artifact.beta_matrix.shape} does not match {expected_beta_shape}."
            )
        if artifact.cov_matrix.shape != expected_cov_shape:
            issues.append(
                f"{model_key}: covariance matrix shape {artifact.cov_matrix.shape} does not match {expected_cov_shape}."
            )

        missing_design_columns = sorted(set(artifact.design_columns) - available_multinomial_columns)
        if missing_design_columns:
            issues.append(
                f"{model_key}: design columns are not generated by the app design matrix: "
                f"{_preview_values(missing_design_columns)}."
            )

    return tuple(issues)


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


def profile_predictions_table(profiles: Sequence[Mapping[str, str]]) -> pd.DataFrame:
    rows = []
    for profile in profiles:
        validated_profile = {column: profile[column] for column in PROFILE_COLUMNS}
        prediction = predict_profile(validated_profile)
        rows.append(
            {
                "profile_label": profile_label(validated_profile),
                **validated_profile,
                "probability_pct": prediction["probability_pct"],
                "ci_lower_pct": prediction["ci_lower_pct"],
                "ci_upper_pct": prediction["ci_upper_pct"],
            }
        )
    return pd.DataFrame(rows)


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


def _validate_interval(
    estimate: float,
    lower: float,
    upper: float,
    label: str,
    lower_bound: float,
    upper_bound: float,
    tolerance: float = VALIDATION_TOLERANCE,
) -> list[str]:
    if not all(np.isfinite(value) for value in (estimate, lower, upper)):
        return [f"{label}: encountered a non-finite value in the estimate or confidence interval."]

    issues = []
    if estimate < lower_bound - tolerance or estimate > upper_bound + tolerance:
        issues.append(f"{label}: estimate {estimate:.6f} falls outside [{lower_bound:.1f}, {upper_bound:.1f}].")
    if lower < lower_bound - tolerance or lower > upper_bound + tolerance:
        issues.append(f"{label}: lower bound {lower:.6f} falls outside [{lower_bound:.1f}, {upper_bound:.1f}].")
    if upper < lower_bound - tolerance or upper > upper_bound + tolerance:
        issues.append(f"{label}: upper bound {upper:.6f} falls outside [{lower_bound:.1f}, {upper_bound:.1f}].")
    if lower > estimate + tolerance or estimate > upper + tolerance:
        issues.append(f"{label}: interval ordering is invalid ({lower:.6f}, {estimate:.6f}, {upper:.6f}).")
    return issues


def validate_binary_prediction(prediction: Mapping[str, float], label: str = "Selected profile") -> list[str]:
    required_keys = {"probability", "probability_pct", "ci_lower_pct", "ci_upper_pct"}
    missing_keys = sorted(required_keys - set(prediction))
    if missing_keys:
        return [f"{label}: prediction output is missing keys: {', '.join(missing_keys)}."]

    issues = _validate_interval(
        float(prediction["probability_pct"]),
        float(prediction["ci_lower_pct"]),
        float(prediction["ci_upper_pct"]),
        label,
        lower_bound=0.0,
        upper_bound=100.0,
    )

    probability = float(prediction["probability"])
    if not np.isfinite(probability) or probability < -VALIDATION_TOLERANCE or probability > 1.0 + VALIDATION_TOLERANCE:
        issues.append(f"{label}: raw probability {probability:.6f} falls outside [0, 1].")
    if abs((100.0 * probability) - float(prediction["probability_pct"])) > VALIDATION_TOLERANCE:
        issues.append(f"{label}: raw and percent probabilities are inconsistent.")
    return issues


def validate_probability_frame(
    df: pd.DataFrame,
    probability_col: str,
    ci_lower_col: str,
    ci_upper_col: str,
    label: str,
    row_label_col: str | None = None,
    lower_bound: float = 0.0,
    upper_bound: float = 100.0,
    tolerance: float = VALIDATION_TOLERANCE,
) -> list[str]:
    required_columns = [probability_col, ci_lower_col, ci_upper_col]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        return [f"{label}: missing required columns {', '.join(missing_columns)}."]
    if df.empty:
        return [f"{label}: no rows were produced."]

    issues = []
    for row_number, (_, row) in enumerate(df.iterrows(), start=1):
        row_label = str(row[row_label_col]) if row_label_col and row_label_col in df.columns else f"row {row_number}"
        issues.extend(
            _validate_interval(
                float(row[probability_col]),
                float(row[ci_lower_col]),
                float(row[ci_upper_col]),
                f"{label} ({row_label})",
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                tolerance=tolerance,
            )
        )
    return issues


def validate_difference_frame(
    df: pd.DataFrame,
    predicted_col: str,
    reference_col: str,
    difference_col: str,
    ci_lower_col: str,
    ci_upper_col: str,
    label: str,
    row_label_col: str | None = None,
    tolerance: float = VALIDATION_TOLERANCE,
) -> list[str]:
    required_columns = [predicted_col, reference_col, difference_col, ci_lower_col, ci_upper_col]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        return [f"{label}: missing required columns {', '.join(missing_columns)}."]
    if df.empty:
        return [f"{label}: no rows were produced."]

    issues = []
    for row_number, (_, row) in enumerate(df.iterrows(), start=1):
        row_label = str(row[row_label_col]) if row_label_col and row_label_col in df.columns else f"row {row_number}"
        issues.extend(
            _validate_interval(
                float(row[difference_col]),
                float(row[ci_lower_col]),
                float(row[ci_upper_col]),
                f"{label} ({row_label})",
                lower_bound=-100.0,
                upper_bound=100.0,
                tolerance=tolerance,
            )
        )

        computed_difference = float(row[predicted_col]) - float(row[reference_col])
        if not np.isfinite(computed_difference):
            issues.append(f"{label} ({row_label}): predicted and reference values are not finite.")
        elif abs(computed_difference - float(row[difference_col])) > tolerance:
            issues.append(
                f"{label} ({row_label}): reported difference {float(row[difference_col]):.6f} "
                f"does not match predicted-reference gap {computed_difference:.6f}."
            )
    return issues


def validate_multinomial_probability_frame(
    df: pd.DataFrame,
    label: str,
    probability_col: str = "predicted_probability_pct",
    ci_lower_col: str = "ci_lower_pct",
    ci_upper_col: str = "ci_upper_pct",
    row_label_col: str = "category",
    tolerance: float = VALIDATION_TOLERANCE,
) -> list[str]:
    issues = validate_probability_frame(
        df,
        probability_col=probability_col,
        ci_lower_col=ci_lower_col,
        ci_upper_col=ci_upper_col,
        label=label,
        row_label_col=row_label_col,
        lower_bound=0.0,
        upper_bound=100.0,
        tolerance=tolerance,
    )

    if probability_col not in df.columns:
        return issues

    total_probability = float(df[probability_col].sum())
    if not np.isfinite(total_probability) or abs(total_probability - 100.0) > tolerance:
        issues.append(f"{label}: predicted probabilities sum to {total_probability:.10f}% instead of 100%.")
    return issues


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
