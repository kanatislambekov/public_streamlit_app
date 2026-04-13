from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app_utils import (
    AGE4_ORDER,
    CHILD_ORDER,
    EDU_ORDER,
    HARMONIZED_REGION_ORDER,
    HOUSEHOLD_GROUP_ORDER,
    MAX_PROFILE_COMBINATIONS,
    SETTLEMENT_ORDER,
    SEX_ORDER,
    YEAR_ORDER,
    age_marginal_effects,
    default_profile,
    display_label,
    education_marginal_effects,
    generate_profile_combinations,
    interaction_probability_grid,
    interaction_wald_test,
    multinomial_probabilities,
    multinomial_specs,
    profile_label,
    profile_predictions_table,
    predict_profile,
    validate_artifacts,
    validate_binary_prediction,
    validate_difference_frame,
    validate_multinomial_probability_frame,
    validate_probability_frame,
)


st.set_page_config(page_title="Smoking Probability Dashboard", layout="wide")


def format_percent_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[col] = out[col].map(lambda value: f"{value:.1f}")
    return out


def horizontal_axis(title: str, label_limit: int = 220) -> alt.Axis:
    return alt.Axis(title=title, labelAngle=0, labelLimit=label_limit)


def format_profile_summary_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["edu_group"] = out["edu_group"].map(display_label)
    out["children_u18_label"] = out["children_u18_label"].map(display_label)
    return out.rename(
        columns={
            "profile_label": "Profile",
            "survey_year_factor": "Survey year",
            "age_group4": "Age group",
            "sex_factor": "Sex",
            "region_harmonized": "Region",
            "edu_group": "Education",
            "children_u18_label": "Child in household",
            "household_group": "Household size",
            "settlement_type": "Settlement type",
            "probability_pct": "Predicted %",
            "ci_lower_pct": "CI lower %",
            "ci_upper_pct": "CI upper %",
        }
    )


def render_validation_status(messages: list[str], title: str, success_message: str) -> None:
    if messages:
        preview = messages[:10]
        st.error(f"{title}\n\n" + "\n".join(f"- {message}" for message in preview))
        if len(messages) > len(preview):
            st.caption(f"Showing the first {len(preview)} of {len(messages)} validation issues.")
        return
    st.success(success_message)


def main() -> None:
    st.title("Smoking Probability Dashboard")
    st.caption(
        "Standalone public-safe app built from sanitized pooled-model artifacts for the 2020-2024 surveys. "
        "No respondent-level survey tables are included in this app package."
    )

    defaults = default_profile()
    wald = interaction_wald_test()
    artifact_issues = list(validate_artifacts())
    if artifact_issues:
        render_validation_status(
            artifact_issues,
            title="Model artifact validation failed.",
            success_message="",
        )
        st.stop()

    with st.sidebar:
        st.header("Profile Inputs")
        st.caption(
            "Select one or more values per input to compare multiple profile combinations. "
            f"The app will stop if the current filters expand past {MAX_PROFILE_COMBINATIONS} profiles."
        )
        survey_years = st.multiselect(
            "Survey year",
            YEAR_ORDER,
            default=[defaults["survey_year_factor"]],
        )
        age_groups = st.multiselect("Age group", AGE4_ORDER, default=[defaults["age_group4"]])
        sexes = st.multiselect("Sex", SEX_ORDER, default=[defaults["sex_factor"]])
        regions = st.multiselect(
            "Region",
            HARMONIZED_REGION_ORDER,
            default=[defaults["region_harmonized"]],
        )
        educations = st.multiselect(
            "Education",
            EDU_ORDER,
            default=[defaults["edu_group"]],
            format_func=display_label,
        )
        child_statuses = st.multiselect(
            "Child in household",
            CHILD_ORDER,
            default=[defaults["children_u18_label"]],
            format_func=display_label,
        )
        household_groups = st.multiselect(
            "Household size",
            HOUSEHOLD_GROUP_ORDER,
            default=[defaults["household_group"]],
        )
        settlements = st.multiselect(
            "Settlement type",
            SETTLEMENT_ORDER,
            default=[defaults["settlement_type"]],
        )

    selections = {
        "survey_year_factor": survey_years,
        "age_group4": age_groups,
        "sex_factor": sexes,
        "region_harmonized": regions,
        "edu_group": educations,
        "children_u18_label": child_statuses,
        "household_group": household_groups,
        "settlement_type": settlements,
    }

    try:
        selected_profiles = generate_profile_combinations(selections, max_profiles=MAX_PROFILE_COMBINATIONS)
    except ValueError as exc:
        st.sidebar.error(str(exc))
        st.stop()

    profile_lookup = {profile_label(profile): profile for profile in selected_profiles}
    profile_labels = list(profile_lookup)
    default_focus_label = profile_labels[0]
    if all(defaults[column] in selections[column] for column in selections):
        default_focus_label = profile_label(defaults)

    st.subheader("Selected Profiles")
    st.caption(
        f"{len(selected_profiles)} profile combination(s) were generated from the current sidebar filters. "
        "The detailed charts below use the focused profile."
    )
    if len(profile_labels) > 1:
        focused_profile_label = st.selectbox(
            "Focused profile",
            options=profile_labels,
            index=profile_labels.index(default_focus_label),
        )
    else:
        focused_profile_label = profile_labels[0]

    selected_profile_predictions = profile_predictions_table(selected_profiles)
    st.dataframe(
        format_percent_frame(
            format_profile_summary_frame(selected_profile_predictions),
            ["Predicted %", "CI lower %", "CI upper %"],
        ),
        use_container_width=True,
        hide_index=True,
    )

    profile = profile_lookup[focused_profile_label]
    age_group = profile["age_group4"]
    region = profile["region_harmonized"]
    education = profile["edu_group"]

    profile_prediction = predict_profile(profile)
    grid = interaction_probability_grid(profile)
    education_effects = education_marginal_effects(profile)
    age_effects = age_marginal_effects(profile)
    model_options = multinomial_specs()
    multinomial_outputs = {
        model_key: multinomial_probabilities(model_key, profile)
        for model_key in model_options
    }

    validation_issues = []
    validation_issues.extend(
        validate_probability_frame(
            selected_profile_predictions,
            probability_col="probability_pct",
            ci_lower_col="ci_lower_pct",
            ci_upper_col="ci_upper_pct",
            label="Selected profile predictions",
            row_label_col="profile_label",
        )
    )
    validation_issues.extend(validate_binary_prediction(profile_prediction, label="Focused profile"))
    validation_issues.extend(
        validate_probability_frame(
            grid,
            probability_col="predicted_probability_pct",
            ci_lower_col="ci_lower_pct",
            ci_upper_col="ci_upper_pct",
            label="Age x education grid",
        )
    )
    validation_issues.extend(
        validate_difference_frame(
            education_effects,
            predicted_col="predicted_probability_pct",
            reference_col="reference_probability_pct",
            difference_col="difference_pct_points",
            ci_lower_col="ci_lower_pct_points",
            ci_upper_col="ci_upper_pct_points",
            label="Education marginal effects",
            row_label_col="comparison",
        )
    )
    validation_issues.extend(
        validate_difference_frame(
            age_effects,
            predicted_col="predicted_probability_pct",
            reference_col="reference_probability_pct",
            difference_col="difference_pct_points",
            ci_lower_col="ci_lower_pct_points",
            ci_upper_col="ci_upper_pct_points",
            label="Age marginal effects",
            row_label_col="comparison",
        )
    )
    for model_key, mlogit_probs in multinomial_outputs.items():
        validation_issues.extend(
            validate_multinomial_probability_frame(
                mlogit_probs,
                label=f"Multinomial model: {model_options[model_key]['title']}",
            )
        )
    if not np.isfinite(wald["chi2"]) or wald["chi2"] < 0:
        validation_issues.append("Age x education joint test produced an invalid chi-squared statistic.")
    if not np.isfinite(wald["p_value"]) or wald["p_value"] < 0 or wald["p_value"] > 1:
        validation_issues.append("Age x education joint test produced an invalid p-value.")

    render_validation_status(
        validation_issues,
        title="Validation checks found inconsistent outputs.",
        success_message=(
            "Validation checks passed for the selected profile inputs, focused binary outputs, "
            "marginal effects, and all multinomial probability tables."
        ),
    )

    focused_education_effects = education_effects.loc[education_effects["age_group4"] == age_group].copy()
    focused_education_effects["education_display"] = focused_education_effects["education_level"].map(display_label)
    focused_age_effects = age_effects.loc[age_effects["edu_group"] == education].copy()

    st.subheader("Focused Profile")
    metric_col, ci_col, test_col = st.columns(3)
    metric_col.metric("Predicted probability of current smoking", f"{profile_prediction['probability_pct']:.1f}%")
    ci_col.metric(
        "95% confidence interval",
        f"{profile_prediction['ci_lower_pct']:.1f}% to {profile_prediction['ci_upper_pct']:.1f}%",
    )
    test_col.metric("Age x education joint test", f"p = {wald['p_value']:.3f}")

    st.write(
        f"For a {age_group} respondent in {region}, with {display_label(education).lower()}, "
        f"the model estimates a {profile_prediction['probability_pct']:.1f}% probability of current smoking "
        f"after holding the other selected characteristics constant."
    )

    st.subheader("Adjusted Probability Grid: Age x Education")
    grid_chart = grid.copy()
    grid_chart["edu_display"] = grid_chart["edu_group"].map(display_label)
    heatmap = (
        alt.Chart(grid_chart)
        .mark_rect(cornerRadius=6)
        .encode(
            x=alt.X(
                "edu_display:N",
                title="Education",
                sort=[display_label(level) for level in EDU_ORDER],
                axis=horizontal_axis("Education"),
            ),
            y=alt.Y("age_group4:N", title="Age group", sort=AGE4_ORDER),
            color=alt.Color("predicted_probability_pct:Q", title="Predicted smoking (%)", scale=alt.Scale(scheme="oranges")),
            tooltip=[
                alt.Tooltip("age_group4:N", title="Age group"),
                alt.Tooltip("edu_display:N", title="Education"),
                alt.Tooltip("predicted_probability_pct:Q", title="Predicted %", format=".1f"),
                alt.Tooltip("ci_lower_pct:Q", title="95% CI lower", format=".1f"),
                alt.Tooltip("ci_upper_pct:Q", title="95% CI upper", format=".1f"),
            ],
        )
        .properties(height=280)
    )
    labels = (
        alt.Chart(grid_chart)
        .mark_text(size=12)
        .encode(
            x=alt.X(
                "edu_display:N",
                sort=[display_label(level) for level in EDU_ORDER],
                axis=horizontal_axis("Education"),
            ),
            y=alt.Y("age_group4:N", sort=AGE4_ORDER),
            text=alt.Text("predicted_probability_pct:Q", format=".1f"),
            color=alt.value("black"),
        )
    )
    st.altair_chart(heatmap + labels, use_container_width=True)

    grid_display = grid.copy()
    grid_display["edu_group"] = grid_display["edu_group"].map(display_label)
    st.dataframe(
        format_percent_frame(
            grid_display.rename(
                columns={
                    "age_group4": "Age group",
                    "edu_group": "Education",
                    "predicted_probability_pct": "Predicted %",
                    "ci_lower_pct": "CI lower %",
                    "ci_upper_pct": "CI upper %",
                }
            ),
            ["Predicted %", "CI lower %", "CI upper %"],
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Marginal Effects for the Age x Education Interaction")
    st.write(
        "These contrasts are reported in percentage points. They compare predicted probabilities rather than raw coefficients."
    )
    edu_col, age_col = st.columns(2)

    with edu_col:
        st.markdown(f"**Education effects within age {age_group}**")
        edu_chart = (
            alt.Chart(focused_education_effects)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X(
                    "education_display:N",
                    title="Education",
                    sort=[display_label(level) for level in EDU_ORDER if level != "Higher+"],
                    axis=horizontal_axis("Education"),
                ),
                y=alt.Y("difference_pct_points:Q", title="Difference vs Higher+ (pp)"),
                color=alt.condition(alt.datum.difference_pct_points >= 0, alt.value("#d95f02"), alt.value("#1b9e77")),
                tooltip=[
                    alt.Tooltip("comparison:N", title="Comparison"),
                    alt.Tooltip("difference_pct_points:Q", title="Difference (pp)", format=".1f"),
                    alt.Tooltip("ci_lower_pct_points:Q", title="95% CI lower", format=".1f"),
                    alt.Tooltip("ci_upper_pct_points:Q", title="95% CI upper", format=".1f"),
                ],
            )
            .properties(height=260)
        )
        st.altair_chart(edu_chart, use_container_width=True)

        edu_table = focused_education_effects.copy()
        edu_table["education_level"] = edu_table["education_level"].map(display_label)
        st.dataframe(
            format_percent_frame(
                edu_table.rename(
                    columns={
                        "education_level": "Education",
                        "comparison": "Comparison",
                        "predicted_probability_pct": "Predicted %",
                        "reference_probability_pct": "Higher+ reference %",
                        "difference_pct_points": "Difference (pp)",
                        "ci_lower_pct_points": "CI lower (pp)",
                        "ci_upper_pct_points": "CI upper (pp)",
                    }
                ),
                ["Predicted %", "Higher+ reference %", "Difference (pp)", "CI lower (pp)", "CI upper (pp)"],
            ),
            use_container_width=True,
            hide_index=True,
        )

    with age_col:
        st.markdown(f"**Age effects within {display_label(education)}**")
        age_chart = (
            alt.Chart(focused_age_effects)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("age_group4:N", title="Age group", sort=AGE4_ORDER[1:], axis=horizontal_axis("Age group")),
                y=alt.Y("difference_pct_points:Q", title=f"Difference vs {AGE4_ORDER[0]} (pp)"),
                color=alt.condition(alt.datum.difference_pct_points >= 0, alt.value("#d95f02"), alt.value("#1b9e77")),
                tooltip=[
                    alt.Tooltip("comparison:N", title="Comparison"),
                    alt.Tooltip("difference_pct_points:Q", title="Difference (pp)", format=".1f"),
                    alt.Tooltip("ci_lower_pct_points:Q", title="95% CI lower", format=".1f"),
                    alt.Tooltip("ci_upper_pct_points:Q", title="95% CI upper", format=".1f"),
                ],
            )
            .properties(height=260)
        )
        st.altair_chart(age_chart, use_container_width=True)

        st.dataframe(
            format_percent_frame(
                focused_age_effects.rename(
                    columns={
                        "age_group4": "Age group",
                        "comparison": "Comparison",
                        "predicted_probability_pct": "Predicted %",
                        "reference_probability_pct": f"{AGE4_ORDER[0]} reference %",
                        "difference_pct_points": "Difference (pp)",
                        "ci_lower_pct_points": "CI lower (pp)",
                        "ci_upper_pct_points": "CI upper (pp)",
                    }
                ),
                [f"{AGE4_ORDER[0]} reference %", "Predicted %", "Difference (pp)", "CI lower (pp)", "CI upper (pp)"],
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.download_button(
        "Download focused age x education grid as CSV",
        data=grid.to_csv(index=False).encode("utf-8"),
        file_name="smoking_age_education_grid.csv",
        mime="text/csv",
    )

    st.subheader("Other Pooled Multinomial Models")
    selected_model = st.selectbox(
        "Choose multinomial model",
        options=list(model_options.keys()),
        format_func=lambda key: str(model_options[key]["title"]),
    )
    mlogit_probs = multinomial_outputs[selected_model]
    st.write(str(model_options[selected_model]["sample_note"]))
    st.write("These bars show profile-specific predicted probabilities across the selected multinomial categories.")

    mlogit_chart = (
        alt.Chart(mlogit_probs)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(
                "category:N",
                title="Category",
                sort=list(mlogit_probs["category"]),
                axis=horizontal_axis("Category", label_limit=260),
            ),
            y=alt.Y("predicted_probability_pct:Q", title="Predicted probability (%)"),
            color=alt.Color("category:N", legend=None),
            tooltip=[
                alt.Tooltip("category:N", title="Category"),
                alt.Tooltip("predicted_probability_pct:Q", title="Predicted %", format=".1f"),
                alt.Tooltip("ci_lower_pct:Q", title="95% CI lower", format=".1f"),
                alt.Tooltip("ci_upper_pct:Q", title="95% CI upper", format=".1f"),
            ],
        )
        .properties(height=300)
    )
    st.altair_chart(mlogit_chart, use_container_width=True)

    st.dataframe(
        format_percent_frame(
            mlogit_probs.rename(
                columns={
                    "category": "Category",
                    "predicted_probability_pct": "Predicted %",
                    "ci_lower_pct": "CI lower %",
                    "ci_upper_pct": "CI upper %",
                }
            ),
            ["Predicted %", "CI lower %", "CI upper %"],
        ),
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()
