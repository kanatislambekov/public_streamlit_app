from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from app_utils import (
    AGE4_ORDER,
    CHILD_ORDER,
    EDU_ORDER,
    HARMONIZED_REGION_ORDER,
    HOUSEHOLD_GROUP_ORDER,
    SETTLEMENT_ORDER,
    SEX_ORDER,
    YEAR_ORDER,
    age_marginal_effects,
    default_profile,
    display_label,
    education_marginal_effects,
    interaction_probability_grid,
    interaction_wald_test,
    multinomial_probabilities,
    multinomial_specs,
    predict_profile,
)


st.set_page_config(page_title="Smoking Probability Dashboard", layout="wide")


def format_percent_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[col] = out[col].map(lambda value: f"{value:.1f}")
    return out


def horizontal_axis(title: str, label_limit: int = 220) -> alt.Axis:
    return alt.Axis(title=title, labelAngle=0, labelLimit=label_limit)


def main() -> None:
    st.title("Smoking Probability Dashboard")
    st.caption(
        "Standalone public-safe app built from sanitized pooled-model artifacts for the 2020-2024 surveys. "
        "No respondent-level survey tables are included in this app package."
    )

    defaults = default_profile()
    wald = interaction_wald_test()

    with st.sidebar:
        st.header("Profile Inputs")
        survey_year = st.selectbox("Survey year", YEAR_ORDER, index=YEAR_ORDER.index(defaults["survey_year_factor"]))
        age_group = st.selectbox("Age group", AGE4_ORDER, index=AGE4_ORDER.index(defaults["age_group4"]))
        sex = st.selectbox("Sex", SEX_ORDER, index=SEX_ORDER.index(defaults["sex_factor"]))
        region = st.selectbox(
            "Region",
            HARMONIZED_REGION_ORDER,
            index=HARMONIZED_REGION_ORDER.index(defaults["region_harmonized"]),
        )
        education = st.selectbox(
            "Education",
            EDU_ORDER,
            index=EDU_ORDER.index(defaults["edu_group"]),
            format_func=display_label,
        )
        child_status = st.selectbox(
            "Child in household",
            CHILD_ORDER,
            index=CHILD_ORDER.index(defaults["children_u18_label"]),
            format_func=display_label,
        )
        household_group = st.selectbox(
            "Household size",
            HOUSEHOLD_GROUP_ORDER,
            index=HOUSEHOLD_GROUP_ORDER.index(defaults["household_group"]),
        )
        settlement = st.selectbox(
            "Settlement type",
            SETTLEMENT_ORDER,
            index=SETTLEMENT_ORDER.index(defaults["settlement_type"]),
        )

    profile = {
        "survey_year_factor": survey_year,
        "age_group4": age_group,
        "sex_factor": sex,
        "region_harmonized": region,
        "edu_group": education,
        "children_u18_label": child_status,
        "household_group": household_group,
        "settlement_type": settlement,
    }

    profile_prediction = predict_profile(profile)
    grid = interaction_probability_grid(profile)
    education_effects = education_marginal_effects(profile)
    age_effects = age_marginal_effects(profile)
    focused_education_effects = education_effects.loc[education_effects["age_group4"] == age_group].copy()
    focused_education_effects["education_display"] = focused_education_effects["education_level"].map(display_label)
    focused_age_effects = age_effects.loc[age_effects["edu_group"] == education].copy()

    st.subheader("Selected Profile")
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
        "Download age x education grid as CSV",
        data=grid.to_csv(index=False).encode("utf-8"),
        file_name="smoking_age_education_grid.csv",
        mime="text/csv",
    )

    st.subheader("Other Pooled Multinomial Models")
    model_options = multinomial_specs()
    selected_model = st.selectbox(
        "Choose multinomial model",
        options=list(model_options.keys()),
        format_func=lambda key: str(model_options[key]["title"]),
    )
    mlogit_probs = multinomial_probabilities(selected_model, profile)
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
