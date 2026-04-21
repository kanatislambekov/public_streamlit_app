from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from app_utils import (
    ALL_OPTION,
    AGE4_ORDER,
    CHILD_ORDER,
    EDU_ORDER,
    HARMONIZED_REGION_ORDER,
    HOUSEHOLD_GROUP_ORDER,
    PROFILE_OPTION_ORDERS,
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
    normalize_selection,
    predict_profile,
    selection_description,
)


st.set_page_config(page_title="Smoking Probability Dashboard", layout="wide")


def format_percent_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[col] = out[col].map(lambda value: f"{value:.1f}")
    return out


def horizontal_axis(title: str, label_limit: int = 220) -> alt.Axis:
    return alt.Axis(title=title, labelAngle=0, labelLimit=label_limit)


def selection_table_row(label: str, values: list[str], format_func=lambda value: value) -> dict[str, str]:
    rendered = [format_func(value) for value in values]
    display = ", ".join(rendered) if len(rendered) <= 3 else f"{', '.join(rendered[:3])}, ..."
    return {"Parameter": label, "Selected values": display, "Count": str(len(values))}


def selection_widget(
    label: str,
    column: str,
    default_value: str,
    format_func=lambda value: value,
) -> list[str]:
    options = PROFILE_OPTION_ORDERS[column]
    raw_selection = st.multiselect(
        label,
        options=[ALL_OPTION, *options],
        default=[default_value],
        format_func=lambda value: ALL_OPTION if value == ALL_OPTION else format_func(value),
    )
    return normalize_selection(raw_selection, options)


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
        survey_years = selection_widget("Survey year", "survey_year_factor", defaults["survey_year_factor"])
        age_groups = selection_widget("Age group", "age_group4", defaults["age_group4"])
        sexes = selection_widget("Sex", "sex_factor", defaults["sex_factor"])
        regions = selection_widget(
            "Region",
            "region_harmonized",
            defaults["region_harmonized"],
        )
        educations = selection_widget(
            "Education",
            "edu_group",
            defaults["edu_group"],
            format_func=display_label,
        )
        child_statuses = selection_widget(
            "Child in household",
            "children_u18_label",
            defaults["children_u18_label"],
            format_func=display_label,
        )
        household_groups = selection_widget(
            "Household size",
            "household_group",
            defaults["household_group"],
        )
        settlements = selection_widget(
            "Settlement type",
            "settlement_type",
            defaults["settlement_type"],
        )

    profile = {
        "survey_year_factor": survey_years,
        "age_group4": age_groups,
        "sex_factor": sexes,
        "region_harmonized": regions,
        "edu_group": educations,
        "children_u18_label": child_statuses,
        "household_group": household_groups,
        "settlement_type": settlements,
    }

    profile_prediction = predict_profile(profile)
    grid = interaction_probability_grid(profile)
    education_effects = education_marginal_effects(profile)
    age_effects = age_marginal_effects(profile)
    focused_education_effects = education_effects.loc[education_effects["age_group4"].isin(age_groups)].copy()
    focused_education_effects["education_display"] = focused_education_effects["education_level"].map(display_label)
    focused_age_effects = age_effects.loc[age_effects["edu_group"].isin(educations)].copy()
    focused_age_effects["education_display"] = focused_age_effects["edu_group"].map(display_label)

    st.subheader("Selected Profile")
    st.caption(
        "When you choose multiple values or `All`, the dashboard reports equal-weight averages across the "
        "corresponding model-based profile combinations. No direct survey records are shown or shared."
    )
    metric_col, ci_col, test_col = st.columns(3)
    metric_col.metric("Predicted probability of current smoking", f"{profile_prediction['probability_pct']:.1f}%")
    ci_col.metric(
        "95% confidence interval",
        f"{profile_prediction['ci_lower_pct']:.1f}% to {profile_prediction['ci_upper_pct']:.1f}%",
    )
    test_col.metric("Age x education joint test", f"p = {wald['p_value']:.3f}")

    st.write(
        f"For profiles spanning {selection_description(age_groups)}, {selection_description(regions)}, and "
        f"{selection_description(educations, display_label).lower()}, the model estimates a "
        f"{profile_prediction['probability_pct']:.1f}% probability of current smoking after averaging across "
        f"{profile_prediction['profiles_averaged']:,} selected profile combinations and holding the remaining "
        f"selected characteristics constant."
    )

    selection_rows = [
        selection_table_row("Survey year", survey_years),
        selection_table_row("Age group", age_groups),
        selection_table_row("Sex", sexes),
        selection_table_row("Region", regions),
        selection_table_row("Education", educations, display_label),
        selection_table_row("Child in household", child_statuses, display_label),
        selection_table_row("Household size", household_groups),
        selection_table_row("Settlement type", settlements),
    ]
    st.dataframe(pd.DataFrame(selection_rows), use_container_width=True, hide_index=True)

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
        if len(age_groups) == 1:
            st.markdown(f"**Education effects within age {age_groups[0]}**")
        else:
            st.markdown("**Education effects within selected age groups**")

        edu_chart_base = (
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
        )
        if len(age_groups) == 1:
            edu_chart = edu_chart_base.properties(height=260)
        else:
            edu_chart = edu_chart_base.encode(
                column=alt.Column("age_group4:N", title="Age group", sort=age_groups),
            ).properties(height=260)
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
        if len(educations) == 1:
            st.markdown(f"**Age effects within {display_label(educations[0])}**")
        else:
            st.markdown("**Age effects within selected education levels**")

        age_chart_base = (
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
        )
        if len(educations) == 1:
            age_chart = age_chart_base.properties(height=260)
        else:
            age_chart = age_chart_base.encode(
                column=alt.Column("education_display:N", title="Education", sort=[display_label(level) for level in educations]),
            ).properties(height=260)
        st.altair_chart(age_chart, use_container_width=True)

        st.dataframe(
            format_percent_frame(
                focused_age_effects.rename(
                    columns={
                        "age_group4": "Age group",
                        "education_display": "Education",
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
