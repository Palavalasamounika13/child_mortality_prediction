"""
Child Mortality Dashboard & Predictor
======================================
A Streamlit app built around the EDA and modeling work done in
childmortality.ipynb. It lets you explore under-5 mortality rates
(deaths per 1,000 live births) by country and region, and generate
forward-looking predictions using the trained Linear Regression
models that were saved from the notebook.

Run with:
    streamlit run app.py
"""

import pickle

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Child Mortality Dashboard",
    page_icon="📉",
    layout="wide",
)

YEAR_COLS_COUNTRY = [f"{y}.5" for y in range(1985, 2021)]   # 1985.5 - 2020.5 (36 cols) -> predicts 2021.5
YEAR_COLS_REGION = [f"{y}.5" for y in range(1990, 2021)]    # 1990.5 - 2020.5 (31 cols) -> predicts 2021.5
TARGET_COL = "2021.5"


# --------------------------------------------------------------------------
# Cached data / model loaders
# --------------------------------------------------------------------------
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="latin-1")


@st.cache_resource
def load_country_model():
    """Linear Regression trained on country-medium.csv (36 yearly features)."""
    with open("lr_model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_region_model():
    """Linear Regression trained on SDG-Medium.csv (31 yearly features)."""
    return joblib.load("modeling_Tomorrow.joblib")


@st.cache_data
def project_future(df: pd.DataFrame, name_col: str, year_cols: list, _model,
                    years_to_predict, decay=0.95):
    """Recreate the notebook's forward-projection logic: predict 2021.5 with
    the trained model, then apply a yearly decay factor for future years so
    the trend continues to decline smoothly."""
    X = df[year_cols]
    base_year = float(years_to_predict[0])
    base_pred = _model.predict(X)

    out = df[[name_col]].copy()
    for year in years_to_predict:
        year_f = float(year)
        decay_factor = decay ** (year_f - base_year)
        out[year] = base_pred * decay_factor
    return out


# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------
st.sidebar.title("📉 Child Mortality")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Country Explorer", "Region Explorer", "Predict the Future"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Data: under-5 mortality rate (deaths per 1,000 live births), "
    "1985.5–2021.5, by country / region / SDG grouping."
)

# --------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------
country_df = load_csv("country-medium.csv")
region_df = load_csv("Regionss-medium.csv")
sdg_df = load_csv("SDG-Medium.csv")
world_df = load_csv("world-medium.csv")

# ==========================================================================
# PAGE: Overview
# ==========================================================================
if page == "Overview":
    st.title("Global Under-5 Child Mortality Overview")
    st.write(
        "Explore how under-5 mortality rates have changed worldwide, "
        "which countries are doing best and worst, and where the trend "
        "is heading."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Countries tracked", country_df["Country.Name"].nunique())
    col2.metric("Regions tracked", region_df["Region.Name"].nunique())
    col3.metric(
        "Global median rate (2021.5)",
        f"{world_df.loc[world_df['Year'] == 2021.5, 'Median'].values[0]:.1f}"
        if (world_df["Year"] == 2021.5).any() else "N/A",
    )

    st.subheader("World Median Trend (Upper / Median / Lower)")
    fig_world = px.line(
        world_df,
        x="Year",
        y=["Upper", "Median", "Lower"],
        labels={"value": "Mortality rate (per 1,000 births)", "variable": "Band"},
        title="World Under-5 Mortality Rate Over Time",
    )
    st.plotly_chart(fig_world, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top 10 Highest Mortality (2021.5)")
        top10 = country_df.nlargest(10, TARGET_COL)
        fig_high = px.bar(
            top10, x="Country.Name", y=TARGET_COL, color=TARGET_COL,
            color_continuous_scale="Reds",
        )
        fig_high.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_high, use_container_width=True)

    with c2:
        st.subheader("Top 10 Lowest Mortality (2021.5)")
        low10 = country_df.nsmallest(10, TARGET_COL)
        fig_low = px.bar(
            low10, x="Country.Name", y=TARGET_COL, color=TARGET_COL,
            color_continuous_scale="Greens",
        )
        fig_low.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_low, use_container_width=True)

    st.subheader("World Map — 2021.5 Under-5 Mortality Rate")
    fig_map = px.choropleth(
        country_df,
        locations="Country.Name",
        locationmode="country names",
        color=TARGET_COL,
        color_continuous_scale="Viridis",
        title="Under-5 Mortality Rate by Country (2021.5)",
    )
    st.plotly_chart(fig_map, use_container_width=True)

# ==========================================================================
# PAGE: Country Explorer
# ==========================================================================
elif page == "Country Explorer":
    st.title("Country Explorer")

    countries = sorted(country_df["Country.Name"].unique())
    default_idx = countries.index("Afghanistan") if "Afghanistan" in countries else 0
    pick = st.multiselect(
        "Choose one or more countries", countries, default=[countries[default_idx]]
    )

    if pick:
        sub = country_df[country_df["Country.Name"].isin(pick)]
        melted = sub.melt(
            id_vars=["Country.Name"],
            value_vars=[c for c in country_df.columns if c.replace(".", "", 1).replace("5", "").isdigit() or c.endswith(".5")],
            var_name="Year",
            value_name="Mortality Rate",
        )
        melted["Year"] = melted["Year"].astype(float)
        fig = px.line(
            melted, x="Year", y="Mortality Rate", color="Country.Name",
            title="Under-5 Mortality Rate Over Time",
            markers=True,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(sub.set_index("Country.Name"), use_container_width=True)
    else:
        st.info("Select at least one country to see its trend.")

# ==========================================================================
# PAGE: Region Explorer
# ==========================================================================
elif page == "Region Explorer":
    st.title("Region & SDG Group Explorer")

    tab1, tab2 = st.tabs(["Regions", "SDG Groupings"])

    with tab1:
        regions = sorted(region_df["Region.Name"].unique())
        pick_r = st.multiselect("Choose region(s)", regions, default=regions[:3], key="regions")
        if pick_r:
            sub = region_df[region_df["Region.Name"].isin(pick_r)]
            melted = sub.melt(
                id_vars=["Region.Name"],
                value_vars=[c for c in region_df.columns if c.endswith(".5")],
                var_name="Year", value_name="Mortality Rate",
            )
            melted["Year"] = melted["Year"].astype(float)
            fig = px.line(melted, x="Year", y="Mortality Rate", color="Region.Name", markers=True)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(sub.set_index("Region.Name"), use_container_width=True)

    with tab2:
        sdgs = sorted(sdg_df["Region.Name"].unique())
        pick_s = st.multiselect("Choose SDG group(s)", sdgs, default=sdgs[:3], key="sdgs")
        if pick_s:
            sub = sdg_df[sdg_df["Region.Name"].isin(pick_s)]
            melted = sub.melt(
                id_vars=["Region.Name"],
                value_vars=[c for c in sdg_df.columns if c.endswith(".5")],
                var_name="Year", value_name="Mortality Rate",
            )
            melted["Year"] = melted["Year"].astype(float)
            fig = px.line(melted, x="Year", y="Mortality Rate", color="Region.Name", markers=True)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(sub.set_index("Region.Name"), use_container_width=True)

# ==========================================================================
# PAGE: Predict the Future
# ==========================================================================
elif page == "Predict the Future":
    st.title("Predict Future Mortality Rates")
    st.write(
        "Uses the trained Linear Regression models from the notebook: "
        "`lr_model.pkl` (trained on country-level data, 1985.5–2020.5) and "
        "`modeling_Tomorrow.joblib` (trained on regional/SDG data, "
        "1990.5–2020.5) to estimate mortality for 2021.5 and project a few "
        "years beyond it using a gradual decay trend."
    )

    level = st.radio("Predict at which level?", ["Country", "Region", "SDG group"], horizontal=True)
    years_ahead = st.slider("Years beyond 2021.5 to project", 0, 10, 4)
    decay = st.slider(
        "Annual decline factor (lower = faster improvement)", 0.80, 1.00, 0.95, 0.01
    )
    years_to_predict = [str(2021.5 + i) for i in range(years_ahead + 1)]

    if level == "Country":
        model = load_country_model()
        df = country_df
        name_col = "Country.Name"
        year_cols = YEAR_COLS_COUNTRY
    elif level == "Region":
        model = load_region_model()
        df = region_df
        name_col = "Region.Name"
        year_cols = YEAR_COLS_REGION
    else:
        model = load_region_model()
        df = sdg_df
        name_col = "Region.Name"
        year_cols = YEAR_COLS_REGION

    missing = [c for c in year_cols if c not in df.columns]
    if missing:
        st.error(f"Selected dataset is missing expected columns: {missing}")
    else:
        result = project_future(df, name_col, year_cols, model, years_to_predict, decay)

        entities = sorted(result[name_col].unique())
        pick = st.multiselect(f"Choose {level.lower()}(s) to view", entities, default=entities[:3])

        if pick:
            sub = result[result[name_col].isin(pick)]
            melted = sub.melt(
                id_vars=[name_col], value_vars=years_to_predict,
                var_name="Year", value_name="Predicted Mortality Rate",
            )
            melted["Year"] = melted["Year"].astype(float)
            fig = px.line(
                melted, x="Year", y="Predicted Mortality Rate", color=name_col,
                markers=True, title=f"Predicted Under-5 Mortality Rate — {level}",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(sub.set_index(name_col), use_container_width=True)
        else:
            st.info(f"Select at least one {level.lower()} to see predictions.")

        st.caption(
            "Note: predictions beyond 2021.5 are not direct model outputs — "
            "they apply a fixed annual decay factor to the model's 2021.5 "
            "estimate, matching the approach used in the original notebook."
        )