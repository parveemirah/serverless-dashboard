import streamlit as st
import pandas as pd
import numpy as np
import io
import altair as alt

st.set_page_config(
    page_title="FinOps at Scale for Serverless Applications",
    layout="wide"
)

st.title("FinOps at Scale for Serverless Applications – RetailNova")
st.caption("INFO49971 – Cloud Economics | Serverless Cost Analysis Activity")

st.sidebar.header("1. Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload Serverless_Data.csv",
    type=["csv"]
)

@st.cache_data
def load_data(file):
    text = file.read().decode("utf-8")
    # Remove the extra quotes around each line
    text = text.replace('"', "")
    df = pd.read_csv(io.StringIO(text))
    # Ensure numeric types
    num_cols = [
        "InvocationsPerMonth",
        "AvgDurationMs",
        "MemoryMB",
        "ColdStartRate",
        "ProvisionedConcurrency",
        "GBSeconds",
        "DataTransferGB",
        "CostUSD",
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col])
    return df

if uploaded_file is None:
    st.info("Please upload `Serverless_Data.csv` in the sidebar to begin.")
else:
    df = load_data(uploaded_file)

    st.subheader("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Functions", len(df))
    with c2:
        st.metric("Total Monthly Cost (USD)", f"${df['CostUSD'].sum():.2f}")
    with c3:
        st.metric("Total Invocations / Month", f"{df['InvocationsPerMonth'].sum():,}")
    with c4:
        st.metric("Environments", ", ".join(sorted(df["Environment"].unique())))

    st.dataframe(df.head(), use_container_width=True)

    # ===== Tabs for each exercise =====
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Exercise 1: Top Cost Contributors",
        "Exercise 2: Memory Right-Sizing",
        "Exercise 3: Provisioned Concurrency",
        "Exercise 4: Low-Value Workloads",
        "Exercise 5: Cost Forecasting",
        "Exercise 6: Containerization Candidates",
    ])

    # ----------------- Exercise 1 -----------------
    with tab1:
        st.header("Exercise 1 – Identify Top Cost Contributors")

        df_ex1 = df.copy()
        total_cost = df_ex1["CostUSD"].sum()
        df_ex1["CostShare"] = df_ex1["CostUSD"] / total_cost
        df_ex1 = df_ex1.sort_values("CostUSD", ascending=False)
        df_ex1["CumCostShare"] = df_ex1["CostShare"].cumsum()

        # 80% cutoff
        top_80 = df_ex1[df_ex1["CumCostShare"] <= 0.8]
        if len(top_80) < len(df_ex1):
            top_80 = df_ex1.iloc[: len(top_80) + 1]

        st.subheader("Functions Contributing ~80% of Cost")
        st.write(f"Number of functions needed to reach ~80% of spend: **{len(top_80)}**")
        st.dataframe(
            top_80[
                [
                    "FunctionName",
                    "Environment",
                    "CostUSD",
                    "CostShare",
                    "CumCostShare",
                ]
            ],
            use_container_width=True,
        )

        st.subheader("Cost vs Invocations")
        chart = (
            alt.Chart(df_ex1)
            .mark_circle(size=60)
            .encode(
                x=alt.X("InvocationsPerMonth:Q", title="Invocations per Month"),
                y=alt.Y("CostUSD:Q", title="Monthly Cost (USD)"),
                color=alt.Color("Environment:N"),
                tooltip=[
                    "FunctionName",
                    "Environment",
                    "InvocationsPerMonth",
                    "CostUSD",
                ],
            )
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)

    # ----------------- Exercise 2 -----------------
    with tab2:
        st.header("Exercise 2 – Memory Right-Sizing")

        st.markdown(
            "- Identify functions with **low duration** but **high memory**.\n"
            "- Here we flag `AvgDurationMs < 1000` and `MemoryMB ≥ 1024`."
        )

        df_ex2 = df[
            (df["AvgDurationMs"] < 1000) & (df["MemoryMB"] >= 1024)
        ].copy()

        def suggested_memory(mb: int) -> int:
            if mb >= 4096:
                return 3072
            if mb >= 3072:
                return 2048
            if mb >= 2048:
                return 1024
            if mb >= 1024:
                return 512
            return mb

        if df_ex2.empty:
            st.info("No functions match the right-sizing heuristic with the current thresholds.")
        else:
            df_ex2["SuggestedMemoryMB"] = df_ex2["MemoryMB"].apply(suggested_memory)
            df_ex2["NewCostEst"] = df_ex2["CostUSD"] * (
                df_ex2["SuggestedMemoryMB"] / df_ex2["MemoryMB"]
            )
            df_ex2["MonthlySavingsEst"] = df_ex2["CostUSD"] - df_ex2["NewCostEst"]

            total_savings = df_ex2["MonthlySavingsEst"].sum()

            st.metric(
                "Estimated Monthly Savings from Right-Sizing",
                f"${total_savings:.2f}",
            )

            st.dataframe(
                df_ex2[
                    [
                        "FunctionName",
                        "Environment",
                        "InvocationsPerMonth",
                        "AvgDurationMs",
                        "MemoryMB",
                        "SuggestedMemoryMB",
                        "CostUSD",
                        "NewCostEst",
                        "MonthlySavingsEst",
                    ]
                ],
                use_container_width=True,
            )

    # ----------------- Exercise 3 -----------------
    with tab3:
        st.header("Exercise 3 – Provisioned Concurrency Optimization")

        df_ex3 = df[df["ProvisionedConcurrency"] > 0].copy()

        def pc_recommendation(row):
            if row["ProvisionedConcurrency"] == 0:
                return "No PC"
            # Low cold starts & moderate volume -> can likely reduce/remove
            if row["ColdStartRate"] < 0.01 and row["InvocationsPerMonth"] < 1_000_000:
                return "Remove PC"
            if row["ColdStartRate"] < 0.02 and row["ProvisionedConcurrency"] >= 3:
                return "Reduce PC"
            return "Keep PC"

        if df_ex3.empty:
            st.info("No functions are using Provisioned Concurrency.")
        else:
            df_ex3["PC_Recommendation"] = df_ex3.apply(pc_recommendation, axis=1)
            st.dataframe(
                df_ex3[
                    [
                        "FunctionName",
                        "Environment",
                        "InvocationsPerMonth",
                        "ColdStartRate",
                        "ProvisionedConcurrency",
                        "CostUSD",
                        "PC_Recommendation",
                    ]
                ],
                use_container_width=True,
            )

    # ----------------- Exercise 4 -----------------
    with tab4:
        st.header("Exercise 4 – Detect Unused / Low-Value Workloads")

        total_inv = df["InvocationsPerMonth"].sum()
        threshold = 0.01 * total_inv  # 1% of total invocations
        cost_q3 = df["CostUSD"].quantile(0.75)  # top 25% cost

        df_ex4 = df[
            (df["InvocationsPerMonth"] < threshold) & (df["CostUSD"] > cost_q3)
        ].copy()

        st.write(
            f"1% of total invocations ≈ **{threshold:,.0f}**. "
            f"High-cost threshold (75th percentile) ≈ **${cost_q3:.2f}**."
        )

        if df_ex4.empty:
            st.info("No low-volume, high-cost functions detected with the current thresholds.")
        else:
            st.dataframe(
                df_ex4[
                    [
                        "FunctionName",
                        "Environment",
                        "InvocationsPerMonth",
                        "AvgDurationMs",
                        "MemoryMB",
                        "CostUSD",
                    ]
                ],
                use_container_width=True,
            )

    # ----------------- Exercise 5 -----------------
    with tab5:
        st.header("Exercise 5 – Cost Forecasting Model")

        compute_work = (
            df["InvocationsPerMonth"]
            * df["AvgDurationMs"]
            * df["MemoryMB"]
        ).astype(float)

        X = np.vstack(
            [compute_work.values, df["DataTransferGB"].values, np.ones(len(df))]
        ).T
        y = df["CostUSD"].values

        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        alpha, beta, gamma = coeffs

        st.markdown("**Model:**")
        st.latex(
            r"\text{CostUSD} \approx \alpha \cdot "
            r"(\text{Invocations} \times \text{Duration} \times \text{Memory}) "
            r"+ \beta \cdot \text{DataTransferGB} + \gamma"
        )

        st.write(
            f"Estimated coefficients from dataset:\n\n"
            f"- α (compute coefficient): **{alpha:.2e}**  \n"
            f"- β (data transfer coefficient): **{beta:.2f}**  \n"
            f"- γ (base cost): **{gamma:.2f}**"
        )

        df_ex5 = df.copy()
        df_ex5["ComputeWork"] = compute_work
        df_ex5["ForecastCost"] = (
            alpha * df_ex5["ComputeWork"]
            + beta * df_ex5["DataTransferGB"]
            + gamma
        )

        st.subheader("Actual vs Forecast Cost")
        chart5 = (
            alt.Chart(df_ex5)
            .mark_circle(size=60)
            .encode(
                x=alt.X("CostUSD:Q", title="Actual Cost (USD)"),
                y=alt.Y("ForecastCost:Q", title="Forecast Cost (USD)"),
                tooltip=["FunctionName", "CostUSD", "ForecastCost"],
            )
            .interactive()
        )
        st.altair_chart(chart5, use_container_width=True)

        st.subheader("What-if Calculator")
        func_choice = st.selectbox(
            "Select a function to simulate changes:",
            df_ex5["FunctionName"].tolist(),
        )
        row = df_ex5[df_ex5["FunctionName"] == func_choice].iloc[0]

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            inv_change = st.slider(
                "Invocation Change (%)", -50, 200, 0, step=10
            )
        with col_b:
            dur_change = st.slider(
                "Duration Change (%)", -50, 200, 0, step=10
            )
        with col_c:
            mem_change = st.slider(
                "Memory Change (%)", -50, 200, 0, step=10
            )

        inv_new = row["InvocationsPerMonth"] * (1 + inv_change / 100)
        dur_new = row["AvgDurationMs"] * (1 + dur_change / 100)
        mem_new = row["MemoryMB"] * (1 + mem_change / 100)

        compute_new = inv_new * dur_new * mem_new
        forecast_new = alpha * compute_new + beta * row["DataTransferGB"] + gamma

        st.write(f"**Function:** {func_choice}")
        st.write(f"- Current Actual Cost: **${row['CostUSD']:.2f}**")
        st.write(f"- Model Forecast (current): **${row['ForecastCost']:.2f}**")
        st.write(f"- Model Forecast (after changes): **${forecast_new:.2f}**")

    # ----------------- Exercise 6 -----------------
    with tab6:
        st.header("Exercise 6 – Containerization Candidates")

        total_inv = df["InvocationsPerMonth"].sum()
        inv_threshold = 0.01 * total_inv

        df_ex6 = df[
            (df["AvgDurationMs"] > 3000)
            & (df["MemoryMB"] > 2048)
            & (df["InvocationsPerMonth"] < inv_threshold)
        ].copy()

        st.write(
            "Criteria: AvgDurationMs > 3000 ms, MemoryMB > 2048 MB, "
            "and invocations < 1% of total."
        )

        if df_ex6.empty:
            st.info("No functions match the containerization criteria with current thresholds.")
        else:
            st.dataframe(
                df_ex6[
                    [
                        "FunctionName",
                        "Environment",
                        "InvocationsPerMonth",
                        "AvgDurationMs",
                        "MemoryMB",
                        "CostUSD",
                    ]
                ],
                use_container_width=True,
            )
