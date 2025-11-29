import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

# ------------------------------------------------
# BASIC PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="FinOps at Scale for Serverless Applications",
    layout="wide"
)

st.title("FinOps at Scale for Serverless Applications – RetailNova")
st.caption("INFO49971 – Cloud Economics | Serverless Cost Analysis Activity")


# ------------------------------------------------
# SIDEBAR: UPLOAD DATASET
# ------------------------------------------------
st.sidebar.header("1. Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload Serverless_Data.csv",
    type=["csv"]
)

st.sidebar.markdown("---")
st.sidebar.caption("After uploading, explore each tab for the exercises.")


# ------------------------------------------------
# FIXED CSV LOADER (IMPORTANT)
# ------------------------------------------------
@st.cache_data
def load_data(file):
    import pandas as pd

    # Read raw file text
    raw = file.read().decode("utf-8", errors="ignore").strip()

    # 1. Remove ALL double quotes (your file has quotes wrapping entire rows)
    raw = raw.replace('"', '')

    # 2. Replace Windows line breaks (CRLF) with normal LF
    raw = raw.replace("\r", "\n")

    # 3. Split lines safely
    lines = [line for line in raw.split("\n") if line.strip()]

    # 4. First line = header
    header = lines[0].split(",")

    # 5. Remaining lines = data rows
    rows = [line.split(",") for line in lines[1:]]

    # 6. Build DataFrame
    df = pd.DataFrame(rows, columns=header)

    # 7. Convert numeric fields
    numeric_cols = [
        "InvocationsPerMonth","AvgDurationMs","MemoryMB",
        "ColdStartRate","ProvisionedConcurrency",
        "GBSeconds","DataTransferGB","CostUSD"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ------------------------------------------------
# LOAD CSV
# ------------------------------------------------
if not uploaded_file:
    st.warning("⬅ Please upload **Serverless_Data.csv** to begin.")
    st.stop()

df = load_data(uploaded_file)
st.success("✅ CSV Loaded Successfully!")


with st.expander("Preview Dataset (first 10 rows)"):
    st.dataframe(df.head(10), use_container_width=True)


# ------------------------------------------------
# KPIs
# ------------------------------------------------
total_cost = df["CostUSD"].sum()
total_inv = df["InvocationsPerMonth"].sum()
num_funcs = df["FunctionName"].nunique()

c1, c2, c3 = st.columns(3)
c1.metric("Total Monthly Cost", f"${total_cost:,.2f}")
c2.metric("Total Invocations", f"{int(total_inv):,}")
c3.metric("Number of Functions", num_funcs)


# ------------------------------------------------
# TABS
# ------------------------------------------------
tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Exercise 1",
    "Exercise 2",
    "Exercise 3",
    "Exercise 4",
    "Exercise 5",
    "Exercise 6"
])


# ------------------------------------------------
# OVERVIEW TAB
# ------------------------------------------------
with tab0:
    st.subheader("📊 Overview")

    overview_chart = alt.Chart(df).mark_circle(size=80).encode(
        x="InvocationsPerMonth",
        y="CostUSD",
        color="Environment",
        tooltip=["FunctionName", "Environment", "InvocationsPerMonth", "CostUSD"]
    ).interactive()

    st.altair_chart(overview_chart, use_container_width=True)


# ------------------------------------------------
# EXERCISE 1
# ------------------------------------------------
with tab1:
    st.subheader("Exercise 1: Top Cost Contributors")

    df_sorted = df.sort_values("CostUSD", ascending=False)
    df_sorted["CumCost"] = df_sorted["CostUSD"].cumsum()
    df_sorted["CumPct"] = (df_sorted["CumCost"] / df_sorted["CostUSD"].sum()) * 100

    top_80 = df_sorted[df_sorted["CumPct"] <= 80]

    st.write("### Functions contributing to the first **80%** of spend:")
    st.dataframe(top_80, use_container_width=True)

    scatter = alt.Chart(df).mark_circle(size=80).encode(
        x="InvocationsPerMonth",
        y="CostUSD",
        color="Environment",
        tooltip=["FunctionName","CostUSD","InvocationsPerMonth"]
    ).interactive()

    st.altair_chart(scatter, use_container_width=True)


# ------------------------------------------------
# EXERCISE 2
# ------------------------------------------------
with tab2:
    st.subheader("Exercise 2: Memory Right-Sizing")

    oversized = df[
        (df["AvgDurationMs"] < 200) &
        (df["MemoryMB"] > 1024)
    ]

    st.write("### High memory, short duration (over-provisioned)")
    st.dataframe(oversized, use_container_width=True)

    estimated_savings = oversized["CostUSD"].sum() * 0.40
    st.success(f"Estimated savings if downsized: ~${estimated_savings:,.2f}")


# ------------------------------------------------
# EXERCISE 3
# ------------------------------------------------
with tab3:
    st.subheader("Exercise 3: Provisioned Concurrency Optimization")

    pc_waste = df[
        (df["ColdStartRate"] < 5) &
        (df["ProvisionedConcurrency"] > 0)
    ]

    st.write("### Low cold-start, but using PC:")
    st.dataframe(pc_waste, use_container_width=True)


# ------------------------------------------------
# EXERCISE 4
# ------------------------------------------------
with tab4:
    st.subheader("Exercise 4: Detect Low-Value Workloads")

    total_invocations = df["InvocationsPerMonth"].sum()

    low_value = df[
        (df["InvocationsPerMonth"] < total_invocations * 0.01) &
        (df["CostUSD"] > df["CostUSD"].median())
    ]

    st.write("### Low invocation (<1%) but high cost:")
    st.dataframe(low_value, use_container_width=True)


# ------------------------------------------------
# EXERCISE 5
# ------------------------------------------------
with tab5:
    st.subheader("Exercise 5: Cost Forecasting Model")

    k = 0.0000000021
    egress = 0.09

    df["PredictedCost"] = (
        df["InvocationsPerMonth"] * (df["AvgDurationMs"]/1000) * (df["MemoryMB"]/1024) * k
        + df["DataTransferGB"] * egress
    )

    st.write("### Actual vs Predicted Cost:")
    st.dataframe(df[["FunctionName","CostUSD","PredictedCost"]], use_container_width=True)

    model_plot = alt.Chart(df).mark_circle(size=80).encode(
        x="CostUSD",
        y="PredictedCost",
        tooltip=["FunctionName","CostUSD","PredictedCost"]
    ).interactive()

    st.altair_chart(model_plot, use_container_width=True)


# ------------------------------------------------
# EXERCISE 6
# ------------------------------------------------
with tab6:
    st.subheader("Exercise 6: Containerization Candidates")

    containers = df[
        (df["AvgDurationMs"] > 3000) &
        (df["MemoryMB"] > 2048) &
        (df["InvocationsPerMonth"] < df["InvocationsPerMonth"].mean())
    ]

    st.write("### Better suited for ECS/Fargate:")
    st.dataframe(containers, use_container_width=True)

    st.info("Long-running + high memory + low invocation = ideal for containers.")
