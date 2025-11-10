# app.py
import streamlit as st
import pandas as pd
import json
from pipeline import run_pipeline

st.set_page_config(page_title="LLM Evaluator Simulator", layout="wide")

st.title("LLM Evaluator Simulator")

# --- Control Panel ---
st.sidebar.header("Simulation Parameters")
use_simulation = st.sidebar.checkbox("Use Simulation Mode", value=True)
num_samples = st.sidebar.number_input("Samples per Variant", min_value=1, max_value=2000, value=50)
hallucination_rate = st.sidebar.slider("Hallucination Rate", 0.0, 1.0, 0.1)
variants = st.sidebar.text_input("Variants (comma separated)", "A,B")
variant_list = [v.strip() for v in variants.split(",")]

st.sidebar.header("Real LLM Mode Parameters")

st.sidebar.markdown(
    """
**Upload variant_test_items.csv** with the following columns:

- `item_id` : Unique identifier for each test item  
- `variant_id` : Variant label (e.g., A, B, C)  
- `prompt` : Text prompt to evaluate  
- `true_score` : True numeric score for reference (used to compute bias and hallucination)  

Ensure the CSV is properly formatted with a header row.
"""
)

uploaded_file = st.sidebar.file_uploader("Upload variant_test_items.csv for real LLM", type="csv")

# --- Run Pipeline ---
if st.button("Run Evaluation"):

    if not use_simulation and uploaded_file is None:
        st.warning("Please upload variant_test_items.csv for real LLM evaluation")
    else:
        df_items = pd.read_csv(uploaded_file) if uploaded_file else None

        evaluator_results, metrics = run_pipeline(
            variant_list=variant_list,
            num_samples=num_samples,
            hallucination_rate=hallucination_rate,
            use_simulation=use_simulation,
            df_items=df_items
        )

        st.success("Evaluation Completed!")

        # --- Load aggregated evaluator metrics ---
        with open("evaluator_bias.json", "r", encoding="utf-8") as f:
            metrics_json = json.load(f)
        df_profile = pd.DataFrame.from_dict(metrics_json, orient="index").reset_index()
        df_profile.rename(columns={"index": "Evaluator"}, inplace=True)

        st.subheader("Evaluator Profile Summary")
        st.dataframe(
            df_profile.style.set_properties(**{
                "text-align": "center",
                "border": "1px solid gray"
            }),
            height=200
        )

        # --- Display detailed hallucination per model ---
        st.subheader("Detailed Hallucination Metrics")
        for model, df_model in evaluator_results.items():
            st.markdown(f"**Model: {model}**")
            st.dataframe(
                df_model[["variant","item_id","score","true_score","hallucination_score","cost_usd"]].style.set_properties(**{
                    "text-align": "center",
                    "border": "1px solid gray"
                }),
                height=300
            )

        # --- Option to download CSVs ---
        st.subheader("Download Results")
        for model, df_model in evaluator_results.items():
            csv = df_model.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"Download {model} detailed results CSV",
                data=csv,
                file_name=f"{model}_hallucination_details.csv",
                mime="text/csv"
            )

        # Download evaluator profile
        csv_profile = df_profile.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download evaluator profile CSV",
            data=csv_profile,
            file_name="llm_evaluator_profile.csv",
            mime="text/csv"
        )
