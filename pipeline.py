import pandas as pd
import json
import random
import numpy as np
from openai import OpenAI

openai_api = ''
client = OpenAI(api_key=openai_api)


# --- Simulation function ---
def simulate_llm_evaluations(model_name, variant_list, num_samples=50, hallucination_rate=0.1):
    records = []
    for variant in variant_list:
        for i in range(num_samples):
            score = round(random.uniform(1,5),2)
            hallucinated = random.random() < hallucination_rate

            if hallucinated:
                response = f"Variant {variant} is perfect 🚀"
                true_score = round(random.uniform(1,5),2)
            else:
                response = f"Variant {variant} performs as expected"
                true_score = score

            # Simulated token usage
            prompt_tokens = random.randint(5,15)
            completion_tokens = random.randint(5,20)
            total_tokens = prompt_tokens + completion_tokens
            cost_usd = total_tokens/1000 * 0.01

            records.append({
                "model": model_name,
                "variant": variant,
                "item_id": i,
                "score": score,
                "true_score": true_score,
                "response": response,
                "hallucination": hallucinated,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost_usd": cost_usd
            })
    return pd.DataFrame(records)


# --- Real LLM evaluation ---
def llm_evaluation(model_name, df_items, cost_per_1k_tokens=0.01):
    records = []
    for _, row in df_items.iterrows():
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role":"user","content":f"Rate this from 1-5: {row.prompt}"}]
        )

        # extract score safely
        try:
            score = float(response.choices[0].message.content.strip())
        except:
            score = 2.5  # fallback neutral

        # cost
        usage = response.usage
        cost_data = {
            "prompt_tokens": getattr(usage,"prompt_tokens",0),
            "completion_tokens": getattr(usage,"completion_tokens",0),
        }
        cost_data["total_tokens"] = cost_data["prompt_tokens"] + cost_data["completion_tokens"]
        cost_data["cost_usd"] = cost_data["total_tokens"]/1000 * cost_per_1k_tokens

        # hallucination
        hallucination = abs(score - row.true_score) > 0.5

        records.append({
            "model": model_name,
            "variant": row.variant_id,
            "item_id": row.item_id,
            "score": score,
            "true_score": row.true_score,
            "response": response.choices[0].message.content,
            "hallucination": hallucination,
            **cost_data
        })

    return pd.DataFrame(records)


# --- Main pipeline ---
def run_pipeline(variant_list=["A","B"],
                 models=["gpt-4.1-mini","gpt-4o-mini"],
                 num_samples=50,
                 hallucination_rate=0.1,
                 use_simulation=True,
                 df_items=None):
    
    evaluator_results = {}
    
    for model in models:
        if use_simulation:
            df_model = simulate_llm_evaluations(model, variant_list, num_samples, hallucination_rate)
        else:
            if df_items is None:
                raise ValueError("df_items must be provided when not using simulation")
            df_model = llm_evaluation(model, df_items)
        
        df_model["hallucination_score"] = np.abs(df_model["score"] - df_model["true_score"])
        df_model.to_json(f"{model}_hallucination_details.json", orient="records", indent=2, force_ascii=False)
        evaluator_results[model] = df_model

    # aggregate metrics
    metrics = {}
    for model, df_model in evaluator_results.items():
        metrics[model] = {
            "avg_bias": float((df_model["score"] - df_model["true_score"]).mean()),
            "bias_std": float((df_model["score"] - df_model["true_score"]).std()),
            "avg_hallucination": float(df_model["hallucination_score"].mean()),
            "hallucination_rate": float(df_model["hallucination"].mean()),
            "avg_cost_usd": float(df_model["cost_usd"].mean())
        }

    with open("evaluator_bias.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    df_profile = pd.DataFrame.from_dict(metrics, orient="index").reset_index()
    df_profile.rename(columns={"index": "evaluator_name"}, inplace=True)

    # Save CSV for app.py / simulator
    df_profile.to_csv("llm_evaluator_profile.csv", index=False)
    return evaluator_results, metrics


# --- Run example ---
if __name__ == "__main__":

    run_pipeline()

    ## Example with real LLM
    # df_items = pd.read_csv("variant_test_items.csv")
    # run_pipeline(use_simulation=False, df_items=df_items)
