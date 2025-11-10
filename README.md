# 🧪 GenAI Evaluator Experiment Simulator

> A Streamlit app for assessing how reliable LLMs are as judges in A/B product experiments.

Companies increasingly use LLMs to score product outcomes — content quality, UX signals, feature success, etc.  
But LLMs **hallucinate**, **cost money**, and **carry bias** that can distort experiment decisions.

This simulator helps product & AI teams **test evaluator reliability before launch**.

---

## ✅ What This Simulator Measures

| Category | Questions It Answers |
|---------|---------------------|
|️⃣ Evaluator Bias | Does the LLM favor Variant A or B? By how much? |
|️⃣ Hallucination Rate | How often does it make up incorrect or unsafe claims? |
|️⃣ Cost Modeling | How much would it cost to scale evaluations? |
|️⃣ Agreement vs Ground Truth | How closely does it match real outcomes? |
|️⃣ Trust Score | Should we use this evaluator for product decisions? |

---

## 🎯 Key Use Cases

✔ Replace slow human QA evaluators  
✔ Compare multiple LLMs as judges  
✔ Understand risk before product rollout  
✔ Optimize experiment strategy and cost  
✔ Communicate GenAI trustworthiness to stakeholders

---

## 🧩 Features at a Glance

- Monte-Carlo A/B experiment simulation  
- Upload or generate LLM evaluation scores
- Cost + hallucination weighted metrics
- Dynamic visualizations:
  - Score distributions
  - Bias impact
  - Hallucination & cost tradeoffs

---

## 📦 Getting Started

### Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/llm-evaluator-simulator.git
cd llm-evaluator-simulator
python3 -m venv .venv
source .venv/bin/activate   # Mac/Linux
# .venv\Scripts\activate    # Windows
pip install -r requirements.txt
streamlit run app.py

/project
 ├── app.py                        # Streamlit web app
 ├── pipeline.py                   # Generates evaluator scores/bias/hallucinations
 ├── llm_evaluator_profile.csv     # Model scoring + cost data
 ├── evaluator_bias.json           # Bias by model & variant
 ├── hallucination_details.json    # Error samples
 ├── requirements.txt
 └── README.md
