<h1 align="center">Vibhuti Gandhi</h1>
<p align="center"><b>Data Scientist · ML Engineer · Applied AI</b></p>

<p align="center">
  Vancouver, BC &nbsp;|&nbsp;
  <a href="https://www.linkedin.com/in/vibhuti-gandhi/">LinkedIn</a> &nbsp;|&nbsp;
  <a href="https://github.com/GanVib18">GitHub</a> &nbsp;|&nbsp;
  <a href="mailto:gandhivibhuti1802@gmail.com">Email</a>
</p>

<p align="center">
Data Scientist specializing in behavioral modeling, Bayesian inference, and ML deployment at scale.
3+ years of experience across research and production, with a track record of shipping models that
directly move revenue. Comfortable across the full ML lifecycle — from experimentation and statistical
validation to Snowflake-integrated deployment and drift monitoring.
</p>

---

## Experience

### Data Scientist — British Columbia Lottery Corporation
*Jan 2025 – Present*

- Drove **$1M+ incremental revenue** by deploying a Markov-based segmentation model (PySpark, Dataiku) to personalize retention campaigns across **600K+ players**.
- Surfaced **$8.9M** in over-redemption across **~11K** flagged player-years by replacing a static 30% heuristic with a context-aware anomaly detection model using tiered Huber regression.
- Built a CLV model that lifted forecast accuracy by **20%**; findings presented directly to Directors of Marketing and Analytics, informing multi-million-dollar marketing budget decisions.
- Developed a Player Trajectory Management system (XGBoost, 0.85 R² CV) predicting behavioral shifts for **250K+ players**, with automated weekly sync of risk profiles from Snowflake to Salesforce CDP/CRM.
- Reduced model pipeline failures by **30%** through automated validation, statistical drift monitoring, and peer-review gates.

### Research Assistant — SFU MAGPIE Group
*Jan 2023 – Dec 2023*

- Modeled COVID-19 Omicron wave dynamics in Databricks using ensemble and clustering methods, uncovering distinct feature-importance shifts between the BA.1 and BA.2 variants.
- Achieved **91% directional accuracy** predicting Omicron wave size with a Random Forest model (leave-one-out CV) across 113 epidemiological and genomic features.
- Co-authored [research on COVID-19 variant dynamics](https://www.medrxiv.org/content/10.1101/2025.09.16.25335896v1), presented at the **31st International Dynamics & Evolution of Human Viruses Conference (2024)**.

### Data Analyst Co-op — BC Public Service, Ministry of Health
*Jan 2023 – Apr 2023*

- Cut physician shift-report generation from 2 days to 15 minutes by building Oracle-to-Power BI pipelines; dashboards adopted by 20+ health administrators.
- Improved reporting accuracy by **15%** through automated reconciliation checks developed with the policy team.

---

## Featured Projects

**[Bayesian Media Mix Modeling for Global Marketing Attribution](https://github.com/GanVib18/Bayesian-Media-Mix-Model)** &nbsp;·&nbsp; [Article](https://medium.com/@gandhivibhuti1802/building-a-bayesian-media-mix-model-from-scratch-af14a2e4485b)
`PyMC` `Bayesian Inference` `Hierarchical Modeling` `SciPy Optimization`
Full-stack MMM pipeline for a fictional outdoor apparel brand across five markets. Applied geometric adstock and Hill saturation transformations, fit a hierarchical Bayesian model to decompose revenue by channel, and ran a SciPy budget optimizer that identified an **8–15% weekly revenue uplift** opportunity via reallocation from Social to Paid Search.

**[Regime-Switching Portfolio Optimization with Hidden Markov Models](https://github.com/GanVib18/Regime-Switching-Asset-Allocation)** &nbsp;·&nbsp; [Article](https://medium.com/@gandhivibhuti1802/smarter-than-60-40-building-a-regime-switching-portfolio-with-machine-learning-b6c1172b84f5)
`Gaussian HMM` `Ledoit-Wolf Shrinkage` `Walk-Forward Validation` `Canadian ETFs`
Three-regime market detection system using a Gaussian HMM on 10 engineered features (realized volatility, momentum, credit spread proxy, yield curve slope), combined with mean-variance optimization. Achieved a **Sharpe ratio of 1.27 vs. 0.88** for a 60/40 benchmark, outperforming in 3 of 4 out-of-sample test periods (2015–2025).

**[StatQL: Statistically Validated Text-to-SQL Agent](https://github.com/GanVib18/StatQL_Agent/)** &nbsp;·&nbsp; [Article](https://medium.com/@gandhivibhuti1802/statistical-validation-in-llm-powered-analytics-agents-5e28d958653b)
`LangGraph` `DuckDB` `FAISS` `FastAPI`
Open-source Text-to-SQL agent with a dedicated statistical validation layer that computes confidence intervals, t-tests, and regressions on query results to reduce LLM hallucinations. FAISS + DuckDB semantic cache achieved a **93.3% warm hit rate** and **95.6% reduction in API latency** on a 540K-row retail dataset.

**[🏆 Hackathon Winner — Forecasting Canada's CPI During the Pandemic](https://github.com/Vancouver-Datajam/CPI/)** &nbsp;·&nbsp; [Video](https://www.youtube.com/watch?v=av6l6yLJ8q0)
`ARIMA` `Box-Jenkins` `Time Series`
Forecasted Canadian CPI during the pandemic with **92% accuracy** using ARIMA — **1st place** among 10 teams at Vancouver Datajam, with policy implications presented to a panel of industry judges.

<details>
<summary><b>More Projects</b></summary>
<br>

| Project | Stack | Highlights |
|---|---|---|
| [Explainable GNNs for Protein Classification](https://github.com/GanVib18/Protein_GNN_Project) | PointNet, GCN, GGS-NN | Explainability analysis for trustworthy geometric deep learning |
| [Diabetes Prediction: Classifiers & SVMs](https://github.com/GanVib18/Diabetes-Prediction) | LR, KNN, LDA, SVM (RBF) | Comprehensive model comparison on the Pima Indians dataset |
| [Dow Jones Time-Series Analysis](https://github.com/GanVib18/Analysis-of-Dow-Jones-Industrial-Average-Returns-using-ARIMA) | ARIMA, Box-Jenkins | Addressed volatility clustering, heteroskedasticity, mean reversion |
| [Bank Churn Analysis](https://github.com/GanVib18/Bank-Churn-Analysis/tree/main) | Azure Synapse, MySQL, Power BI | Interactive 4-year churn dashboard |
| [Vancouver Weather Forecast](https://www.kaggle.com/code/vibhutigandhi/vancouver-weather-forecast-using-neural-prophet/notebook) | Neural Prophet, PyTorch | 78% accuracy on 80 years of climate data |
| [Wikipedia Summarizer GPT App](https://github.com/GanVib18/Wikipedia-Summarizer-GPT-App) | RAG, Flan-T5 XXL, Streamlit | End-to-end NLP app hosted on Streamlit |
| [Multi-Agent TikTok Content Analysis](https://github.com/GanVib18/Tiktok-Analysis) | VADER, Empath, Gemini-2.5-flash, MLM | Custom AI-agent pipeline quantifying the "sell-out effect" across 2,500 videos |

</details>

---

## Skills

| Category | Skills |
|---|---|
| **Languages** | Python · R · SQL · MATLAB |
| **ML / AI** | PyTorch · Scikit-learn · XGBoost · PyMC · Probabilistic Modeling · Markov Chains · Time Series |
| **AI Engineering & MLOps** | RAG · LangGraph · LangChain · FAISS · FastAPI · Docker · AWS SageMaker · Azure ML · CI/CD · Git · Model Monitoring · Dataiku |
| **Data & Cloud** | Snowflake · Oracle · Databricks · PySpark · Power BI · A/B Testing |
| **Visualization** | Power BI · Plotly · Streamlit · Seaborn |

---

## Education

**BSc Data Science — Simon Fraser University** &nbsp;|&nbsp; *Sep 2020 – Dec 2025*
Relevant coursework: Statistical Learning, Bayesian Statistics, Time Series Analysis, Artificial Intelligence, Applied Multivariate Analysis, Database Systems, Linear Optimization, Sampling & Experimental Design

---

## Certifications

<details>
<summary><b>Anthropic</b></summary>

- [Claude API](https://verify.skilljar.com/c/unji2sw5mh9r)
- [Model Context Protocol](https://verify.skilljar.com/c/eny6dhc6zdeu)
- [Model Context Protocol: Advanced Topics](https://verify.skilljar.com/c/7zmoickp6ybf)
- [Agent Skills](https://verify.skilljar.com/c/xsx2rt5imdke)
- [Subagents](https://verify.skilljar.com/c/w9ppagekz23k)
- [AI Fluency](https://verify.skilljar.com/c/c3xzyw4by6au)

</details>

<details>
<summary><b>Dataiku</b></summary>

- [ML Practitioner](https://github.com/GanVib18/portfolio/blob/main/Documents/certificate-nbigh2m9xtet-1737492950.pdf)
- [MLOps Practitioner](https://github.com/GanVib18/portfolio/blob/main/Documents/document_2.pdf)
- [Generative AI Practitioner](https://github.com/GanVib18/portfolio/blob/main/Documents/certificate-xhdmexqosxoc-1737661517.pdf)
- [Advanced Designer](https://github.com/GanVib18/portfolio/blob/main/Documents/document_1.pdf)
- [Core Designer](https://github.com/GanVib18/portfolio/blob/main/Documents/certificate-7kdyftxrzfkn-1736803601.pdf)
- [Developer](https://github.com/GanVib18/portfolio/blob/main/Documents/certificate-kgogqocp8yda-1736977872.pdf)

</details>

<details>
<summary><b>Microsoft, Kaggle & Udemy</b></summary>

- Microsoft: [Azure AI Fundamentals](https://www.credly.com/badges/d446bfbd-0f65-4112-bcd6-9e31a09cfe74)
- Kaggle: [Intro to Deep Learning](https://www.kaggle.com/learn/certification/vibhutigandhi/intro-to-deep-learning) · [Computer Vision](https://www.kaggle.com/learn/certification/vibhutigandhi/computer-vision) · [Geospatial Analysis](https://www.kaggle.com/learn/certification/vibhutigandhi/geospatial-analysis)
- Udemy: [Machine Learning A-Z](https://www.udemy.com/certificate/UC-34b4e8aa-f18a-4be9-acaf-08fc674e4e01/) · [PySpark Essentials](https://www.udemy.com/certificate/UC-b3dc284a-c077-41db-a293-b659184e76b7/)

</details>

---

## Teaching & Mentorship

- **Workshop Lead** — [House Price Prediction](https://github.com/GanVib18/DSSS-Workshop-House-Prices): guided 20+ participants through MLR, regularization, and EDA (DSSS).
- **Mentor** — [Word Embeddings (ELMo & BERT)](https://github.com/Vancouver-Datajam/Word-Embeddings): coached participants on contextual vector representations applied to climate and disaster event analysis.

---

<p align="center"><i>Open to Data Scientist / ML Engineer roles across British Columbia.</i></p>
