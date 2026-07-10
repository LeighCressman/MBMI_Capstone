## Detecting Asymptomatic Bacteriuria in Clinical Notes using Large Language Models (LLMs)
This repository is the methods and code companion for my capstone project for a master's in Biomedical Informatics at the University of Pennsylvania. 

Contained in this repository are 1) prompt templates for urinary tract infection (UTI) symptoms; 2) code for calculating the cost to use large language models (LLMs) to process clinical notes; 3) code for preprocessing clinical notes, hitting model endpoints, evaluating model performance, and tracking experiments in Mlflow; 4) synthetic data for testing code; and 5) documentation for adapting code to other studies.

## Note on LLM Inference Infrastructure
This project uses Databricks Model Serving endpoints on the Penn Medicine Databricks platform for performing LLM inference. Said platform is a HIPAA-compliant, institutionally managed environment. No patient notes were sent to external OpenAI APIs. The model endpoints referenced (e.g., "databricks-gpt-5-mini") are Databricks-hosted serving endpoints that use OpenAI models which are part of Penn Medicine's Azure environment.

