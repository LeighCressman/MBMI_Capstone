## Detecting Asymptomatic Bacteriuria in Clinical Notes using Large Language Models (LLMs)
This repository contains the code for Leigh Cressman's capstone project in the Master of Biomedical Informatics (MBMI) program at the University of Pennsylvania (crel@pennmedicine.upenn.edu). 

Included in this repository are: 
1) Script for generating synthetic data
2) Prompt templates for urinary tract infection (UTI) symptoms
3) Code for calculating the cost to use large language models (LLMs) to process clinical notes
4) Code for preprocessing clinical notes
5) Code for hitting model endpoints, evaluating model performance, and tracking experiments in MLflow

## Note on LLM Inference Infrastructure
This project used Databricks Model Serving endpoints on the Penn Medicine Databricks platform for performing LLM inference. Said platform is a HIPAA-compliant, institutionally managed environment. No patient notes were sent to external OpenAI APIs. The model endpoints used in the study (e.g., "databricks-gpt-5-mini") are Databricks-hosted serving endpoints that use OpenAI models which are part of Penn Medicine's Azure environment.

The code is set up so that you can use your own API key to run OpenAI API on synthetic patient notes.

## Usage
Run the Python scripts in this order:
1) generate_synthetic_notes.py
2) pre_process_notes.py
3) calculate_cost.py (optional)
4) run_inference.py

## Software
LLM inference: Databricks Model Serving on the Penn Medicine Databricks platform

## License
GPL-3.0 license

## Acknowledgements 
This repo uses code from https://github.com/ugurcanvurgun/sdoh-llm-clinical-notes. Additionally, Sy Hwang, PhD student at University of Pennsylvania, contributed to this project's code.



