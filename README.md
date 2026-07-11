## Detecting Asymptomatic Bacteriuria in Clinical Notes using Large Language Models (LLMs)
This repository contains the methods and code for Leigh Cressman's capstone project in the Master of Biomedical Informatics (MBMI) program at the University of Pennsylvania (crel@pennmedicine.upenn.edu). 

Included in this repository are: 
1) Prompt templates for urinary tract infection (UTI) symptoms
2) Synthetic data for testing code
3) Code for calculating the cost to use large language models (LLMs) to process clinical notes
4) Code for preprocessing clinical notes, hitting model endpoints, evaluating model performance, and tracking experiments in Mlflow
6) Documentation for adapting code to other studies

## Note on LLM Inference Infrastructure
This project uses Databricks Model Serving endpoints on the Penn Medicine Databricks platform for performing LLM inference. Said platform is a HIPAA-compliant, institutionally managed environment. No patient notes were sent to external OpenAI APIs. The model endpoints referenced (e.g., "databricks-gpt-5-mini") are Databricks-hosted serving endpoints that use OpenAI models which are part of Penn Medicine's Azure environment.

## Repository Contents
This 

## Study Overview

## Installation

## Usage

## Prompt Engineering Approach

## Evaluation Metrics

## Error Taxonomy 

## Software
LLM inference: Databricks Model Serving on the Penn Medicine Databricks platform, temperature 0.1 (where supported)

## License

## Acknowledgements 



