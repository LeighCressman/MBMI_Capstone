## Detecting Asymptomatic Bacteriuria in Clinical Notes using Large Language Models (LLMs)
This repository contains the code for Leigh Cressman's capstone project in the Master of Biomedical Informatics (MBMI) program at the University of Pennsylvania (crel@pennmedicine.upenn.edu). 

Included in this repository are: 
1) Code for generating synthetic patient notes
2) Prompt templates for urinary tract infection (UTI) symptoms
3) Code for calculating the cost to use large language models (LLMs) to process clinical notes
4) Code for preprocessing clinical notes
5) Code for hitting model endpoints, evaluating model performance, and tracking experiments in MLflow

## Note on LLM Inference Infrastructure
This project used Databricks Model Serving endpoints on the Penn Medicine Databricks platform for performing LLM inference. Said platform is a HIPAA-compliant, institutionally managed environment. No patient notes were sent to external OpenAI APIs. The model endpoints used in the study (e.g., "databricks-gpt-5-mini") are Databricks-hosted serving endpoints that use OpenAI models which are part of Penn Medicine's Azure environment.

The code in run_inference.py is configured to allow users to provide their own API key when sending synthetic patient notes to the OpenAI API.

## Study Background
Antimicrobial stewardship programs (ASPs) are limited in managing misdiagnosis of asymptomatic bacteriuria (ASB) as urinary tract infection (UTI) because the latter diagnosis is symptom based. This study develops and validates a zero-shot, LLM-based approach for identifying nine UTI symptoms in clinical notes. Symptoms in the study include 1) urinary urgency, 2) urinary frequency, 3) dysuria, 4) hematuria, 5) fever, 6) septic shock, 7) suprapubic pain, 8) flank pain, and 9) costovertebral tenderness.

The original study cohort includes 286 inpatient and emergency department (ED) patient encounters occurring between 2023 and 2024 at two urban hospitals in the northeast U.S. Patients were treated for UTIs, had positive urine cultures, and were not catheterized. Expert-based manual chart review was leveraged to generate the reference standard. Data were randomly split into training (n = 200) and test (n = 86) sets, with two batches sampled from the training set for prompt engineering. A third validation batch was sampled from the test set.

GPT-5-Mini achieved 0.812, 0.898, and 0.929 accuracy across nine UTI symptoms in the first, second, and third batches, respectively. GPT-5 achieved 0.938 accuracy in the third batch.

No real patient data is used in this repository. The code in generate_synthetic_notes.py creates a cohort of the same size as the study cohort (n = 286). Furthermore, the prevalence rates for individual UTI symptoms and overall asymptomatic bacteriuria in the script reflect those of the real cohort in our study.

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



