"""
Hit Model Endpoints 

Contents:
  - Create load and fill helper functions
  - Split into training and test sets
  - Load prompt and fill with note text 
  - Send synthetic notes to OpenAI API
  - Merge with hand-labeled synthetic notes
  - Evaluate model performance and export results to MLflow
"""

import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import mlflow.deployments as mlfd
from datetime import datetime, timezone
import pytz
from openai import OpenAI
est = pytz.timezone('US/Eastern')


################################################################
# 1) Create load and fill helper functions
# from https://github.com/ugurcanvurgun/sdoh-llm-clinical-notes
################################################################

# function definitions
def load_prompts(path: Optional[Path] = None) -> Dict[str, str]:
    """Load all prompts from JSON folder."""
    prompt_path = path 
    with prompt_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)

def fill_prompt(note_text: str, prompts: Optional[Dict[str, str]] = None) -> str:
    """Return a ready-to-send prompt with the note text substituted in."""
    if prompts is None:
        prompts = load_prompts()
    template = list(prompts.values())[0]
    return template.format(text = note_text)

########################################
# 2) Split into training and test dfs
########################################

# import concatenated notes
notes_concat = pd.read_csv("./synthetic_notes_concat.csv")

data = [i for i in range(100)]
train, test = train_test_split(notes_concat, test_size = 0.3, random_state = 42)
print(f"Train size: {len(train)}, Test size: {len(test)}")

training_data = train

################################################################
# 3) Load prompt and fill with note text for training dataset
################################################################

json_folder = Path("./uti_prompts")

results = []

for filename in os.listdir(json_folder):
    if filename.endswith(".json"):
        # load prompt
        prompt_path = json_folder / filename
        prompts = load_prompts(prompt_path)
        prompt_name = filename.replace(".json", "")

        # fill prompt for each note
        for index, row in training_data.iterrows():  
            filled_prompt = fill_prompt(note_text = row["note_text"], prompts = prompts)
            results.append({
                "prompt_file": prompt_name,
                "patient_id": row["patient_id"],
                "filled_prompt": filled_prompt
                
            })


# Convert filled prompts to df
# 200 training patients x 9 prompts per patient
results_df = pd.DataFrame(results)
print(f"Total filled prompts: {len(results_df)}")
print(results_df.head())


########################################
# 4) Send synthetic notes to OpenAI API
########################################

client = OpenAI(api_key = "YOUR_API_KEY")

results = []

for _, row in results_df.iterrows():

    response = client.responses.create(
        model="gpt-5-mini",
        input=row["filled_prompt"]
    )

    response_text = response.output_text

    results.append({
        "record_id": row["record_id"],
        "prompt_file": row["prompt_file"],
        "filled_prompt": row["filled_prompt"],
        "response": response_text
    })

results_df = pd.DataFrame(results)


###################################################
# 5) create 1/0 col for presence/absence of symptom
###################################################
results_df["llm_label"] = results_df["response"].str.contains("Presence", case = False, na = False).astype(int)

# get subset of cols
labelled_df = results_df[['patient_id', 'prompt_file', 'llm_label']]

##################################################
# 6) Import hand labels and merge with LLM output
##################################################

# import synthetic hand labels
# rename prompt type column for merging
hand_labels = pd.read_csv("./synthetic_labels.csv").rename(columns={"prompt_type": "prompt_file"})

# merge LLM labels and hand labels
labels_merged = pd.merge(labelled_df, hand_labels, on = ['patient_id', 'prompt_file'], how = 'inner')


##################################################
# 7) Set MLflow experiment and generate metrics
##################################################

# close active runs if open
if mlflow.active_run():
    mlflow.end_run()

# set experiment
mlflow.set_experiment("./YOUR_EXPERIMENT_PATH")

# load data
labels_merged["llm_label"] = labels_merged["llm_label"].astype(int)
labels_merged["symptom_present"] = labels_merged["symptom_present"].astype(int)

y_true = labels_merged["symptom_present"]
y_pred = labels_merged["llm_label"]

# start run
with mlflow.start_run(run_name="YOUR_RUN_NAME"):

    # 1) overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    mlflow.log_metric("overall_accuracy", accuracy)
    mlflow.log_metric("overall_precision", precision)
    mlflow.log_metric("overall_recall", recall)
    mlflow.log_metric("overall_f1", f1)

    # 2) ROC curve
    y_scores = y_pred  

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle='--')  # diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.savefig("roc_curve.png")
    plt.close()

    mlflow.log_artifact("roc_curve.png")
    mlflow.log_metric("roc_auc", roc_auc)

    # 3) precision-recall curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_scores)

    plt.figure()
    plt.plot(recall_vals, precision_vals)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")

    plt.savefig("pr_curve.png")
    plt.close()

    mlflow.log_artifact("pr_curve.png")

    # area under PR curve
    pr_auc = auc(precision_vals, recall_vals)
    mlflow.log_metric("pr_auc", pr_auc)

    # 4) confusion matrix 
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.title("Confusion Matrix")

    plt.savefig("confusion_matrix.png")
    plt.close()  # important to avoid overlapping plots

    mlflow.log_artifact("confusion_matrix.png")

    # 5) metrics by prompt_file
    for prompt, group in labels_merged.groupby("prompt_file"):
        y_true_group = group["symptom_present"]
        y_pred_group = group["llm_label"]

        mlflow.log_metric(f"{prompt}_accuracy",
                          accuracy_score(y_true_group, y_pred_group))
        mlflow.log_metric(f"{prompt}_precision",
                          precision_score(y_true_group, y_pred_group, zero_division = 0))
        mlflow.log_metric(f"{prompt}_recall",
                          recall_score(y_true_group, y_pred_group, zero_division = 0))
        mlflow.log_metric(f"{prompt}_f1",
                          f1_score(y_true_group, y_pred_group, zero_division = 0))
        
        # add confusion matrix for each prompt
        disp = ConfusionMatrixDisplay.from_predictions(y_true_group, y_pred_group)
        plt.title(f"Confusion Matrix: {prompt}")

        filename = f"confusion_matrix_{prompt}.png"
        plt.savefig(filename)
        plt.close()  

        mlflow.log_artifact(filename)


    # 6) log metadata 
        mlflow.log_param("dataset", "YOUR_DATASET.csv")
        mlflow.log_param("n_rows", len(labels_merged))
        mlflow.log_param("label_column", "symptom_present")
        mlflow.log_param("prediction_column", "llm_label")
        mlflow.log_param("task", "binary_classification")
        mlflow.log_param("prompt_version", "YOUR_PROMPT_VERSION")

print("All metrics logged in ONE MLflow run")
