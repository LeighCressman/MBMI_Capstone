"""
Generates synthetic clinical notes 

Contents:
  - Parameters which are similar to actual study (cohort size, symptom prevalence, etc.)
  - Helper functions
  - Code for creating dataset with human labels
  - Code for creating note templates
  - Code for creating dataset with synthetic notes
"""


import numpy as np
import pandas as pd
from datetime import datetime, timedelta


########################################
# 1) Set parameters
########################################

# set seed
np.random.seed(42)

# set to same cohort size as study
N_PATIENTS = 286

# UTI symptoms
SYMPTOMS = [
    "dysuria",
    "cvat",
    "hematuria",
    "fever",
    "flank_pain",
    "septic_shock",
    "suprapubic_pain",
    "urinary_frequency",
    "urinary_urgency"
]

# based on symptom prevalence in study cohort
SYMPTOM_PREV = {
    "dysuria": 0.13,
    "cvat": 0.01,
    "hematuria": 0.02,
    "fever": 0.15,
    "flank_pain": 0.06,
    "septic_shock": 0.01,
    "suprapubic_pain": 0.05,
    "urinary_frequency": 0.08,
    "urinary_urgency": 0.04
}

# based on overall prevalence of asymptomatic bacteriuria (ASB) in cohort
ASB_PREVALENCE = 0.64


######################################
# 2) Define helper functions
######################################
# generate random date
def random_date(start, end):
    delta = (end - start).days
    return start + timedelta(days = np.random.randint(0, delta + 1))

# generate random 6-digit patient ids
def generate_patient_ids(n):
    return np.random.choice(
        np.arange(100000, 999999),
        size = n,
        replace = False
    )


######################################
# 3) Create dataset with hand labels
######################################

# generate patient ids
patient_ids = generate_patient_ids(N_PATIENTS)

# randomly select which patients have asymptomatic bacteriuria (ASB)
# and convert to set
asb_patients = set(
    np.random.choice(
        patient_ids,
        size = int(N_PATIENTS * ASB_PREVALENCE),
        replace = False
    )
)

# create new dataframe 
labels = []
patient_symptoms = {}
for patient_id in patient_ids:
    patient_symptoms[patient_id] = {}

    # if patient has ASB, all symptoms are absent
    if patient_id in asb_patients:
        for symptom in SYMPTOMS:
            patient_symptoms[patient_id][symptom] = 0
    else:
        # otherwise generate symptoms randomly
        for symptom in SYMPTOMS:
            patient_symptoms[patient_id][symptom] = int(
                np.random.random() < SYMPTOM_PREV[symptom]
            )

        # check that non-ASB patients have at least one symptom
        if sum(patient_symptoms[patient_id].values()) == 0:
            symptom = np.random.choice(SYMPTOMS)
            patient_symptoms[patient_id][symptom] = 1

    for symptom in SYMPTOMS:
        labels.append({
            "patient_id": patient_id,
            "prompt_type": symptom,
            "symptom_present": patient_symptoms[patient_id][symptom]
        })

labels_df = pd.DataFrame(labels)


######################################
# 4) Create note templates
######################################

# templates for presence of UTI symtoms
POSITIVE_TEXT = {
    "dysuria": [
        "Reports dysuria.",
        "Endorses dysuria.",
        "Complains of burning with urination.",
        "Reports painful urination.",
        "Reports burning w/ urination."
    ],

    "cvat": [
        "CVAT present on exam.",
        "Positive CVAT.",
        "Costovertebral angle tenderness noted."
    ],

    "hematuria": [
        "Reports hematuria.",
        "Notes blood in urine.",
        "Gross hematuria reported."
    ],

    "fever": [
        "Febrile overnight.",
        "Reports fever and chills.",
        "Temperature elevated during admission."
    ],

    "flank_pain": [
        "Reports flank pain.",
        "Complains of right flank pain.",
        "Endorses bilateral flank discomfort."
    ],

    "septic_shock": [
        "Developed septic shock requiring vasopressors.",
        "Septic shock documented.",
        "Shock requiring pressor support."
    ],

    "suprapubic_pain": [
        "Reports suprapubic pain.",
        "Complains of suprapubic discomfort.",
        "Suprapubic tenderness present."
    ],

    "urinary_frequency": [
        "Reports urinary frequency.",
        "Endorses increased frequency.",
        "Voiding more frequently than usual."
    ],

    "urinary_urgency": [
        "Reports urinary urgency.",
        "Endorses urgency.",
        "Difficulty delaying urination."
    ]
}

# templates for absence of UTI symtoms
NEGATIVE_TEXT = {
    "dysuria": [
        "Denies dysuria.",
        "No dysuria reported.",
        "Negative for dysuria.",
        "No burning w/ urination."
    ],

    "cvat": [
        "No CVAT.",
        "Negative CVAT.",
        "No costovertebral angle tenderness."
    ],

    "hematuria": [
        "Denies hematuria.",
        "No blood in urine.",
        "Negative for hematuria."
    ],

    "fever": [
        "Afebrile.",
        "No fever reported.",
        "Negative for fever."
    ],

    "flank_pain": [
        "Denies flank pain.",
        "No flank pain.",
        "Negative for flank pain."
    ],

    "septic_shock": [
        "No evidence of septic shock.",
        "Hemodynamically stable.",
        "No septic shock documented."
    ],

    "suprapubic_pain": [
        "Denies suprapubic pain.",
        "No suprapubic discomfort.",
        "Negative for suprapubic tenderness."
    ],

    "urinary_frequency": [
        "Denies urinary frequency.",
        "No frequency reported.",
        "Negative for urinary frequency."
    ],

    "urinary_urgency": [
        "Denies urgency.",
        "No urinary urgency.",
        "Negative for urgency."
    ]
}

# note introduction phrases
INTRO_PHRASES = [
    "Hospital medicine progress note.",
    "Internal medicine admission note.",
    "Daily inpatient progress note.",
    "Clinical documentation review.",
    "Hospitalist follow-up note."
]

# note ending phrases
ENDING_PHRASES = [
    "Urine culture positive during admission.",
    "Positive urine culture noted.",
    "Microbiology results reviewed.",
    "Clinical status monitored.",
    "Care plan discussed with patient."
]


######################################
# 5) Dataset with notes
######################################

# create new dataframe 
notes = []

# 8-digit note id
note_id_counter = 10000000

# urine culture start and end dates
ucx_start = datetime(2023, 6, 20)
ucx_end = datetime(2024, 6, 30)

# patient note start and end dates
note_start = datetime(2023, 6, 18)
note_end = datetime(2024, 7, 4)

# randomly choose number of notes for patient
# each patient can have 1 - 4 notes  
# chance of patient having 1 note = 60%, etc.
for patient_id in patient_ids:
    n_notes = np.random.choice(
        [1, 2, 3, 4],
        p=[0.60, 0.25, 0.10, 0.05]
    )

    # randomly select urine culture date
    ucx_date = random_date(ucx_start, ucx_end)
    
    # symptom labels for patient
    symptoms = patient_symptoms[patient_id]

    # create individual notes for patient
    for _ in range(n_notes):

        # note date must be within 2 days prior to and 5 days after urine culture
        note_window_start = ucx_date - timedelta(days = 2)
        note_window_end = ucx_date + timedelta(days = 5)

        # select random date within window
        note_date = random_date(
            note_window_start,
            note_window_end
        )

        # randomly select text for intro phrase
        note_parts = [
            np.random.choice(INTRO_PHRASES)
        ]

        # randomly shuffle order of symptoms
        symptom_order = np.random.permutation(SYMPTOMS)

        # randomly select text for each symptom present in note
        for symptom in symptom_order:
            if symptoms[symptom] == 1:
                note_parts.append(
                    np.random.choice(POSITIVE_TEXT[symptom])
                )
            else:
                note_parts.append(
                    np.random.choice(NEGATIVE_TEXT[symptom])
                )

        # randomly select ending phrase
        note_parts.append(
            np.random.choice(ENDING_PHRASES)
        )

        # combine note parts
        note_text = " ".join(note_parts)

        notes.append({
            "patient_id": patient_id,
            "note_id": note_id_counter,
            "ucx_collect_date": ucx_date.date(),
            "note_date": note_date.date(),
            "note_text": note_text
        })

        note_id_counter += 1

notes_df = pd.DataFrame(notes)


######################################
# 6) Check counts
######################################

# unique patients
print(f"Patients: {len(patient_ids)}")

# unique notes
print(f"Notes: {len(notes_df)}")

# rows in hand-labeled dataset
print(f"Label rows: {len(labels_df)}")


######################################
# 7) Export datasets
######################################
# dataset with notes
notes_df.to_csv("./synthetic_notes.csv", index = False)

# dataset with hand labels
labels_df.to_csv("./synthetic_labels.csv", index = False)
