
"""
Processes synthetic clinical notes 

Use output for calculating costs and running inference

Contents:
  - Filter notes based on urine culture date
  - De-duplicate notes
  - Add start and end indicator to each note
  - Concatenate all notes for each patient
  - Export processed notes
"""

import pandas as pd

# import synthetic notes
notes_df = pd.read_csv("./synthetic_notes.csv")

#################################################################
# 1) Restrict to notes 2 days prior to and 5 days post UCX date
#################################################################
# convert to date type
notes_df["ucx_collect_date"] = pd.to_datetime(notes_df["ucx_collect_date"]).dt.date
notes_df["note_date"] = pd.to_datetime(notes_df["note_date"]).dt.date

# add vars for 2 days prior and 5 days post urine culture collect
notes_df["2d_prior"] = notes_df["ucx_collect_date"] - pd.DateOffset(days = 2)
notes_df["5d_post"] = notes_df["ucx_collect_date"] + pd.DateOffset(days = 5)

# ensure date is within range
notes_df["within_date_range"] = (
    (notes_df["note_date"] >= notes_df["ucx_collect_date"] - pd.Timedelta(days = 2)) &
    (notes_df["note_date"] <= notes_df["ucx_collect_date"] + pd.Timedelta(days = 5))
).astype(int)

# filter to dates within range
notes_df = notes_df[notes_df["within_date_range"] == 1]


########################################
# 2) De-duplicate notes
########################################
# if there are multiple timestamps per note_id, take the most recent one
# this should be the most updated version of the note

# get index of max note_date for each note_id
idx = notes_df.groupby("note_id")["note_date"].idxmax()

# return cols from original df
result_df = notes_df.loc[idx].reset_index(drop = True)

# select cols
result_df = result_df[['patient_id', 'note_id', 'ucx_collect_date', 'note_date', 
                       'note_text']].sort_values(by = ['patient_id', 'note_date'])


############################################################
# 3) Add start and end indicator to each note
############################################################
# unique patient may have multiple notes associated with encounter
# add S_O_R (start of report) before note and E_O_R (end of report) 
# after note to distinguish between each note
result_df["note_text"] = "S_O_R " + result_df["note_text"] + "\nE_O_R  "


##############################################
# 4) Concatenate all notes for each patient
##############################################
notes_concat = result_df.groupby('patient_id')['note_text'].agg(' '.join).reset_index() 


##############################################
# 5) Export processed notes
##############################################
notes_concat.to_csv("./synthetic_notes_concat.csv", index = False)
