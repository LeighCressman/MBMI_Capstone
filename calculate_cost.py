"""
Tokenizes notes and prompts for cost estimate 

Contents:
  - Count tokens for prompts
  - Count tokens for notes
  - Combine prompt and note tokens
  - Print summary
  - Generate cost
"""


import json
import os
import tiktoken
import pandas as pd

###############################
# 1) Count tokens for prompts
###############################

# used for both gpt-5-mini and gpt-5
enc = tiktoken.get_encoding("o200k_base")

json_folder = "./uti_prompts"

# count tokens for each prompt file 
prompt_token_counts = {}

for filename in os.listdir(json_folder):
    if filename.endswith(".json"):
        filepath = os.path.join(json_folder, filename)
        with open(filepath, "r") as f:
            data = json.load(f)
        prompt_text = list(data.values())[0]
        token_count = len(enc.encode(prompt_text))
        prompt_token_counts[filename] = token_count


###############################
# 2) Count tokens for notes
###############################

# file generated in pre-processing step
full_notes_df = pd.read_csv("./synthetic_notes_concat.csv")

# count tokens for each note
full_notes_df["note_token_count"] = full_notes_df["note_text"].apply(lambda x: len(enc.encode(str(x))))
total_note_tokens = full_notes_df["note_token_count"].sum()
num_notes = len(full_notes_df)


###################################
# 3) Combine prompt and note tokens
###################################

# combine prompt and note tokens
summary_rows = []

# multiply number of tokens in each note by number of notes; add tokens from text of all notes
for filename, prompt_tokens in prompt_token_counts.items():
    total_input_tokens = (prompt_tokens * num_notes) + total_note_tokens
    summary_rows.append({
        "prompt_file": filename,
        "prompt_tokens": prompt_tokens,
        "note_tokens": total_note_tokens,
        "num_notes": num_notes,
        "total_input_tokens": total_input_tokens
    })

summary_df = pd.DataFrame(summary_rows)


###################################
# 4) Print summary
###################################
# summary
print(f"Number of notes: {num_notes}")
print(f"Total note tokens: {total_note_tokens}")
print(f"Average note tokens: {total_note_tokens / num_notes:.0f}")
print(f"\nPer-prompt token summary:")
print(summary_df.to_string(index = False))
print(f"\nGrand total input tokens across all prompts: {summary_df['total_input_tokens'].sum():,}")


###################################
# 5) Generate cost
###################################
# use grand total from above
total_input_tokens = 722_352 

# pricing (per 1M tokens) 
# GPT-4o:      $2.50 input / $10.00 output
# GPT-4o mini: $0.15 input / $0.60 output  
# GPT-4 (legacy): $30.00 input / $60.00 output
# GPT-5-Mini: $0.25 input / $2.00 output
# GPT-5: $1.25 input / $10.00 output

input_price_per_1M  = 0.25   
output_price_per_1M = 2.00  

# estimate output tokens 
# typical short classification response is ~100-200 tokens
estimated_output_tokens_per_note = 150
num_notes = len(full_notes_df)
num_prompts = 9
total_output_tokens = estimated_output_tokens_per_note * num_notes * num_prompts

# cost calculation 
input_cost  = (total_input_tokens  / 1_000_000) * input_price_per_1M
output_cost = (total_output_tokens / 1_000_000) * output_price_per_1M
total_cost  = input_cost + output_cost

print(f"Total input tokens:           {total_input_tokens:,}")
print(f"Total output tokens (est.):   {total_output_tokens:,}")
print(f"\nInput cost:                   ${input_cost:,.2f}")
print(f"Output cost:                  ${output_cost:,.2f}")
print(f"Total estimated cost:         ${total_cost:,.2f}")
