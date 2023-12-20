import openai
import pandas as pd
import math

# Load your dataset
df = pd.read_csv('/home/kevin/acm-new.csv')

# Check if there's a column to track processed rows, if not create one
if 'processed' not in df.columns:
    df['processed'] = False

# Function to generate summary using OpenAI API
def generate_summary(citation):
    prompt = (
        "Task: Given each research paper with its citation, identify and summarize the paper using 75 words.\n"
        "Chain of Thought: First, find the paper and understand the main focus. Then, look for specific terms "
        "or phrases that are frequently mentioned or hold significant importance. Consider the context of these "
        "terms in relation to the overall research field and the specific topic of the paper.\n"
        f"Citation: {citation}\n"
        "Summary:"
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt}]
    )

    return response['choices'][0]['message']['content']

# Function to process a batch of citations
def process_batch(dataframe, start, end):
    for index in range(start, end):
        if not dataframe.at[index, 'processed']:  # Check if row is already processed
            citation = dataframe.at[index, 'Citation']
            summary = generate_summary(citation)
            dataframe.at[index, 'summary'] = summary
            dataframe.at[index, 'processed'] = True  # Mark as processed

# Determine the number of batches
batch_size = 300
total_rows = len(df)
num_batches = math.ceil(total_rows / batch_size)

# Process only two batches at a time
batches_processed = df['processed'].sum() // batch_size
for batch in range(batches_processed, min(batches_processed + 5, num_batches)):
    start_index = batch * batch_size
    end_index = min(start_index + batch_size, total_rows)
    process_batch(df, start_index, end_index)
    # Save after each batch
    df.to_csv(f'ACM-new_batch_{batch}.csv', index=False)

# Optionally save the DataFrame after processing two batches
df.to_csv('ACM-new_partial.csv', index=False)

