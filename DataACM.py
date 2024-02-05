import openai
from openai import OpenAI
import pandas as pd
import math
from dotenv import load_dotenv
import os

load_dotenv()  # This loads the .env file

client = OpenAI(
    api_key = os.getenv('OPENAI_API_KEY')
)

# Load your dataset
df = pd.read_csv('/home/kevin/ACM-PrePro/LLM-and-GNNs/3025acm.csv')

# Add the 'processed' and 'summary' columns if they don't exist
if 'processed' not in df.columns:
    df['processed'] = False
    #df['summary'] = ""
    df['KeyTerms'] = ""

# Function to generate summary and key terms using OpenAI API
def generate_summary_and_key_terms(citation):
    prompt = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": 
        "Task: Find the research paper, and summarize the paper using 75 terms in a TF-IDF friendly format. "
        "Then create a bag of words or a series of distinct semantic terms that encapsulate the main concepts of the paper, "
        "suitable for TF-IDF analysis. These key terms should be unique and not repeat any words or phrases used in the summary.\n"
        "Chain of Thought: First, find the research paper using the citation and read and understand the content. "
        "Then, identify the key themes and ideas in the paper for the summary, ensuring the summary consists of distinct, "
        "concise phrases. Lastly, select individual words or phrases that represent the core concepts, relevant to the paper, "
        "research field, and the specific topic of the paper, and list them separately for TF-IDF processing.\n "
        f"Citation: {citation}\n"
        "75 Key Terms (Formatted for TF-IDF):"}
    ]


    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=prompt
    )
    key_terms = response.choices[0].message.content

    # Check if 'content' contains the expected delimiter '\n\n

    return key_terms

# Function to process a batch of citations
def process_batch(dataframe, start, end):
    for index in range(start, end):
        if not dataframe.at[index, 'processed']:
            citation = str(dataframe.iloc[index])
            summary, key_terms = generate_summary_and_key_terms(citation)
            #dataframe.at[index, 'Summary'] = summary
            dataframe.at[index, 'KeyTerms'] = key_terms
            dataframe.at[index, 'processed'] = True
        print(f"Processed row index: {index}")

# Determine the number of batches
batch_size = 150
total_rows = len(df)
num_batches = math.ceil(total_rows / batch_size)

# Process all batches
batches_processed = df['processed'].sum() // batch_size
for batch in range(batches_processed, num_batches):
    start_index = batch * batch_size
    end_index = min(start_index + batch_size, total_rows)
    process_batch(df, start_index, end_index)

    # Save after every two batches
    if (batch - batches_processed + 1) % 2 == 0 or batch == num_batches - 1:
        df.to_csv('3025SumTerm.csv', index=False)
    print(f"Batch {batch}, Start Index: {start_index}")
