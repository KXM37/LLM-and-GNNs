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
df = pd.read_csv('/home/kevin/ACM-PrePro/LLM-and-GNNs/PvAvC1.csv')

paper_db = np.isin(df['Conferences'], ['SIGMOD', 'VLDB'])
paper_dm = np.isin(df['Conferences'], ['KDD'])
paper_wc = np.isin(df['Conferences'], ['SIGCOMM', 'MobiCOMM'])

# Combine indices
paper_idx = np.sort(np.concatenate([
    np.where(paper_db)[0],
    np.where(paper_dm)[0],
    np.where(paper_wc)[0]
]))


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
        "Task: Find the research paper and represent the paper using 75-150  semantic terms in a TF-IDF friendly format. "
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
    key_terms_GPT = response.choices[0].message.content

    key_terms = transform_terms(key_terms_GPT)

    return key_terms

def transform_terms(term_str):
    return [term.split('. ', 1)[-1] for term in term_str.split('\n') if term]

# Function to process a batch of citations
def process_batch(dataframe, batch_indices):
    for index in batch_indices:
        if not dataframe.at[index, 'processed']:
            citation = str(dataframe.iloc[index])
            key_terms = generate_summary_and_key_terms(citation)
            #dataframe.at[index, 'Summary'] = summary
            dataframe.at[index, 'KeyTerms'] = key_terms
            dataframe.at[index, 'processed'] = True
            print(f"Processed row index: {index}")
            

# Determine the number of batches
batch_size = 150
total_rows = len(paper_idx)
num_batches = math.ceil(total_rows / batch_size)
#num_batches = (len(paper_idx) + batch_size - 1) // batch_size  # Calculate number of batches needed

for batch in range(num_batches):
    start_index = batch * batch_size
    end_index = min(start_index + batch_size, len(paper_idx))
    batch_indices = paper_idx[start_index:end_index]
    process_batch(df, batch_indices)

    # Save after every batch for simplicity in this example
    df.to_csv('3025SumTerm.csv', index=False)
    print(f"Batch {batch}, Processed Indices Range: {start_index} to {end_index}")
