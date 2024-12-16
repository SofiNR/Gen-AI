import ollama
import pandas as pd
from datasets import load_dataset

from datasets import Dataset 
from ragas.metrics import summarization_score
from ragas import evaluate

#from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
#pip install -U langchain-community
import matplotlib.pyplot as plt

import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

ds = load_dataset("sujayC66/text_summarization_512_length_1_4000")
df = pd.DataFrame(ds['train'])
df = df.rename(columns={"content": "text"})
print(df.info())

df = df.drop(columns = ["summary", "__index_level_0__"])

df = df.sample(10)

def word_count(text):
    return len(text.split())

df["nwords_text"] = df["text"].apply(word_count)
print(df.info())

def generate_summary(text, model):
    
    nwords_summary = int(0.2*word_count(text))
    
    prompt =  f"Your goal is to summarize the given text in maximum {nwords_summary} words. \
               Extract the most important information. \
               Only output the summary without any additional text."

    
    response = ollama.chat(model=model, messages=[
        {
            'role': 'system',
            'content': prompt
        },
        {
            'role': 'user',
            'content': text,
        },
    ])
    summary = response['message']['content']
    
    return summary

#ollms = ["llama3.2:3b", "mistral:7b", "phi3.5:3.8b",  "gemma2:9b", "qwen2.5:7b"]  

ollms = ["llama3.2:3b", "mistral:7b"]

# As its time consuming, doing with only 2 models from ollama

# ollama run llama3.2:3b #2GB
# ollama run mistral:7b #4.1GB
# ollama run phi3.5:3.8b #2.2GB
# ollama run gemma2:9b #5.1GB
# ollama run qwen2.5:7b #4.7GB

for llm in ollms: 
    print(llm)
    df[llm + "_summary"] = df["text"].apply(lambda x: generate_summary(x, model=llm))

print(df.info())

for llm in ollms: 
    col = llm + "_summary"
    col_cnt = llm + "_summary_nwords_percnt"
    df[col_cnt] = df[col].apply(word_count)
    df[col_cnt] = 100*df[col_cnt]/df['nwords_text']
    
print(df.info())

df.to_csv('1_df.csv')

contexts =  [[text] for text in df["text"]]

evaluator_llm  = ChatOpenAI(model_name="gpt-4o-mini")

df_scores = pd.DataFrame()

for llm in ollms: 
    print(llm)
    col = llm + "_summary"
    # data_samples = {
    #     'contexts' : contexts,
    #     'summary': df[col].to_list()
    # }

    data_samples = {
        'reference_contexts' : contexts,
        'response': df[col].to_list()
    }
    dataset = Dataset.from_dict(data_samples)
    
    score = evaluate(dataset,
                     metrics=[summarization_score],
                     llm=evaluator_llm,
                    )
    
    df_scores[llm] = score.to_pandas()['summary_score']

print(df_scores.info())

df_scores.to_csv('2_scores.csv')

# Assuming you already have the DataFrame `df`
# Calculating the mean and standard deviation for each model
mean_scores = df_scores.mean()
std_scores = df_scores.std()

# Creating a DataFrame for plotting
summary_df = pd.DataFrame({
    'Model': mean_scores.index,
    'Mean Score': mean_scores.values,
    'Standard Deviation': std_scores.values
})

# Sorting the DataFrame to ensure the order in the plot
summary_df = summary_df.sort_values('Mean Score', ascending=True)

summary_df['Mean Score'] = summary_df['Mean Score'].round(2)

print(summary_df)

summary_df.to_csv('3_summary.csv')

# Create the plot with a transparent figure
fig, ax = plt.subplots(figsize=(12, 5), facecolor='none', edgecolor='none')
fig.patch.set_alpha(0)

# Create horizontal bar plot
bars = ax.barh(summary_df['Model'], summary_df['Mean Score'], height=0.65, color="green"
              )

# Remove y-axis labels
ax.set_yticks([])

# Remove the frame
for spine in ax.spines.values():
    spine.set_visible(False)

# Add count labels to the end of each bar
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2, f' {width}', 
            ha='left', va='center', fontweight='bold', fontsize=15, color='red')

# Add name labels inside each bar
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width/2, bar.get_y() + bar.get_height()/2, summary_df['Model'].iloc[i], 
            ha='center', va='center', fontweight='bold', fontsize=15, color='white')

# Set the title
#ax.set_title('Count by Name', fontsize=16, fontweight='bold', pad=20)

# Remove x-axis label
ax.set_xlabel('')

# Set the background of the axis to transparent
ax.patch.set_alpha(0)

# Adjust layout and display the plot
plt.tight_layout()

plt.suptitle("LLMs Text Summarisation Performance", fontsize=20)
# Save the figure with transparent background
plt.savefig('summary1.png', dpi=300, bbox_inches='tight', transparent=True)

# Display the plot (optional, remove if you only want to save the file)
plt.show()