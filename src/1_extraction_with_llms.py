'''
# Step 1: Load Example Data from the STI Unlabelled Dataset

We begin by selecting a sample of research abstracts from the STI (Science, Technology, and Innovation) unlabelled dataset. This data will serve as the input for our information extraction pipeline.
'''

import pandas as pd
from datasets import load_dataset

# load dataset 
ds_dict = load_dataset("SIRIS-Lab/unlabelled-sti-corpus")

# get train split
ds_train = ds_dict["train"]

# get 30 random exmaples
df_sampled_examples = pd.DataFrame(ds_train.shuffle(seed=42).select(range(500)))

# display(df_sampled_examples.head(4))

'''
# Step 2: Annotate Each Example Using Two LLMs

For each title and abstract, we will generate key information extractions using **two different language models (LLMs)**. Each model will produce its own structured output for the defined dimensions (motivations, objectives, methods, impact, and research topic). This provides options for downstream human annotation and quality comparison.
'''

prompt_base = '''Given the title and abstract of a research proposal/publications, extract and summarize the following information in a clear, structured, and programmatically friendly JSON format:

- **Motivations:**  
  Provide a list of 1 to 3 clear and concise sentences summarizing the main motivations or problems addressed by the research. Each motivation should be independent and not reference others. Avoid redundancy—if motivations are similar, combine or rephrase for clarity.

- **Objectives:**  
  Provide a list of 1 to 3 clear and concise sentences summarizing the main objectives of the research. Each objective should be independent and not reference others. If two objectives are highly similar, combine them into a single sentence to avoid redundancy.

- **Methods:**  
  Provide a list of 1 to 3 clear and concise sentences summarizing the main methods, techniques, or approaches. Omit specific details such as sample sizes, participant numbers, and timeframes.

- **Results:**  
  Provide a list of 1 to 3 clear and concise sentences summarizing the main results, expected results or impact. Omit details such as sample sizes, participant numbers, and timeframes.

- **Research Subject:**  
  Summarize the central research subject as a single, concise phrase or sentence.

**Formatting instructions:**  
Return only the output as valid JSON, following this structure (with no additional explanation or text):

```json
{
  "motivations": ["First motivation here", "Second motivation here", ...],
  "objectives": ["First research objective here", "Second research objective here", ...],
  "methods": ["First research method here", "Second research method here", ...],
  "results": ["First result/impact here", "Second result/impact here", ...],
  "research_subject": "Specific research subject here"
}

***Return only the JSON content, and nothing else. Return empty list if the dimensions are not described in the text.***

-----------------
Input:

Title: 
Democratising and making sense out of heterogeneous scholarly content

Abstract: 
SciLake's mission is to build upon the OpenAIRE ecosystem and EOSC services to (a) facilitate and empower the creation, interlinking and maintenance of Scientific/Scholarly Knowledge Graphs (SKGs) and the execution of data science and graph mining queries on top of them, (b) contribute to the democratization of scholarly content and the related added value services implementing a community-driven management approach, and (c) offer advanced, AI-assisted services that exploit customised perspectives of scientific merit to assist the navigation of the vast scientific knowledge space. In brief, SciLake will develop, support, and offer customisable services to the research community following a two-tier service architecture. First, it will offer a comprehensive, open, transparent, and customisable scientific data-lake-as-a-service (service tier 1), empowering and facilitating the creation, interlinking, and maintenance of SKGs both across and within different scientific disciplines. On top of that, it will build and offer a tier of customisable, AI-assisted services that facilitate the navigation of scholarly content following a scientific merit-driven approach (tier 2), focusing on two merit aspects which are crucial for the research community at large: impact and reproducibility. The services in both tiers will leverage advanced AI techniques (text and graph mining) that are going to exploit and extend existing technologies provided by SciLake's technology partners. Finally, to showcase the value of the provided services and their capability to address current and anticipated needs of different research communities, four scientific domains (neuroscience, cancer research, transportation, and energy) have been selected to serve as pilots. For each, the developed services will be customised, to accommodate differences in research procedures, practices, impact measures and types of research objects, and will be validated and evaluated through real-world use cases.

Output:

```json
{
  "motivations": [
    "Researchers face challenges in navigating, interlinking, and maintaining heterogeneous scholarly content, which limits the accessibility and usability of scientific knowledge.",
    "There is a need for democratized access and advanced tools to analyze and evaluate scholarly content based on scientific merit such as impact and reproducibility."
  ],
  "objectives": [
    "Facilitate the creation, interlinking, and maintenance of Scientific Knowledge Graphs across and within various scientific disciplines.",
    "Develop and provide AI-assisted, customizable services to support navigation and analysis of scholarly content based on scientific merit.",
    "Promote the democratization of scholarly content through a community-driven management approach."
  ],
  "methods": [
    "Implement a scientific data-lake-as-a-service to support the creation and maintenance of knowledge graphs.",
    "Leverage advanced AI techniques, including text and graph mining, to enable enhanced navigation and analysis of scholarly content.",
    "Customize and validate the developed services in pilot domains such as neuroscience, cancer research, transportation, and energy."
  ],
  "results": [
    "Empower researchers with effective tools to navigate, analyze, and utilize scholarly content.",
    "Enhance the reproducibility and impact assessment of scientific research.",
    "Support diverse research communities through tailored AI-driven services and promote wider access to scientific knowledge."
  ],
  "research_subject": "Democratization and advanced analysis of heterogeneous scholarly content using AI and knowledge graphs"
}
```
'''

from dotenv import load_dotenv

load_dotenv()

import os
import re
import json
from tqdm import tqdm
import openai
from together import Together
import anthropic

# Directory to store interim results
os.makedirs("data/interim/pseodoannotation", exist_ok=True)
interim_dir = "data/interim/pseodoannotation"

llama_client = Together(api_key=os.environ['TOGETHER_AI'])
openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def extract_json(text):
    json_match = re.search(r'```json(.*?)```', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
    else:
        print("No JSON block found.")
    return None

for i, row in tqdm(df_sampled_examples.iterrows(), total=len(df_sampled_examples)):
    id = row['id'].replace('/','_')
    interim_path = os.path.join(interim_dir, f"{id}.json")
    if os.path.exists(interim_path):
        # Skip if this example is already processed
        continue

    title = row['title']
    abstract = row['abstract'].replace('\n', ' ').replace('  ', ' ')
    prompt = f"""{prompt_base}
-----------------
Title:
{title}

Abstract:
{abstract}"""

    # --- Llama call ---
    llama_response = llama_client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.1,
        top_p=0.1,
        top_k=50,
        repetition_penalty=1,
        stop=["<|eot_id|>", "<|eom_id|>"],
        stream=False
    )
    llama_output = llama_response.choices[0].message.content

    # --- OpenAI call ---
    openai_response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful academic data assistant that extracts information from research documents."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=1500,
    )
    openai_output = openai_response.choices[0].message.content.strip()

    # --- Anthropic (Claude) call ---
    anthropic_response = anthropic_client.messages.create(
        model="claude-opus-4-20250514",  # or another claude-3 model if you prefer
        max_tokens=1500,
        temperature=0.1,
        system="You are a helpful academic data assistant that extracts information from research documents.",
        messages=[{"role": "user", "content": prompt}]
    )
    # For Claude v3, output is in .content (as a list of message blocks)
    anthropic_output = anthropic_response.content[0].text if hasattr(anthropic_response, "content") and anthropic_response.content else ""

    # Extract JSON outputs
    llama_json = extract_json(llama_output)
    openai_json = extract_json(openai_output)
    anthropic_json = extract_json(anthropic_output)

    result = {
        "index": i,
        "title": title,
        "abstract": abstract,
        "llama_json": llama_json,
        "openai_json": openai_json,
        "anthropic_json": anthropic_json,
        "llama_raw": llama_output,
        "openai_raw": openai_output,
        "anthropic_raw": anthropic_output,
    }

    # Save each result as a JSON file
    with open(interim_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
