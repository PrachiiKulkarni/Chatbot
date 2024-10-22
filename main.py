__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import streamlit as st
from openai import OpenAI
from datasets import Dataset
from ragas.metrics import answer_relevancy, faithfulness
from ragas import evaluate
import chromadb
from chromadb.config import Settings

open_ai_key = st.secrets("OPENAI_API_KEY")
LLM_model = st.secrets("LLM_model")

# Load an existing Chroma collection
def load_chroma_collection(name, persist_directory):
    client = chromadb.Client(Settings(persist_directory=persist_directory))
    return client.get_collection(name=name)

def get_relevant_passage(query, db, n_results):
        results = db.query(query_texts=[query], n_results=n_results)
        return results

# Construct a prompt for the generation model based on the query and retrieved data
def make_rag_prompt(query, relevant_passage):
    # escaped_passage = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = f"""You are a helpful and informative bot that answers questions using text from the reference passage included below.
If the question is in "French" language, extract query from documents whose text is in French.
If the question is in "English" language, extract query from documents whose text is in English.
The input question might be in "English" or "French" language but be sure to answer the query only in "English" language.
Be sure to respond in a complete sentence, being comprehensive, including all relevant background information.
However, you are talking to a non-technical audience, so be sure to break down complicated concepts and
strike a friendly and conversational tone.
If you dont know the answer to a question, say "I don't know" or "I couldn't find answer to the question in given information stack".
QUESTION: '{query}'
PASSAGE: '{relevant_passage}'

ANSWER:
"""
    return prompt

def generate_answer(prompt, open_ai_key, LLM_model):
    '''
    Using 'gpt-3.5-turbo' model to generate answer based on the prompt.
    '''
    client = OpenAI(api_key=open_ai_key)
    chat_completion = client.chat.completions.create(
        messages=[{
                "role": "user",
                "content": prompt,
            }],
        model=LLM_model,
        max_tokens=300,
        temperature=0.7
    )

    return chat_completion.choices[0].message.content

def evaluate_answer(query, answer, relevant_text):
    data_samples = {
        'question': [query],
        'answer': [answer],
        'contexts' : relevant_text['documents']
        }
    dataset = Dataset.from_dict(data_samples)
    score = evaluate(dataset,metrics=[faithfulness, answer_relevancy])
    df = score.to_pandas()
    return df['faithfulness'].values[0], df['answer_relevancy'].values[0]

# Interactive function to process user input and generate an answer
def process_query_and_generate_answer(query):
    if not query:
        return f"No query provided."
    
    collection = "RAG2"
    n_results = 5
    persist_directory = "RAG2/"

    db = load_chroma_collection(collection, persist_directory)
    relevant_text = get_relevant_passage(query, db, n_results=n_results)

    if not relevant_text:
        return f"No relevant information found for the given query."
    final_prompt = make_rag_prompt(query, relevant_text['documents'] )
    answer = generate_answer(final_prompt, open_ai_key, LLM_model)
    faithfulness, answer_relevancy = evaluate_answer(query, answer, relevant_text)

    return f"The answer for given query is {answer}. \n\n The evaluation metrics for generated answer are \n 1. Faithfulness: {faithfulness} \n 2. Answer Relevancy: {answer_relevancy} \n\n and relevant documents are {relevant_text}."

#### Streamlit app ####
st.title("💬 Chatbot")
st.caption("🚀 A Streamlit chatbot powered by OpenAI")

#first message as assistant
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Input message from user
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User submits its input message(called prompt)
if prompt := st.chat_input():
    # if not openai_api_key:
    #     st.info("Please add your OpenAI API key to continue.")
    #     st.stop()

    client = OpenAI(api_key=open_ai_key)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    query = st.session_state.messages[-1]['content']
    # response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    response = process_query_and_generate_answer(query)

    # msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
