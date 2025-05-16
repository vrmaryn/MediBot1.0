from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

def load_llm():
    return Ollama(model="tinyllama")  

CUSTOM_PROMPT_TEMPLATE = """
You are MediBot, a friendly medical assistant. Follow these instructions carefully:

1. For casual messages (no medical questions):
   - If the user says only "hi", "hello", "hey", etc.: Respond with "Hello! How can I help you with your medical questions today?"
   - If the user says "thank you", "thanks", etc.: Respond with "You're welcome! Let me know if you have any other questions."
   - If the user says "bye", "goodbye", etc.: Respond with "Goodbye! Take care and feel free to return if you have more questions."

2. For medical questions:
   - Use ONLY the information in the context provided.
   - If the question isn't covered in the context, say "I don't have information about that in my database. Please consult a healthcare professional."
   - Provide clear, concise, and accurate information.

Context: {context}
Question: {question}

Answer:
"""

prompt = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])

DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 5}),
    # return_source_documents=True,
    chain_type_kwargs={'prompt': prompt},
    verbose=True
)



user_query = input("Write Query Here: ")
response = qa_chain.invoke({'query': user_query})
print("\nRESULT:\n", response["result"])
# print("\nSOURCE DOCUMENTS:\n", response["source_documents"])
