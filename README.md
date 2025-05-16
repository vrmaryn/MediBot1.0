# MediBot1.0
Langchain Based Medical Assistance that will behave like a doctor and give answer based on your Problems. 
# 🩺 MediBot - AI Medical Assistant Chatbot

MediBot is an intelligent medical assistant chatbot built using **Streamlit**, designed to assist users with medical queries. It uses a **local FAISS vector store** to retrieve relevant medical documents and leverages **LLMs** to provide concise and reliable answers. If the query cannot be answered from local data, it can optionally fall back to **Wikipedia** or **arXiv** for information.

## 🚀 Features

- 🧠 **LLM-Powered Q&A**: Provides answers to user questions using large language models.
- 🗂️ **Local Vector Store (FAISS)**: Retrieves information from a custom dataset of medical documents.
- 🌐 **Fallback to External Sources**: Optionally queries Wikipedia and arXiv when local data is insufficient.
- 💬 **Streamlit Chat Interface**: A clean and user-friendly interface built using Streamlit.
- 🔒 **Offline Support**: Works offline using locally stored vector index and documents (no external API needed unless fallback is enabled).

