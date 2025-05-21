from typing import Generator
import streamlit as st
from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.openai import OpenAI
import time

from pinecone import Pinecone
import os

from dotenv import load_dotenv

load_dotenv(override=True)


# os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
# os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

def gerador_resposta(texto):
    for palavra in texto.split(" "):
        yield palavra + " "
        time.sleep(0.02)


system_prompt = (
    "Você é um sistema de perguntas e respostas especializado em dicas de viagem para o Japão, confiável em todo o mundo.\n"
    "Sempre responda à consulta usando apenas as informações do contexto fornecido, "
    "e não conhecimentos prévios.\n"
    "Algumas regras a seguir:\n"
    "1. Nunca faça referência direta ao contexto dado em sua resposta.\n"
    "2. Evite declarações como 'Com base no contexto, ...' ou "
    "'As informações do contexto ...' ou qualquer coisa semelhante.\n"
)

context_prompt = ("""
            A seguir, uma conversa amigável entre um usuário e um assistente de IA.
            O assistente é comunicativo e fornece muitos detalhes específicos do seu contexto.
            Se o assistente não souber a resposta para uma pergunta, ele diz honestamente que
            não há dados nos documentos para responder.

            Aqui estão os documentos relevantes para o contexto:

            {context_str}

            Instrução: Com base nos documentos acima, forneça uma resposta detalhada, listando todas as informações possíveis, para a pergunta do usuário abaixo.
            Responda com uma estrutura em markdown e sempre que possível deixando em negrito caminhos, títulos e nomes.
            Responda "não sei" se não estiver presente no documento.
            """)

INDEX_NAME = "indice-rag"
LLM_MODEL = "gpt-4.1-mini"
EMBED_MODEL = "text-embedding-3-small"

pc = Pinecone()
pinecone_index = pc.Index(INDEX_NAME)
vector_store = PineconeVectorStore(pinecone_index)
vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
retriver = VectorIndexRetriever(index=vector_index, similarity_top_k=5)

Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL)
Settings.llm = OpenAI(model=LLM_MODEL, temperature=0.1)

chat_engine = CondensePlusContextChatEngine.from_defaults(retriever=retriver, system_prompt=system_prompt, context_prompt=context_prompt)

# Interface do chatbot
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Faça sua pergunta:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        # Obter resposta do chat engine
        resposta_completa = chat_engine.chat(prompt)
        
        # Exibir resposta com efeito de digitação
        resposta_stream = gerador_resposta(str(resposta_completa.response))
        resposta = st.write_stream(resposta_stream)
    
    st.session_state.messages.append({"role": "assistant", "content": resposta})
