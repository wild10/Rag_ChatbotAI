#############################################################################
## Chatbot que recupera información de una lista de PDfs usando RAG + LLMs
## version local ( lee pdfs desde disco)
## author:wildr.10@gmail.com
#############################################################################


import os
import re 

# Importar las librerias necesarias.
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

import fitz #PyMuPDF para extraccion de texto
import gradio as gr 

# librerias langchain
from langchain_community.embeddings import OllamaEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOllama 

from langchain.prompts import PromptTemplate

###---------------------------------------------------------------
##  Cargar PDFs en la memoria local desde folder "data"
###---------------------------------------------------------------

def load_pdfs_from_folder(folder_path = "./data"):
    texts = []
    
    for file in os.listdir(folder_path):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file )
            text = ""

            # Extraer el texto con fitz
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text()
            
            texts.append(text)
    
    return texts 

# Carga (local) PDFs
folder_name = "./data"
pdf_texts = load_pdfs_from_folder(folder_name)

# print(f"pdf cargados:{len(pdf_texts)}")
# print(f"el 1 pdf: {pdf_texts[1]}")

###---------------------------------------------------------------
##  Dividir el documento en chunks(partes)
###---------------------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500, # Num characters : 300-800 (tamaño chunks)
    chunk_overlap = 100, # existe overlaping para no perder conexto
    length_function = len,
    is_separator_regex = False,
)

documents = [Document(page_content=text) for text in pdf_texts]
# Separad chunks desde el pdf
splitted_documents = text_splitter.split_documents(documents)
list_texts = [doc.page_content for doc in splitted_documents]

# print(f"size of chunks:{len(list_texts)} chunk: {list_texts[0]}")

###---------------------------------------------------------------
##  Crear embeddings (Spanish)
###---------------------------------------------------------------
spanish_embed = OllamaEmbeddings(model='jina/jina-embeddings-v2-base-es')
vectors = spanish_embed.embed_documents(list_texts)
dimension = len(spanish_embed.embed_query("hola"))


# print(vectors[0])

###---------------------------------------------------------------
##  Crear Pinecone index
###---------------------------------------------------------------

index_name = "spanish-test-index02"
pc = Pinecone(api_key="your_pinecone_apikey")

# Intentar conectarse al índice existente
try:
    index = pc.Index(index_name)
    print(f"Conectado al índice existente '{index_name}'.")
except Exception as e:
    print(f"No se pudo conectar al índice: {e}")
    # Solo crear si estamos seguros de que no existe
    print(f"Creando índice '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    index = pc.Index(index_name)
    print(f"Índice '{index_name}' creado y conectado.")

# Initialize vector store y agregar los textos(pdfs)
index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index, embedding=spanish_embed)
vector_store.add_documents(documents=splitted_documents)

# print("Documents added to Pinecone index.")

###---------------------------------------------------------------
##  Prompt y respuesta del chatbot con LLM
###---------------------------------------------------------------

# Setup RAG con LLM
retriever = vector_store.as_retriever(search_type="similarity")
llm = ChatOllama(model="llama3")
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)


template = """
Eres un asistente útil que SIEMPRE responde en español de Tienda Pago.
Usa únicamente la información de los documentos recuperados.
Si son consultas cortas busca palabras relacionadas ,raiz o sinonimos.
Si la información no está presente, responde: lo siento, aun no se encuentra en mi base de conocimiento.

Pregunta:
{question}

Documentos recuperados:
{context}

Respuesta (en español):
"""

prompt_es = PromptTemplate(
    template=template,
    input_variables=["question", "context"]
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_es}
)

def chat_with_llm(message, history):
    # respusta del modelo RAG
    result = qa.invoke({"query": message})
    clean_response = re.sub(r'<think>.*?</think>', '', result['result'], flags=re.DOTALL).strip()

    # print("SOURCE DOCS:", result["source_documents"])  # Debug
    return clean_response


# interface util
gr.ChatInterface(
    fn = chat_with_llm,
    type = "messages"
    ).launch()
