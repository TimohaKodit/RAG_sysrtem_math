from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from itertools import batched
import time
import os
from dotenv import load_dotenv

load_dotenv()






file_path = 'data/p.pdf'
loader = PyPDFLoader(file_path)
docs = loader.load()


embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    api_key=os.getenv("GOOGLE_API_KEY")
)

if __name__ == "main":
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


    vector_store = Chroma(embedding_function=embeddings, persist_directory='./db')
    pages = text_splitter.split_documents(docs)
    batch_size = 50

    for batch in batched(pages, batch_size):
        vector_store.add_documents(batch)
        time.sleep(10)
else:
    vector_store = Chroma(embedding_function=embeddings, persist_directory='./db')




retriever = vector_store.as_retriever(search_kwargs={"k": 3})