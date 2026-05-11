from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from itertools import batched
import time
import os
from dotenv import load_dotenv
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import ChatCohere, CohereRerank
import asyncio

load_dotenv()









embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    api_key=os.getenv("GOOGLE_API_KEY")
)
vector_store = Chroma(embedding_function=embeddings, persist_directory='./db')
def dc(file_path, user_id):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # unique_sources = {doc.metadata['source'] for doc in docs}

    # print(f"Найдено файлов: {len(unique_sources)}")
    # for source in unique_sources:
    #     print(f"Файл в обработке: {source}")



    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    

    
    pages = text_splitter.split_documents(docs)
    pages = [doc for doc in pages if doc.page_content.strip()]
    for doc in pages:
        doc.metadata['user_id'] = int(user_id)
    batch_size = 1
    for batch in batched(pages, batch_size):
        vector_store.add_documents(batch)

    retriever = vector_store.as_retriever(search_kwargs={'filter': {'user_id': user_id}, "k": 10})

    compressor = CohereRerank(model='rerank-multilingual-v3.0', cohere_api_key=os.getenv("COHERE_API_KEY"), top_n=10)
    compression = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )


    # for source in unique_sources:
    # # Просим базу вернуть ID записей, где метаданные source совпадают с нашим файлом
    #     existing_docs = vector_store.get(where={'source': source})
    
    # # Если список IDs пустой, значит файла в базе нет
    #     if len(existing_docs['ids']) == 0:
    #         print(f"❌ Файла {source} нет в базе. Начинаю загрузку...")
        
    #     # Тут ты фильтруешь только те страницы из docs, которые относятся к этому файлу
    #         file_docs = [doc for doc in docs if doc.metadata['source'] == source]
        
    #     # Режем их на чанки
    #         chunks = text_splitter.split_documents(file_docs)
        
    #     # И добавляем в базу (тут твой цикл с batched)
    #     # vector_store.add_documents(chunks)
    #         for batch in batched(chunks, batch_size):
            
    #             vector_store.add_documents(batch)
    #             time.sleep(10)
    #     else:
    #         print(f"✅ Файл {source} уже существует. Пропускаю.")





    






