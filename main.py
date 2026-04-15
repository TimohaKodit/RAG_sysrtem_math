from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from itertools import batched
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain



file_path = 'p.pdf'
loader = PyPDFLoader(file_path)
docs = loader.load()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    api_key=""
)




text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

pages = text_splitter.split_documents(docs)
vector_store = Chroma(embedding_function=embeddings, persist_directory='./db')
batch_size = 50

# for batch in batched(pages, batch_size):
#     vector_store.add_documents(batch)
#     time.sleep(10)

llm = ChatGoogleGenerativeAI(
    model= "gemini-2.5-flash",
    api_key=""

)

simillar_docs = vector_store.similarity_search("Циклы в Python")
# for i in simillar_docs:
#     print(i.page_content[:500])

retriever = vector_store.as_retriever(search_kwargs={"k": 3})


system_prompt = (
    "Ты полезный ассистент, который обучает языку Python"
    "Исппользуй ТОЛЬКО фрагменты, из контекста,чтобы отвечать пользователю"
    "Если информации нет в контексте, просто скажи что не знаешь"
    "Контекст:\n{context}"
)


prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)


rag_chain = create_retrieval_chain(retriever, question_answer_chain)
msg = rag_chain.stream({"input": "Какие типы данных есть в python"})


for i in msg:
    if "answer" in i:
        print(i["answer"], end="", flush=True)
