
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from ingest import vector_store
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import ChatCohere, CohereRerank

import os
from dotenv import load_dotenv

load_dotenv()



contextualixe_q_system_prompt= (
    "Возьми историю чата за последнии вопросы"
    "И переформулируй вопрос исходя из истории сообщений"
    "НЕ ОТВЕЧАЙ на вопрос, только перефразируй его, если нужно."
)


system_prompt = (
    "Ты полезный ассистент, который обучает языку Python"
    "Исппользуйфрагменты из контекста,чтобы отвечать пользователю"
    "Отвечай подробно и развернуто, чтобы твой ответ был максимально понятен"
    "Если информации нет в контексте, просто скажи что не знаешь"
    "Контекст:\n{context}"
)



contextulize = ChatPromptTemplate.from_messages(
    [
        ("system", contextualixe_q_system_prompt),
        MessagesPlaceholder('chat_history'),
        ('human', "{input}")
    ]
)


prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)





def get_rag_chain(user_id):

    retriever = vector_store.as_retriever(search_kwargs={'filter': {'user_id': user_id}, 'k': 10})
    compressor = CohereRerank(model='rerank-multilingual-v3.0', cohere_api_key=os.getenv("COHERE_API_KEY"), top_n=10)
    compression = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )

    history_retriever = create_history_aware_retriever(llm, compression, contextulize)
    rag_chain = create_retrieval_chain(history_retriever, question_answer_chain)

    return rag_chain




