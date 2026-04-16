
from core import rag_chain
import streamlit as st 
from langchain_core.messages import HumanMessage, AIMessage




st.title("RAG-SYSTEM")



prom = st.chat_input("Say")



if "messages" not in st.session_state:
    st.session_state.messages = []


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prom:

    st.session_state.messages.append({"role": "user", "content": prom})
    with st.chat_message('user'):
        st.write(prom)

    with st.chat_message('assistant'):
        placeholder = st.empty()
        ful = ""

        for chunk in rag_chain.stream({
            "input": prom,
            "chat_history": st.session_state.chat_history
        }):
            if 'answer' in chunk:
                ful += chunk['answer']
                placeholder.markdown(ful + "▌")
        placeholder.markdown(ful)



    st.session_state.messages.append({"role": "assistant", "content": ful})

    st.session_state.chat_history.extend([
        HumanMessage(content=prom),
        AIMessage(content=ful)
    ])
    