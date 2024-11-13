import streamlit as st # type: ignore
import os
from dotenv import load_dotenv

import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from PyPDF2 import PdfReader

def create_chain(text):
    text_splitter = RecursiveCharacterTextSplitter()
    text = text_splitter.split_text(text=text)
        
    # Initialize the embeddings model
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY,model="text-embedding-ada-002")
        
    # create vectorstore using FAISS
    vectorstore = FAISS.from_texts(text, embeddings)

    # create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    #Initialize llm
    llm = OpenAI(api_key=API_KEY)
        
    template = """"
            You are an assistant for question-answering tasks.
            You will respond based on the given context only.
            Do not hallucinate, if you do not know the answer respond politely saying I dont know.
            Use the provided context only to answer the following question:

            <context>
            {context}
            </context>

            Question: {input}
            """
    prompt = ChatPromptTemplate.from_template(template)
    doc_chain = create_stuff_documents_chain(llm, prompt)
        
    # create retrieval chain
    chain = create_retrieval_chain(retriever, doc_chain)
    return(chain)

def file_processing(pdfs):
    text=""
    for file in pdfs:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text = text + page.extract_text()
    return(text)

def enable_chain():
    st.session_state['process_chain'] = 1    

def disable_chain():
    st.session_state['process_chain'] = 0 

if __name__ == "__main__":
    load_dotenv()
    API_KEY = os.getenv("OPENAI_API_KEY")
    
    if 'process_chain'not in st.session_state:
        st.session_state['process_chain'] = 0
    
    tab1, tab2 = st.tabs(["PDF Upload","QnA"])
    
    tab1.subheader("Document Processing")
    with tab1:
        pdfs = tab1.file_uploader("Upload files", accept_multiple_files=True, on_change=enable_chain)
        text=""
        text = file_processing(pdfs)
        #tab1.write(len(text))
        #tab1.write(np.random. randint(1, 50))
        #tab1.write(st.session_state['process_chain'])
        if pdfs:
            if (len(text) !=0) and (st.session_state['process_chain'] !=0):      
                chain = create_chain(text)
                st.session_state['chain'] = chain
        else:
            if 'chain' in st.session_state:
                del st.session_state['chain']    
    
    tab2.subheader("Chat with your PDFs here")
    with tab2:
        with st.form(key='user_form', clear_on_submit=True):
            user_input = st.text_input(label="Question:", placeholder="Ask about your PDF", key='input')
            submit_button = st.form_submit_button(label='Send', on_click=disable_chain)
            tab2.write(st.session_state['process_chain'])
            if submit_button and user_input:
                if 'chain' in st.session_state:
                    with st.spinner('Generating response...'):
                        chain = st.session_state['chain']
                        response = chain.invoke({"input": user_input})
                    st.write(response["answer"])
                else:
                    st.write("RAG chain not built, upload the PDFS and comeback.")