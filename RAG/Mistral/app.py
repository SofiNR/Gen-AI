import streamlit as st
import os
from dotenv import load_dotenv

import numpy as np
# to read from PDFs
from PyPDF2 import PdfReader
# for chunking the texts from PDFs
from langchain.text_splitter import RecursiveCharacterTextSplitter
# for embedding
from langchain_mistralai.embeddings import MistralAIEmbeddings
# for vector store
from langchain_community.vectorstores import Chroma
# chat model to use in the RAG
from langchain_mistralai.chat_models import ChatMistralAI
# to use custom prompts
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Convert loaded documents into strings by concatenating their content
# and ignoring metadata
# the result of similarity search are list of documents carrying metadata as well
# to get rid of those metadata and work on only the actual data chunks
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_chain(text):
    text_splitter = RecursiveCharacterTextSplitter()
    text = text_splitter.split_text(text=text)
        
    # Initialize the embeddings model
    embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=API_KEY)
        
    # create vectorstore using ChromaDB
    vectorstore = Chroma.from_texts(
                            texts=text,
                            embedding=embeddings
                        )

    #Initialize llm
    llm = ChatMistralAI(model="mistral-large-latest", mistral_api_key=API_KEY)
        
    RAG_TEMPLATE = """
    You are an assistant for question-answering tasks. Use the following context to answer the question. If you don't know the answer, politely say that you don't know. Use three sentences maximum and keep the answer concise.

    <context>
    {context}
    </context>

    Answer the following question:

    {question}"""
    
    rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    
    rag_chain = (
        RunnablePassthrough.assign(context=lambda input: format_docs(input["context"]))
        | rag_prompt
        | llm
        | StrOutputParser()
    )    
    
    return(rag_chain, vectorstore)

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
    API_KEY = os.getenv("MISTRAL_API_KEY")
    if 'process_chain'not in st.session_state:
        st.session_state['process_chain'] = 0
    
    st.title("RAG using LangChain-Mistral-ChromaDB")
    
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
                chain, vectorstore = create_chain(text)
                st.session_state['chain'] = chain
                st.session_state['vectorstore'] = vectorstore
        else:
            if 'chain' in st.session_state:
                del st.session_state['chain']
                del st.session_state['vectorstore']    
    
    tab2.subheader("Chat with your PDFs here")
    with tab2:
        with st.form(key='user_form', clear_on_submit=True):
            user_input = st.text_input(label="Question:", placeholder="Ask about your PDF", key='input')
            submit_button = st.form_submit_button(label='Send', on_click=disable_chain)
            #tab2.write(st.session_state['process_chain'])
            if submit_button and user_input:
                if 'chain' in st.session_state:
                    with st.spinner('Generating response...'):
                        chain = st.session_state['chain']
                        vectorstore = st.session_state['vectorstore']
                        docs = vectorstore.similarity_search(user_input)
                        if len(docs) > 0:
                            response = chain.invoke({"context": docs, "question": user_input})
                            st.write(response)
                        else:
                            st.write("Something went wrong: No similar vector found")                    
                else:
                    st.write("RAG chain not built, upload the PDFS and comeback.")