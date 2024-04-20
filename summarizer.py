import streamlit as st
import os

from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_community.llms import HuggingFaceHub

os.environ["HUGGINGFACEHUB_API_TOKEN"] = # put api key here

def generate_response(txt):
    repo_id = "google/flan-t5-xxl"

    llm = HuggingFaceHub(
        repo_id=repo_id, model_kwargs={"temperature": 0.5}
    )
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)
    docs = [Document(page_content=t) for t in texts]
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    return chain.run(docs)

st.set_page_config(page_title='🦜🔗 Text Summarization App')
st.title('🦜🔗 Text Summarization App')

txt_input = st.text_area('Enter your text', '', height=200)

result = []
with st.form('summarize_form', clear_on_submit=True):
    submitted = st.form_submit_button('Submit')
    if submitted:
        with st.spinner('Calculating...'):
            response = generate_response(txt_input)
            result.append(response)
if len(result):
    st.info(response)
