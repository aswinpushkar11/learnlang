import os
from dotenv import load_dotenv
#Import Libraries
from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
#Langsmith tracking
os.environ["LangChain_API"] = os.getenv("LangChain_API")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LangChain_Project"]=os.getenv("LangChain_Project")
prompt = ChatPromptTemplate.from_messages(
[
    ("system","You are a physicist ,only answer related question and just deny answers for physics unrelated questions"),
    ("user","Question:{question}")
]
)
st.title("Gemma Model")
input_text = st.text_input("Ask me anthing Physics")
#Ollama model
llm = Ollama(model='gemma3:1b')
o_parser = StrOutputParser()
chain = prompt | llm | o_parser
if input_text:
    st.write(chain.invoke({"question":input_text}))