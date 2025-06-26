import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,TavilySearchResults
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
load_dotenv()

## Arxiv and wikipedia tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wikipedia = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

tavily_api=os.getenv('TAVILY_API_KEY')
search = TavilySearchResults(  max_results=1,
    include_answer=True,
    include_raw_content=True,
    include_images=False,)


st.title("🔎 LangChain - Chat with search")
st.markdown("""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).

""")

##Sidebar for API Key
st.sidebar.title('Configure API')
groq_api_key = st.sidebar.text_input('Enter Your Groq  API Key:',type='password')

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant",
         "content":"Hi,I'm a chatbot who can search the web.How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt := st.chat_input(placeholder="Enter your Query Here"):
    st.session_state.messages.append({"role":"human",
                                      "content":prompt})
    st.chat_message("human").write(prompt)

    llm =ChatGroq(api_key=groq_api_key,model='llama3-8b-8192',streaming=True)
    tools = [arxiv,wikipedia,search]

    search_agent = initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True)


    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({'role':'assistant','content':response})
        st.write(response)