import streamlit as st
import os
import config
import PyPDF2
import io
from llama_index.agent import OpenAIAssistantAgent
from llama_index import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.tools import QueryEngineTool, ToolMetadata


# Set up OpenAI API Key
OPENAI_API_KEY = config.OPENAI
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def save_uploaded_files(uploaded_files):
    saved_paths = []
    for file in uploaded_files:
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())
        saved_paths.append(file.name)
    return saved_paths

# Function to load documents and create an index
def load_documents_and_create_index(document_paths):
    try:
        storage_context = StorageContext.from_defaults(
            persist_dir="./storage/lyft"
        )
        lyft_index = load_index_from_storage(storage_context)

        storage_context = StorageContext.from_defaults(
            persist_dir="./storage/uber"
        )
        uber_index = load_index_from_storage(storage_context)

        index_loaded = True
    except:
        index_loaded = False
    if not index_loaded:
        docs = SimpleDirectoryReader(input_files=document_paths).load_data()
        index = VectorStoreIndex.from_documents(docs)
        index.storage_context.persist(persist_dir="./storage/index")
    return index

# Function to create a query engine
def create_engine(index):
    engine = index.as_query_engine(similarity_top_k=3)
    query_engine_tools = [
        QueryEngineTool(
            query_engine=engine,
            metadata=ToolMetadata(
                name="index",
                description="Provides information about various documents"
            ),
        ),
    ]
    return query_engine_tools

# Function to initialize the agent
def initialize_agent(query_engine_tools):
    agent = OpenAIAssistantAgent.from_new(
        name="Document Answer",
        instructions='''You are a QA assistant designed to retrieve relevant information from documents and give straightforward, 
                     precise answers. If you do not know the answer based on the context do not give a false answer.''',
        tools=query_engine_tools,
        verbose=True,
        run_retrieve_sleep_time=1.0,
    )
    return agent

# Streamlit UI Components
st.title("Document Chat Interface")

with st.sidebar:
    uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True, type=['pdf'])
    if uploaded_files:
        document_paths = save_uploaded_files(uploaded_files) # Save uploaded files
        index = load_documents_and_create_index(document_paths)
        query_engine_tools = create_engine(index)
        agent = initialize_agent(query_engine_tools)
        st.success("Documents uploaded and indexed successfully!")

# Main column for chat textbox

user_query = st.text_input("Ask a question")

if st.button("Submit Query") and user_query:
    response = agent.chat(user_query)
    not_found = False
    try: 
        st.write(response.sources[0].raw_output.response)
        not_found = True

    except:
        None
    st.write("Response:", response.response)
    st.write("Source Information")
    try:
    
        source = response.source_nodes[0]
        st.write("Source Document:", source.metadata)
        st.write("Excerpt:", source.text)
    except:
        st.write("Sources for this are not found")


# Run the Streamlit app with `streamlit run your_script.py`
