import openai
import os
import secrets

OPENAI_API_KEY = secrets.OPENAI
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

from llama_index.agent import OpenAIAssistantAgent

from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)

from llama_index.tools import QueryEngineTool, ToolMetadata


docs = SimpleDirectoryReader(
    input_files=["/content/drive/MyDrive/RajMehta_resume_SWE.pdf",
                  "/content/drive/MyDrive/Cornell/Business Fundamentals/TECHIE_5310__Business_Fundamentals_-_Khaire___Gjondrekaj__Fall_2023__10812911121746/Financial_Accounting_Reading__Analyzing_Financial_Statements.pdf"]
).load_data()

index = VectorStoreIndex.from_documents(docs)
index.storage_context.persist(persist_dir="./storage/index")




"""### 2. Create Engine"""

engine = index.as_query_engine(similarity_top_k=3)

query_engine_tools = [
    QueryEngineTool(
        query_engine=engine,
        metadata=ToolMetadata(
            name="index",
            description=(
                "Provides information about various documents"
                "Focus on given straightforward, accurate answers"
            ),
        ),
    ),
]

"""### 3. Now the query engine tools is being created, let's try it out with this tools."""

agent = OpenAIAssistantAgent.from_new(
    name="Document Answer",
    instructions="You are a QA assistant designed to retrieve relevant information from documents and give straightforward, precise answers.",
    tools=query_engine_tools,
    verbose=True,
    run_retrieve_sleep_time=1.0,
)

response = agent.chat("What was the gross margin of Prada")

response.response

for source in response.source_nodes:
  print(source.metadata)
  print(source.text)