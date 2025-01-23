# langchain modules
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import JSONLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveJsonSplitter
from langchain import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import OpenAIEmbeddings
# general modules
import os
from csv_to_json import csv_to_json
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv


class ChatBot:
    # getting API keys
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    # embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large"
    )

    # initialising JSON loader and chunking it
    csv_to_json('CARS_1.csv')
    loader = JSONLoader(
        file_path="data.json",
        jq_schema=".",
        text_content=False,
        json_lines=True
    )
    
    data = loader.load()

    splitter = RecursiveJsonSplitter(max_chunk_size=300)
    json_chunks = splitter.split_json(json_data=data)

    # Initialize pinecone client
    pinecone.init(
        api_key=os.getenv('PINECONE_API_KEY'),
        environment='gcp-starter'
    )

    # Defining Index Name
    index_name = "caristification"

    # Checking Index
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name, metric="cosine", dimension=768)
        docsearch = Pinecone.from_documents(data, embeddings, index_name=index_name)
    else:
        docsearch = Pinecone.from_existing_index(index_name, embeddings)

    # Initialising LLM
    llm = ChatOpenAI(model="gpt-4o-mini")

    # Prompt Engineering
    template = """
    You are a bot who is going to suggest car models. These Human will describe their lifestyle and will be giving Passengers Capacity they want, budget they have and what type of car they want.
    Use the piece of context to suggest them car models. Generate atleast 3 car models at one time.
    If they dont like the car models, then suggest them with more car models.
    The Answer Structure should be in the following way:

    1. The Name of the Car
       Starting Price: The Starting ex showroom price the car has
       Ending Price/Top Model Price: Car's Top model's ex showroom price
       Fuel Type: Fuel Type that the car has
       Seating Capacity: Seating Capacity the car has
       Body Type: Body Type of the Car
       Rating: The Rating the car has
       Fuel Tank: The Fuel Tank Capacity the car has
    
       Engine Specifications
       Torque: Torque Produced by the car
       Power: The Horsepower Produced by the car
       Number of Cylinders: Number of Cylinder the car has
       Transmission: Transmission the car has
       Maximum RPM: The Maximum RPM the car can achieve

    use the same structure for the other 2 recommendations 

    Context: {context}
    Details: {details}
   
    """

    prompt = PromptTemplate(
        template=template,
        input_variable=["context", "question"]
    )

    rag_chain = (
        {"context": docsearch.as_retriever(), "details": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
