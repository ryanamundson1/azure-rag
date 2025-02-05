import os
import openai
from openai import OpenAI
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import AzureSearch
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
import pandas as pd
import kagglehub

app = FastAPI()

openai.api_base = os.getenv("OPENAI_API_BASE")  # Your Azure OpenAI resource's endpoint value.
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_type = "azure"
openai.api_version = "2023-05-15" 

#embeddings = OpenAIEmbeddings(deployment="demo-embedding", chunk_size=1)
# Get the data for embeddings
# Download latest version
path = kagglehub.dataset_download("cryptexcode/mpst-movie-plot-synopses-with-tags")

print("Path to dataset files:", path)

df = pd.read_csv(f'{path}/mpst_full_data.csv')
df = df[df['plot_synopsis'].notna()] # remove any NaN values as it blows up serialization
data = df.sample(1000).to_dict('records') # Get only 700 records. More records will make it slower to index
len(data)

qdrant = QdrantClient(":memory:") 

# Create collection to store wines
encoder = SentenceTransformer('all-MiniLM-L6-v2') 

qdrant.recreate_collection(
    collection_name="movie_plots",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(), # Vector size is defined by used model
        distance=models.Distance.COSINE
    )
)

qdrant.upload_points(
    collection_name="movie_plots",
    points=[
        models.PointStruct(
            id=idx,
            vector=encoder.encode(doc["plot_synopsis"]).tolist(),
            payload=doc,
        ) for idx, doc in enumerate(data) # data is the variable holding all the wines
    ]
)


client = OpenAI(
    base_url="http://localhost:1234/v1", # "http://<Your api-server IP>:port"
    api_key = "sk-no-key-required"
)
# Connect to Azure Cognitive Search
#acs = AzureSearch(azure_search_endpoint=os.getenv('SEARCH_SERVICE_NAME'),
#                 azure_search_key=os.getenv('SEARCH_API_KEY'),
#                 index_name=os.getenv('SEARCH_INDEX_NAME'),
#                 embedding_function=embeddings.embed_query)

class Body(BaseModel):
    query: str


@app.get('/')
def root():
    return RedirectResponse(url='/docs', status_code=301)


@app.post('/ask')
def ask(body: Body):
    """
    Use the query parameter to interact with the Azure OpenAI Service
    using the Azure Cognitive Search API for Retrieval Augmented Generation.
    """
    search_result = search(body.query)
    chat_bot_response = assistant(body.query, search_result)
    return {'response': chat_bot_response}



def search(query):
    """
    Send the query to Azure Cognitive Search and return the top result
    """
    #docs = acs.similarity_search_with_relevance_scores(
    #    query=query,
    #    k=5,
    #)
    #result = docs[0][0].page_content
    result = qdrant.search(
        collection_name="movie_plots",
        query_vector=encoder.encode(query).tolist(),
        limit=3
    )
    print(result)
    return result


def assistant(query, context):
    messages=[
        # Set the system characteristics for this chat bot
        {"role": "system", "content": "Assistant is a chatbot that helps you find movies based on plots."},

        # Set the query so that the chatbot can respond to it
        {"role": "user", "content": str(query)},

        # Add the context from the vector search results so that the chatbot can use
        # it as part of the response for an augmented context
        {"role": "assistant", "content": str(context)}
    ]

    #response = openai.ChatCompletion.create(
    #    engine="demo-alfredo",
    #    messages=messages,
    #)
    completion = client.chat.completions.create(
        model="LM_STUDIO_DEEPSEEK",
        messages=messages
    )
    return completion.choices[0].message