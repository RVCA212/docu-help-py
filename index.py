from flask import Flask, jsonify, request
from os import environ
import json
from langchain_openai import OpenAIEmbeddings
from pinecone_text.sparse import SpladeEncoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone

app = Flask(__name__)

OPENAI_API_KEY = environ.get('OPENAI_API_KEY')
PINE_API_KEY = environ.get('PINE_API_KEY')

embed = OpenAIEmbeddings(
    model='text-embedding-3-small',
    openai_api_key=OPENAI_API_KEY,
    dimensions=768
)
index_name = 'splade'
pc = Pinecone(api_key=PINE_API_KEY)
index = pc.Index(index_name)
splade_encoder = SpladeEncoder()

@app.route("/")
def welcome():
    return "Hello world"

@app.route("/home")
def home():
    return jsonify({"msg": "Home page here"})

@app.route("/retrieve", methods=["POST"])
def retrieve():
    try:
        requestBody = request.get_json()
        q = requestBody['question']
        namespace_name = requestBody.get('namespace_name', 'default_namespace')  # Provide a default namespace if not specified

        retriever = PineconeHybridSearchRetriever(
            embeddings=embed,
            sparse_encoder=splade_encoder,
            index=index,
            namespace=namespace_name,
            top_k=4
        )
        
        results = retriever.invoke(q)

        json_response = [
            {
                "page_content": doc.metadata['parent_content'],
                "source": doc.metadata['source']
            } for doc in results
        ]

        return jsonify(json_response)
    except KeyError as e:
        return jsonify({"error": f"Missing key in request: {str(e)}"}), 400



