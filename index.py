from flask import Flask, jsonify, request
app = Flask(__name__)
from os import environ
OPENAI_API_KEY=environ.get('OPENAI_API_KEY')
from os import environ
PINE_API_KEY=environ.get('PINE_API_KEY')

@app.route("/")
def welcome():
    return "Hello world"

@app.route("/home")
def home():
    return jsonify({"msg": "Home page here"})


@app.route("/retrieve", methods=["POST"])
def retrieve():
    requestBody = request.get_json()
    q = requestBody['question']
    namespace_name = requestBody['namespace_name']

    # Import necessary modules
    import time
    from langchain_openai import OpenAIEmbeddings
    from pinecone_text.sparse import SpladeEncoder
    from langchain_community.retrievers import PineconeHybridSearchRetriever
    from pinecone import Pinecone
    import json

    embed = OpenAIEmbeddings(
        model='text-embedding-3-small',
        openai_api_key=OPENAI_API_KEY,
        dimensions=768
    )

    index_name = 'splade'

    pc = Pinecone(api_key=PINE_API_KEY)
    
    index = pc.Index(index_name)
    # Wait a moment for connection
    time.sleep(0.1)
    
    splade_encoder = SpladeEncoder()
    
    text_field = "text"
    
    retriever = PineconeHybridSearchRetriever(
        embeddings=embed, sparse_encoder=splade_encoder, index=index, namespace=namespace_name, top_k=4
    )

    results = retriever.invoke(q)

    # Structure the results into a list of dictionaries with desired keys
    json_response = [
        {
            "page_content": doc.page_content,
            "source": doc.metadata['source']  # Make sure this line is exactly as shown
        } for doc in results
    ]

    return jsonify(json_response)


