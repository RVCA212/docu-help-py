from flask import Flask, jsonify, request
app = Flask(__name__)
from os import environ
OPENAI_API_KEY=environ.get('OPENAI_API_KEY')
from os import environ
ANTHROPIC_API_KEY=environ.get('ANTHROPIC_API_KEY')

@app.route("/")
def welcome():
    return "Hello world"

@app.route("/home")
def home():
    return jsonify({"msg": "Home page here"})

@app.route("/chatting/<string:namespace_name>/<string:q>")
def chat(namespace_name, q):
    # self.send_response(200)
    # self.send_header('Content-type','text/plain')
    # self.end_headers()
    import requests
    from langchain_openai import OpenAIEmbeddings
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough
    from langchain_anthropic import ChatAnthropic
    # Import Module
    import json


    
    # Define the API retrieval function
    def retriever(query):
        url = f"http://api.rag.pro/getModel/Langchain/{query}?top_k=1"
        response = requests.get(url)
        return response.json()

    # RAG prompt
    template = """You are an expert software developer who specializes in APIs.  Answer the user's question based only on the following context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    model = ChatAnthropic(temperature=0, anthropic_api_key=ANTHROPIC_API_KEY, model_name="claude-3-haiku-20240307")

    def format_docs_with_sources(docs):
        # This function formats the documents and includes their sources.
        formatted_docs = []
        for doc in docs:
            content = doc.page_content
            source = doc.metadata.get('source', 'Unknown source')
            formatted_docs.append(f"{content}\nSource: {source}")
        return "\n\n".join(formatted_docs)

    rag_chain = (
        RunnablePassthrough.assign(context=(lambda x: format_docs_with_sources(x["context"])))
        | prompt
        | model
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain)

    # response = rag_chain_with_source.invoke("Query goes here")

    # response = rag_chain_with_source.invoke("what is langchain")

    response = rag_chain_with_source.invoke(q)

    answer = response['answer']  # Extracting the 'answer' part

    sources = [doc.metadata['source'] for doc in response['context']]

    # formatted_response = f"Answer: {answer}\n\nSources:\n" + "\n".join(sources)

    json_response = {"answer": answer, "sources": sources}

    # print(formatted_response)
    # self.wfile.write('Hello, world!'.encode('utf-8'))
    
    # self.wfile.write(formatted_response.encode('utf-8'))
    # return
    # return jsonify({'hello': 'world'})
    return jsonify(json_response)


@app.route("/chat", methods=["POST"])
def chatting():
    requestBody = request.get_json()
    q = requestBody['question']
    namespace_name = requestBody['namespace_name']
    # self.send_response(200)
    # self.send_header('Content-type','text/plain')
    # self.end_headers()
    import requests
    from langchain_openai import OpenAIEmbeddings
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough
    from langchain_anthropic import ChatAnthropic
    # Import Module
    import json


    def retriever(query):
        url = f"http://api.rag.pro/getModel/Langchain/{query}?top_k=1"
        response = requests.get(url)
        return response.json()


    # RAG prompt
    template = """You are an expert software developer who specializes in APIs.  Answer the user's question based only on the following context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # RAG
    model = ChatAnthropic(temperature=0, anthropic_api_key=ANTHROPIC_API_KEY, model_name="claude-3-haiku-20240307")

    def format_docs_with_sources(docs):
        # This function formats the documents and includes their sources.
        formatted_docs = []
        for doc in docs:
            content = doc.page_content
            source = doc.metadata.get('source', 'Unknown source')
            formatted_docs.append(f"{content}\nSource: {source}")
        return "\n\n".join(formatted_docs)

    rag_chain = (
        RunnablePassthrough.assign(context=(lambda x: format_docs_with_sources(x["context"])))
        | prompt
        | model
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain)

    # response = rag_chain_with_source.invoke("Query goes here")

    # response = rag_chain_with_source.invoke("what is langchain")

    response = rag_chain_with_source.invoke(q)

    answer = response['answer']  # Extracting the 'answer' part

    sources = [doc.metadata['source'] for doc in response['context']]

    # formatted_response = f"Answer: {answer}\n\nSources:\n" + "\n".join(sources)

    json_response = {"answer": answer, "sources": sources}

    # print(formatted_response)
    # self.wfile.write('Hello, world!'.encode('utf-8'))
    
    # self.wfile.write(formatted_response.encode('utf-8'))
    # return
    # return jsonify({'hello': 'world'})
    return jsonify(json_response)
