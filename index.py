from flask import Flask, jsonify, request
app = Flask(__name__)
from os import environ
OPENAI_API_KEY=environ.get('OPENAI_API_KEY')
from os import environ
ANTHROPIC_API_KEY=environ.get('ANTHROPIC_API_KEY')
from os import environ
PINE_API_KEY=environ.get('PINE_API_KEY')

@app.route("/")
def welcome():
    return "Hello world"

@app.route("/home")
def home():
    return jsonify({"msg": "Home page here"})

@app.route("/chatting/<string:q>")
def chat(q):
    # self.send_response(200)
    # self.send_header('Content-type','text/plain')
    # self.end_headers()
    from pinecone import Pinecone as PineconeClient
    import time
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Pinecone
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough
    from langchain_anthropic import ChatAnthropic
    # Import Module
    import json

    
    model_name = 'text-embedding-ada-002'

    embed = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=OPENAI_API_KEY
    )

#    

    index_name='docu-help'

    namespace_name='Langchain'

    # pc = Pinecone(api_key=PINE_API_KEY)
    pc = PineconeClient(api_key=PINE_API_KEY)

    index = pc.Index(index_name)
    # wait a moment for connection
    time.sleep(1)

    index.describe_index_stats()


    text_field = "text"

    # switch back to normal index for langchain
    index = pc.Index(index_name)

    vectorstore = Pinecone(
        index, embed.embed_query, text_field, namespace=namespace_name
    )
    retriever = vectorstore.as_retriever()


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
    # self.send_response(200)
    # self.send_header('Content-type','text/plain')
    # self.end_headers()
    from pinecone import Pinecone as PineconeClient
    import time
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Pinecone
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough
    from langchain_anthropic import ChatAnthropic
    # Import Module
    import json

    model_name = 'text-embedding-ada-002'

    embed = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=OPENAI_API_KEY
    )

    index_name='docu-help'

    namespace_name='Langchain'

    # pc = Pinecone(api_key=PINE_API_KEY)
    pc = PineconeClient(api_key=PINE_API_KEY)

    index = pc.Index(index_name)
    # wait a moment for connection
    time.sleep(1)

    index.describe_index_stats()


    text_field = "text"

    # switch back to normal index for langchain
    index = pc.Index(index_name)

    vectorstore = Pinecone(
        index, embed.embed_query, text_field, namespace=namespace_name
    )
    retriever = vectorstore.as_retriever()


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
