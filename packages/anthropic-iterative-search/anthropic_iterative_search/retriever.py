from langchain.tools import tool
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
import time


PINE_API_KEY = ""

index_name='splade'

namespace_name='Langchain'

pc = Pinecone(api_key=PINE_API_KEY)

index = pc.Index(index_name)
time.sleep(0.5)

embed = OpenAIEmbeddings(
    model='text-embedding-3-small',
    dimensions = 768
)
print("embeddings loaded")

from pinecone_text.sparse import SpladeEncoder

splade_encoder = SpladeEncoder()

from langchain_community.retrievers import PineconeHybridSearchRetriever

text_field = "context"

retriever = PineconeHybridSearchRetriever(
    embeddings=embed, sparse_encoder=splade_encoder, index=index, namespace=namespace_name, top_k=1
)

# This is used to tell the model how to best use the retriever.

search_description = """You will be asked a question by a human user. You have access to the following tool to help answer the question. <tool_description> Documentation Search Engine Tool * The search engine will exclusively search over software Documentation for pages similar to your query. Use this tool whenever there is no link present in the query.  It returns for each page its title and full page content. Use this tool if you want to get up-to-date and comprehensive information on a topic to help answer queries. Queries should be as atomic as possible -- they only need to address one part of the user's question. For example, if the user's query is "what is the color of a basketball?", your search query should be "basketball". Here's another example: if the user's question is "Who created the first neural network?", your first query should be "neural network". As you can see, these queries are quite short. Think keywords, not phrases. * At any time, you can make a call to the search engine using the following syntax: <search_query>query_word</search_query>. * You'll then get results back in <search_result> tags.</tool_description>"""  # noqa: E501

# This should be the same as the function name below
RETRIEVER_TOOL_NAME = "search"


web_tool_description = """You will be asked a question by a human user. You have access to the following tool to help answer the question. <tool_description> Internet Search Engine Tool * Use only if a link is present in the user's prompt. If a url is presented by the user, use this tool to search the web for the url, if not, use the other tool.  Then, inspect the content retrieved from the web to answer the user's question. It returns for each page its title and full page content. Use this tool if you want to get up-to-date and comprehensive information on a topic to help answer queries. Queries should be as atomic as possible -- they only need to address one part of the user's question. For example, if the user's query is "Can I ", your search query should be "basketball". Here's another example: if the user's question is "Who created the first neural network?", your first query should be "neural network". As you can see, these queries are quite short. Think keywords, not phrases. * At any time, you can make a call to the search engine using the following syntax: <search_query>query_word</search_query>. * You'll then get results back in <search_result> tags.</tool_description>"""  # noqa: E501



@tool
def search(query):
    """Searching the Documentation."""
    return retriever.get_relevant_documents(query)


# search = DuckDuckGoSearchResults()


# @tool
# def browse(query):
#     """Browsing the Web"""
#     return search.run(query)
