from flask import Flask, jsonify, request
from langchain.agents import tool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import Tool
from langchain import hub
from langchain_anthropic import ChatAnthropic
from langchain.agents.output_parsers import XMLAgentOutputParser
from langchain.agents import AgentExecutor
import requests
import os

app = Flask(__name__)

# Define the schema for the API retrieval tool input
class APIQueryInput(BaseModel):
    query: str = Field()

# Define the API retrieval function
def new_api_retriever(query):
    url = f"http://api.rag.pro/getModel/Langchain/{query}?top_k=2"
    response = requests.get(url)
    return response.json()

# Define the tool using the retrieval function
api_retrieval_tool = Tool.from_function(
    func=new_api_retriever,
    name="APIRetriever",
    description="Retrieves data from Langchain's Documentation, Source Code, Examples, etc.",
    args_schema=APIQueryInput
)

tools = [api_retrieval_tool]

# Defining XML Agent
prompt = hub.pull("hwchase17/xml-agent-convo")

# Initialize connection to Anthropic
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')

llm = ChatAnthropic(
    anthropic_api_key=ANTHROPIC_API_KEY,
    model_name="claude-3-haiku-20240307",
    temperature=0.0
)

def convert_intermediate_steps(intermediate_steps):
    log = ""
    for action, observation in intermediate_steps:
        log += (
            f"<tool>{action.tool}</tool><tool_input>{action.tool_input}"
            f"</tool_input><observation>{observation}</observation>"
        )
    return log

def convert_tools(tools):
    return "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

agent = (
    {
        "input": lambda x: x["input"],
        "chat_history": lambda x: x["chat_history"],
        "agent_scratchpad": lambda x: convert_intermediate_steps(
            x["intermediate_steps"]
        ),
    }
    | prompt.partial(tools=convert_tools(tools))
    | llm.bind(stop=["</tool_input>", "</final_answer>"])
    | XMLAgentOutputParser()
)

agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True
)

@app.route("/")
def welcome():
    return "Hello world"

@app.route("/home")
def home():
    return jsonify({"msg": "Home page here"})

@app.route("/chatting/<string:namespace_name>/<string:q>")
def chat(namespace_name, q):
    response = agent_executor.invoke({
        "input": q,
        "chat_history": ""
    })
    answer = response["output"]
    json_response = {"answer": answer}
    return jsonify(json_response)

@app.route("/chat", methods=["POST"])
def chatting():
    requestBody = request.get_json()
    q = requestBody['question']
    namespace_name = requestBody['namespace_name']
    response = agent_executor.invoke({
        "input": q,
        "chat_history": ""
    })
    answer = response["output"]
    json_response = {"answer": answer}
    return jsonify(json_response)

if __name__ == "__main__":
    app.run(debug=True)
