from typing import Literal
from typing import List

from langchain_core.tools import tool
from langchain_core.runnables.config import RunnableConfig

from langgraph.prebuilt import ToolNode

# from langchain_anthropic import ChatAnthropic
from langchain_openai import AzureChatOpenAI
from custom_objects import MockChatModel

from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langgraph.graph import StateGraph, MessagesState, START, END

user_to_pets = {}


@tool(parse_docstring=True)
def update_favorite_pets(
    # NOTE: config arg does not need to be added to docstring, as we don't want it to be included in the function signature attached to the LLM
    pets: List[str],
    config: RunnableConfig,
) -> None:
    """Add the list of favorite pets.

    Args:
        pets: List of favorite pets to set.
    """
    user_id = config.get("configurable", {}).get("user_id")
    user_to_pets[user_id] = pets


@tool
def delete_favorite_pets(config: RunnableConfig) -> None:
    """Delete the list of favorite pets."""
    user_id = config.get("configurable", {}).get("user_id")
    if user_id in user_to_pets:
        del user_to_pets[user_id]


@tool
def list_favorite_pets(config: RunnableConfig) -> None:
    """List favorite pets if any."""
    user_id = config.get("configurable", {}).get("user_id")
    return ", ".join(user_to_pets.get(user_id, []))


tools = [update_favorite_pets, delete_favorite_pets, list_favorite_pets]
tool_node = ToolNode(tools)

# model_with_tools = ChatAnthropic(
#     model="claude-3-haiku-20240307", temperature=0
# ).bind_tools(tools)
model_with_tools = MockChatModel(
    name="dummy",
).bind_tools(tools)

def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END


def call_model(state: MessagesState):
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


builder = StateGraph(MessagesState)

# Define the two nodes we will cycle between
builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, ["tools", END])
builder.add_edge("tools", "agent")

graph = builder.compile()


inputs = {"messages": [HumanMessage(content="my favorite pets are cats and dogs")]}

# invoke ----------------------------------------------------
print(f"Test: invoke {'-' * 70}")
response = graph.invoke(inputs, {"configurable": {"user_id": "123"}})
response["messages"][-1].pretty_print()  # fake response

# stream ----------------------------------------------------
print(f"Test: stream {'-' * 70}")
for chunk in graph.stream(
    inputs, {"configurable": {"user_id": "123"}}, stream_mode="values"
):
    chunk["messages"][-1].pretty_print()  # fake response


# Save: dill ----------------------------------------------------

# import dill

# dump_filepath = "graph.dill"
# with open(dump_filepath, "wb") as f:
#     dill.dump(graph, f)
#     print(f"graph saved: {dump_filepath}")


# Save: cloudpickle ----------------------------------------------------

import cloudpickle


print(f"Save {'-' * 75}")

builder_filepath = 'builder.pkl'
with open(builder_filepath, "wb") as f:
    cloudpickle.dump(builder, f)
    print(f"builder saved: {builder_filepath}")


graph_filepath = "graph.pkl"
with open(graph_filepath, "wb") as f:
    cloudpickle.dump(graph, f)
    print(f"graph saved: {graph_filepath}")
