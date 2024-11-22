import cloudpickle

builder_filepath = "builder.pkl"
with open(builder_filepath, "rb") as f:
    builder = cloudpickle.load(f)
    print(f"builder loaded: {builder_filepath}")


# graph_filepath = "graph.pkl"
# with open(graph_filepath, "rb") as f:
#     graph = cloudpickle.load(f)
#     print(f"graph loaded: {graph_filepath}")


# Test Graph ----------------------------------------------------

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

graph = builder.compile()

inputs = {"messages": [HumanMessage(content="my favorite pets are cats and dogs")]}

response = graph.invoke(inputs, {"configurable": {"user_id": "123"}})
response["messages"][-1].pretty_print()  # fake response

for chunk in graph.stream(
    inputs, {"configurable": {"user_id": "123"}}, stream_mode="values"
):
    chunk["messages"][-1].pretty_print()  # fake response
