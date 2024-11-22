# langgraph-dill


```bash
# bash ./create-condaenv.sh langgraph-dill
pip install -r requirements.txt
cd test
```

# Save the graph

```bash
$ python test_save_graph.py
Test: invoke ----------------------------------------------------------------------
================================== Ai Message ==================================

fake response
Test: stream ----------------------------------------------------------------------
================================ Human Message =================================

my favorite pets are cats and dogs
================================== Ai Message ==================================

fake response
Save ---------------------------------------------------------------------------
builder saved: builder.pkl
graph saved: graph.pkl
```

# Load the graph

```bash
$ python test_load_graph.py
Load ---------------------------------------------------------------------------
builder loaded: builder.pkl
Test: invoke ----------------------------------------------------------------------
================================== Ai Message ==================================

fake response
Test: stream ----------------------------------------------------------------------
================================ Human Message =================================

my favorite pets are cats and dogs
================================== Ai Message ==================================

fake response
```


