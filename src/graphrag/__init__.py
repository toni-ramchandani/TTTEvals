"""GraphRAG over the Helix handbook.

Build a graph of atomic facts from the handbook, then answer by linking the
question to entities and traversing. Same trace schema as Helix.

    from graphrag import GRAPH_DEFAULT, build, answer
    g = build(GRAPH_DEFAULT)
    trace = answer("Which office has the longest primary caregiver leave?", g, GRAPH_DEFAULT)
"""

from .extract import GraphBuildConfig, build_graph
from .graph import Edge, Fact, Graph, Node
from .pipeline import GRAPH_DEFAULT, GRAPH_NAIVE, GraphConfig, answer, build

__all__ = [
    "GraphBuildConfig", "build_graph", "Edge", "Fact", "Graph", "Node",
    "GRAPH_DEFAULT", "GRAPH_NAIVE", "GraphConfig", "answer", "build",
]
