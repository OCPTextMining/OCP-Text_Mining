import networkx as nx
import glob
import json
from config import Config


def build_graph(final_path):
    files = glob.glog(final_path + "*.csv")

    G = nx.Graph()
    edges = []
    for filename in files:
        with open(final_path + filename + ".txt", 'r') as f:
            for l in f.readlines():
                triple = json.loads(l)
                sub, rel, obj = triple[0], triple[1], triple[2]
                edges.append((sub, obj, {'relation': rel}))

    G.add_edges_from(edges)
    return G

if __name__ == "__main__":
    final_path = Config.TRIPLES + "final/"
    print("Building graph")