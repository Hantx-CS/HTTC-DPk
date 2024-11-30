import networkx as nx
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-fi", "--filein", help="Input File Name")
    # parser.add_argument("-fo", "--fileout", help="Output File Name")
    args = parser.parse_args()
    filein = args.filein
    fileout = os.path.dirname(filein) + "/edge.csv"

    G = nx.read_edgelist(filein)
    nodes = list(G.nodes())
    sorted(nodes)
    m = dict()
    for i in range(len(nodes)):
        m[nodes[i]] = i
    H = nx.relabel_nodes(G, m, True)

    print(len(list(G.edges())))
    print(type(H))

    fp = open(fileout, 'w')
    print("# Directed graph (each unordered pair of nodes is saved once): Cit-HepTh.txt ", file=fp)
    print("# Paper citation network of Arxiv High Energy Physics Theory category", file=fp)
    print("# Nodes: 27770 Edges: 352807", file=fp)
    print("# FromNodeId	ToNodeId", file=fp)
    edges = list(H.edges())
    sorted(edges)
    for edge in edges:
        print("{}\t{}".format(edge[0], edge[1]), file=fp)
    fp.close()
