import networkx as nx
import argparse
import os


def getDataSetName(filename: str):
    bindex = filename.rfind('/')
    eindex = filename.rfind('.')
    return filename[bindex + 1: eindex]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-fi", "--filein", help="Input File Name")
    # parser.add_argument("-fo", "--fileout", help="Output File Name")
    args = parser.parse_args()
    filein = args.filein
    if len(os.path.dirname(filein)) == 0:
        fileout = os.path.dirname(filein) + "edge.txt"
    else:
        fileout = os.path.dirname(filein) + "/edge.txt"
    # dataname = getDataSetName(filein)
    dataname = filein.split('/')[-1].split('.')[0]

    G = nx.read_edgelist(filein)
    nodes = list(G.nodes())
    sorted(nodes)
    m = dict()
    print(len(nodes))
    for i in range(len(nodes)):
        m[nodes[i]] = i
    H = nx.relabel_nodes(G, m, True)

    print("The number of edges is ", len(list(G.edges())))
    print(type(H))
    print("Output file is {}".format(fileout))

    fp = open(fileout, 'w')
    print("# nodes of {}".format(dataname), file=fp)
    print("# {}".format(len(nodes)), file=fp)
    print("# node, node", file=fp)
    edges = list(H.edges())
    sorted(edges)
    for edge in edges:
        print("{}\t{}".format(edge[0], edge[1]), file=fp)
    fp.close()
