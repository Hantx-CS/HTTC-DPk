import datetime
import random
import sys
import numpy as np
import networkx as nx
import time
import math
import os
import copy
import argparse


DIR_PATH = "../../data/"
DATASET = "Wiki-Vote"
ERROR_FILE = "error.log"
LOG_FILE = "run.log"
RESULT_FILE = "result.log"
COMMUNITY_FILE = DATASET + "/community.dat"
CANDIDATE_LENGTH = []
Version = 0
random.seed(2024)
np.random.seed(2024)
GM_Random = random.Random(2024)
RR_Random = random.Random(2024)
EM_Random = random.Random(2024)
Sample_Random = random.Random(2024)


def incrementDict(d: dict, key):
    if key in d:
        d[key] += 1
    else:
        d[key] = 1
    return d


def outputInfo(s, fileList=[]):
    print(s)
    for filename in fileList:
        print(s, file=filename)


def listSet2DFlatList(list1):
    result = []
    for subSet in list1:
        result.extend(list(subSet))
    return result


def listSet2DList(list1):
    result = []
    for subSet in list1:
        result.append(list(subSet))
    return result


def listSet2DFlatSet(list1):
    result = set()
    for subSet in list1:
        result.update(subSet)
    return result


class GlobalController:
    def __init__(self, filename):
        self.G = nx.read_edgelist(filename)
        print(len(nx.nodes(self.G)))
        self.communities = []
        self.boundary_edge = []  # boundary_edge
        self.boundary_node = []
        self.subG = []
        self.dmax1, self.dmax2 = dict(), dict()
        self.triangle1, self.triangle2, self.triangle3 = dict(), dict(), dict()
        self.triangle = dict()
        self.triangleSum1, self.triangleSum2, self.triangleSum3 = 0, 0, 0

        if os.path.exists(COMMUNITY_FILE):
            self.ReadCommunity()
        else:
            self.LouvainPartition(2)
            # self.RandomPartition(2, [0.5, 0.5])
            self.StoreCommunity()
        for i in range(0, len(self.communities)):
            for j in range(0, len(self.communities)):
                if i == j:
                    continue
                self.boundary_edge[i][j].sort(key=lambda x: (x[0], x[1]))

        print(len(self.communities), file=LOG_FILE)
        node_number = []
        for i in range(0, len(self.communities)):
            j = 1 - i
            node_number.append(len(nx.nodes(self.subG[i])))
            self.deg1 = findMaximumDegree(self.communities[i], set(nx.edges(self.subG[i])))
            self.deg2 = findMaximumDegree(self.communities[i], set(self.boundary_edge[i][j]))
            self.dmax1[i], self.dmax2[i] = max(self.deg1.values()), max(self.deg2.values())
        print(node_number, file=LOG_FILE)

        print("{}: Len of boundary edge is {},{},{},{}".format(
            datetime.datetime.now(), len(self.boundary_edge[0][0]), len(self.boundary_edge[0][1]),
            len(self.boundary_edge[1][0]), len(self.boundary_edge[1][1])))
        print("{}: Len of boundary node is {},{},{},{}".format(
            datetime.datetime.now(), len(self.boundary_node[0][0]), len(self.boundary_node[0][1]),
            len(self.boundary_node[1][0]), len(self.boundary_node[1][1])))
        self.triangleCount()
        # print(max(nx.degree(self.G)))

    def ReadCommunity(self):
        file = open(COMMUNITY_FILE, 'r')
        for line in file:
            self.communities.append(line.split())
        file.close()
        print("Read {} successfully!".format(COMMUNITY_FILE))
        print("Number of communities is {}".format(len(self.communities)))

        for i in range(0, len(self.communities)):
            subG = nx.subgraph(self.G, self.communities[i]).copy()
            self.subG.append(subG)

        for i in range(0, len(self.communities)):
            self.boundary_edge.append([])
            self.boundary_node.append([])
            for j in range(0, len(self.communities)):
                if i == j:
                    self.boundary_edge[i].append([])
                    self.boundary_node[i].append([])
                    continue
                self.boundary_edge[i].append(
                    list(nx.edge_boundary(self.G, nx.nodes(self.subG[i]), nx.nodes(self.subG[j]))))
                self.boundary_node[i].append(
                    list(nx.node_boundary(self.G, nx.nodes(self.subG[j]), nx.nodes(self.subG[i]))))
        # printLine3Count(self.boundary)
        # self.subG.append(nx.Graph(self.boundary_edge[0][1]))

    def StoreCommunity(self):
        file = open(COMMUNITY_FILE, 'w')
        for community in self.communities:
            print(' '.join(map(str, community)), file=file)
        file.close()
        print("Store {} successfully!".format(COMMUNITY_FILE))
        print("Number of communities is {}".format(len(self.communities)))

    def LouvainPartition(self, maxNum=3):
        print("Louvain Partition, community number is {}".format(maxNum))
        louvain = nx.community.louvain_communities(self.G)
        for i in range(0, len(louvain)):
            self.communities.append(list(louvain[i]))
            # print(i, len(self.communities[i]), self.communities[i])

        while len(self.communities) > maxNum:
            self.communities.sort(key=len, reverse=True)
            temp = self.communities[maxNum]
            del self.communities[maxNum]
            self.communities[maxNum - 1].extend(temp)
        random.shuffle(self.communities)
        # printLine2(self.communities)

        for i in range(0, len(self.communities)):
            subG = nx.subgraph(self.G, self.communities[i]).copy()
            self.subG.append(subG)

        for i in range(0, len(self.communities)):
            self.boundary_edge.append([])
            self.boundary_node.append([])
            for j in range(0, len(self.communities)):
                if i == j:
                    self.boundary_edge[i].append([])
                    self.boundary_node[i].append([])
                    continue
                self.boundary_edge[i].append(
                    list(nx.edge_boundary(self.G, nx.nodes(self.subG[i]), nx.nodes(self.subG[j]))))
                self.boundary_node[i].append(
                    list(nx.node_boundary(self.G, nx.nodes(self.subG[i]), nx.nodes(self.subG[j]))))
        # printLine3Count(self.boundary)
        # self.subG.append(nx.Graph(self.boundary_edge[0][1]))

    def RandomPartition(self, maxNum=3, fraction=[1, 1, 1]):
        print("fraction = {}".format(fraction))
        nodes = []
        numList = list(range(maxNum))
        for i in range(0, maxNum):
            nodes.append([])
        for node in nx.nodes(self.G):
            i = random.choices(numList, weights=fraction, k = 1)[0]
            nodes[i].append(node)
        for i in range(0, maxNum):
            self.communities.append(nodes[i])
            subG = nx.subgraph(self.G, nodes[i]).copy()
            self.subG.append(subG)
        for i in range(0, len(self.communities)):
            self.boundary_edge.append([])
            self.boundary_node.append([])
            for j in range(0, len(self.communities)):
                if i == j:
                    self.boundary_edge[i].append(set())
                    self.boundary_node[i].append(set())
                    continue
                self.boundary_edge[i].append(
                    list(nx.edge_boundary(self.G, nx.nodes(self.subG[i]), nx.nodes(self.subG[j]))))
                self.boundary_node[i].append(
                    list(nx.node_boundary(self.G, nx.nodes(self.subG[j]), nx.nodes(self.subG[i]))))
        # self.subG.append(nx.Graph(self.boundary_edge[0][1]))

    def triangleCount1(self, i):
        G = self.subG[i]
        result = sum(nx.triangles(G).values()) / 3
        self.triangle1[i] = result
        return result

    def triangleCount2(self, i, j):
        G = nx.subgraph(self.G, self.boundary_node[i][j])
        count = sum(nx.triangles(G).values()) / 3
        G = nx.Graph(list(nx.edges(G)) + list(self.boundary_edge[i][j]))
        result = sum(nx.triangles(G).values()) / 3 - count
        self.triangle2[i][j] = result
        return result

    def triangleCount3(self, i, j, k):
        G = nx.subgraph(self.G, self.communities[i] + self.communities[j] + self.communities[k])
        result = (sum(nx.triangles(G).values()) / 3
                  - self.triangle1[i] - self.triangle1[j] - self.triangle1[k]
                  - self.triangle2[i][j] - self.triangle2[i][k] - self.triangle2[j][k]
                  - self.triangle2[j][i] - self.triangle2[k][i] - self.triangle2[k][j])
        self.triangle3[i][j][k] = result
        self.triangle3[i][k][j] = result
        self.triangle3[j][i][k] = result
        self.triangle3[j][k][i] = result
        self.triangle3[k][i][j] = result
        self.triangle3[k][j][i] = result
        return result

    def triangleCount(self):
        for i in range(len(self.communities)):
            self.triangleSum1 += self.triangleCount1(i)
            self.triangle2[i] = dict()
            self.triangle3[i] = dict()

        for i in range(len(self.communities)):
            for j in range(len(self.communities)):
                self.triangleSum2 += self.triangleCount2(i, j)
                self.triangle3[i][j] = dict()
                self.triangle3[j][i] = dict()

        for i in range(len(self.communities)):
            for j in range(i + 1, len(self.communities)):
                for k in range(j + 1, len(self.communities)):
                    self.triangleSum3 += self.triangleCount3(i, j, k)

        print("All triangle of G is {}, one party is {}, two parties is {}, three parties is {}".format(
            sum(nx.triangles(self.G).values()) / 3, self.triangleSum1, self.triangleSum2, self.triangleSum3
        ), file=LOG_FILE)


# Test 2- star and 3-path
def countStructure(gc: GlobalController):
    for i in range(0, len(gc.communities)):
        for j in range(0, len(gc.communities)):
            if i == j:
                continue
            # 2- star
            count, count_2star, count_path = 0, 0, 0
            for p in range(0, len(gc.boundary_edge[i][j])):
                for q in range(p + 1, len(gc.boundary_edge[i][j])):
                    e1 = gc.boundary_edge[i][j][p]
                    e2 = gc.boundary_edge[i][j][q]
                    count += 1
                    if(e1[0] == e2[0]):
                        count_2star += 1
                    elif (e1[0], e2[0]) in nx.edges(gc.subG[i]):
                        count_path += 1
            print("From party {} to party {}, all possible (e1, e2) is {}, 2-star is {}, 3-path is {}".format(i, j, count, count_2star, count_path))


# Test boundary edges
def countBoundaryNodeConnection(gc: GlobalController):
    for i in range(0, len(gc.communities)):
        for j in range(0, len(gc.boundary_node)):
            if i == j:
                continue

            count, count_existing_edge = 0, 0
            for p in range(0, len(gc.boundary_node[i][j])):
                for q in range(p + 1, len(gc.boundary_node[i][j])):
                    n1 = gc.boundary_node[i][j][p]
                    n2 = gc.boundary_node[i][j][q]
                    # print(n1, n2)
                    # print((n1, n2))
                    # print((n1, n2) in nx.edges(gc.subG[i]))
                    # sys.exit(1)
                    # if (n1, n2) in nx.edges(gc.G):
                    #     print(n1, n2)
                    #     print((n1, n2) in nx.edges(gc.subG[0]))
                    #     print((n1, n2) in nx.edges(gc.subG[1]))
                    #     print((n1, n2) in nx.edges(gc.subG[2]))
                    #     print((n1, n2) in gc.boundary_edge[i][j])
                    #     print((n1, n2) in gc.boundary_edge[j][i])
                    #     if (n1, n2) in gc.subG[2]:
                    #         sys.exit(1)
                    count += 1
                    if (n1, n2) in nx.edges(gc.subG[j]):
                        count_existing_edge += 1
            print("Party {} possible boundary node can connect {} edges, and actually they connect {} edges".format(i, count, count_existing_edge))
            # sys.exit(1)


def GM0(epsilon, delta):
    alpha = math.exp(-epsilon / delta)
    beta = 1 + alpha
    r = GM_Random.random()
    res = np.random.geometric(1 - alpha)
    if r < 1 - alpha: # beta * (1 - alpha) / (1 + alpha)
        return 0
    else:
        return res


def GM(epsilon, delta):
    alpha = math.exp(-epsilon / delta)
    r = GM_Random.random()
    if r < (1 - alpha) / (1 + alpha):
        r = GM_Random.random()
        res = np.random.geometric(1 - alpha)
        return 0
    else:
        r = GM_Random.random()
        res = np.random.geometric(1 - alpha)
        if r < 0.5:
            return res
        else:
            return -res


def RR(epsilon, flag):
    p = math.exp(epsilon)
    p = p / (1 + p)
    r = RR_Random.random()
    if r > p:
        flag = not flag
    return flag


def EM(epsilon, itemSet, delta=2):
    weights = []
    p1 = math.exp(epsilon * 1 / (2 * delta))
    p2 = math.exp(epsilon * (-1) / (2 * delta))
    for i in range(0, len(itemSet)):
        edge, v = itemSet[i]
        if edge[0] == v:
            weights.append(p1)
        else:
            weights.append(p2)
    return random.choices(itemSet, weights=weights)[0]


def findMaximumDegree(nodes, edges):
    deg = dict()
    for e in edges:
        n1, n2 = e
        if n1 in nodes:
            if n1 in deg.keys():
                deg[n1] += 1
            else:
                deg[n1] = 1
        if n2 in nodes:
            if n2 in deg.keys():
                deg[n2] += 1
            else:
                deg[n2] = 1
    dmax = max(deg.values())
    print("{}:Find max degree is {}".format(datetime.datetime.now(), dmax))
    return deg


def countTwoStars(nodes, edges):
    deg, twoStars = dict(), dict()
    for e in edges:
        n1, n2 = e
        if n1 in nodes:
            if n1 in deg.keys():
                deg[n1] += 1
            else:
                deg[n1] = 1
        if n2 in nodes:
            if n2 in deg.keys():
                deg[n2] += 1
            else:
                deg[n2] = 1
    dmax = max(deg.values())
    number = 0
    for key in deg.keys():
        twoStars[key] =  deg[key] * (deg[key] - 1) / 2
        number += twoStars[key]
    print("Count number of 2-stars is {}".format(number))
    return number, twoStars


class Algorithm:
    def __init__(self, gc: GlobalController, epsilonList, kAnony):
        self.gc = copy.deepcopy(gc)
        self.epsilon1 = epsilonList[0]
        self.epsilon2 = epsilonList[1]
        self.epsilon3 = epsilonList[2]
        self.degrees = dict()
        self.estimate1 = 0
        self.estimate2 = 0
        self.boundary_edges = []
        self.kAnony = kAnony
        self.candidatesList = []
        self.responsesList = []

    def GetDegree(self):
        for i in range(0, len(self.gc.communities)):
            self.degrees[i] = dict()
            self.degrees[i]['intra'] = self.gc.dmax1[i] + GM0(self.epsilon1, 1)
            self.degrees[i]['inter'] = self.gc.dmax2[i] + GM0(self.epsilon1, 1)
        print(self.degrees)

    def TriangleCount1(self):
        for i in range(0, len(self.gc.communities)):
            self.estimate1 += self.gc.triangle1[i] + GM(self.epsilon2, self.degrees[i]['intra'] - 1)
        return self.estimate1
    
    def TriangleCount2(self):
        self.Anony()
        if Version == 1:
            self.Choice()
        self.Interaction()
        self.Aggregator()

    def Anony(self):
            self.boundary_edges.append(self.gc.boundary_edge[0][1])
            self.boundary_edges.append(self.gc.boundary_edge[1][0])
            for i in range(0, len(self.gc.communities)):
                candidates = []
                for edge in self.boundary_edges[i]:
                    cand = (edge, edge[0])
                    candidates.append(cand)
                    dummyList = random.choices(self.gc.communities[i], k=self.kAnony - 1)
                    for dummy in dummyList:
                        cand = (edge, dummy)
                        candidates.append(cand)
                        # print(candidates)
                        # sys.exit(1)
                self.candidatesList.append(candidates)
            print("Len of boundary edges is {}, after anonymity is {}".format(len(self.boundary_edges[0]), len(self.candidatesList[0])))
            return self.candidatesList
    
    def Choice(self):
        print("Version = {}, this is Choice".format(Version))
        SelItemsList = []
        for i in range(0, len(self.gc.communities)):
            SelItems = []
            ItemSets = [self.candidatesList[i][j:j + self.kAnony] for j in range(0, len(self.candidatesList[i]), self.kAnony)]
            count = -1
            for itemSet in ItemSets:
                if count == -1:
                    count += 1
                    print(itemSet)
                item = EM(self.epsilon3, itemSet)
                if item[0][0] == item[1]:
                    count += 1
                SelItems.append(item)
            print("True edge is {}, all edge is {}".format(count, len(SelItems)))
            SelItemsList.append(SelItems)
        self.candidatesList = SelItemsList
        return self.candidatesList
    
    def Response(self, number, subList, node):
        # print(number, subList, node)
        response = []
        for i in range(0, len(subList)):
            for j in range(i + 1, len(subList)):
                edge1, edge2 = subList[i][0], subList[j][0]
                node1, node2 = edge1[1], edge2[1]
                flag = self.gc.subG[number].has_edge(node1, node2)
                flag = RR(self.epsilon3, flag)
                response.append((edge1, edge2, flag))
            # print(response)
            # sys.exit(1)
        return response

    def Interaction(self):
        for i in range(0, len(self.gc.communities)):
            responses = []
            self.candidatesList[i].sort(key=lambda x: (x[1], x[0][0], x[0][1]))
            candidates = self.candidatesList[i]
            temp_value = candidates[0][1]
            temp_index = 0
            for j in range(0, len(candidates)):
                cand = candidates[j]
                current_value = cand[1]
                if temp_value == current_value:
                    continue
                subList = candidates[temp_index: j]
                responses.extend(self.Response(abs(1 - i), subList, temp_value))
                temp_value = current_value
                temp_index = j
            subList = candidates[temp_index: ]
            responses.extend(self.Response(abs(1 - i), subList, temp_value))
            self.responsesList.append(responses)
        return self.responsesList

    def Aggregator(self):
        for i in range(0, len(self.gc.communities)):
            j = 1 - i
            number, twoStars = countTwoStars(self.gc.communities[i], set(self.gc.boundary_edge[i][j]))
            count, count_true, calibrate = 0, 0, 0
            responses = self.responsesList[i]
            for response in responses:
                edge1, edge2, flag = response
                if edge1[0] == edge2[0]:
                    count += 1
                    if flag:
                        count_true += 1
            p = p = math.exp(self.epsilon3)
            p = p / (1 + p)
            if Version ==1:
                print("Version = ", Version)
                calibrate = (number / count) * ((p - 1) * count / (2 * p - 1)  + count_true / (2 * p - 1)) + GM(self.epsilon2, self.degrees[i]['inter'] - 1)
            else:
                calibrate = (p - 1) * count / (2 * p - 1)  + count_true / (2 * p - 1) + GM(self.epsilon2 + self.epsilon3, self.degrees[i]['inter'] - 1)
            outputInfo("Party {} estimates triangle2: count : {}, true: {}, calibrate: {}, trueSum: {}".
                       format(i, count, count_true, calibrate, self.gc.triangle2[i][1 - i]), [LOG_FILE])
            self.estimate2 += calibrate

    def run(self):
        self.GetDegree()
        self.TriangleCount1()
        if Version == 2:
            return 2
        self.TriangleCount2()


def init_logfile():
    global ERROR_FILE
    global LOG_FILE
    global RESULT_FILE

    ERROR_FILE = DATASET +"/" + ERROR_FILE
    LOG_FILE = DATASET +"/" + LOG_FILE
    RESULT_FILE = DATASET +"/" + RESULT_FILE

    ERROR_FILE = open(ERROR_FILE, 'a', buffering=1)
    LOG_FILE = open(LOG_FILE, 'a', buffering=1)
    RESULT_FILE = open(RESULT_FILE, 'a', buffering=1)
    print(datetime.datetime.now(), file=ERROR_FILE)
    print(datetime.datetime.now(), file=LOG_FILE)
    print(datetime.datetime.now(), file=RESULT_FILE)


def close_logfile():
    ERROR_FILE.close()
    LOG_FILE.close()
    RESULT_FILE.close()


def init_random(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    GM_Random.seed(seed)
    RR_Random.seed(seed)
    EM_Random.seed(seed)
    Sample_Random.seed(seed)


if __name__ == '__main__':
    e, kAnony, roundNumber = 1, 2, 10
    gcList, gc = [], None
    parser = argparse.ArgumentParser(description="Process command line arguments.")
    parser.add_argument('-d', '--dataset', type=str, help='The dataset name', required=False)
    parser.add_argument('-v', '--version', type=int, help='The version of algorithm', required=False)
    parser.add_argument('-e', '--epsilon', type=int, help='Epsilon', required=False)
    parser.add_argument('-k', '--kAnony', type=int, help='kAnony', required=False)
    args = parser.parse_args()
    if args.dataset is not None:
        DATASET = args.dataset
        COMMUNITY_FILE = DATASET + "/community.dat"
    if args.version is not None:
        Version = args.version
    if args.epsilon is not None:
        e = args.epsilon
    if args.kAnony is not None:
        kAnony = args.kAnony
    
    init_logfile()
    init_random(2024)

    if DATASET =="IMDB":
        print("This dataset is {}.".format(DATASET))
        for i in range(0, roundNumber):
            COMMUNITY_FILE = DATASET + "/community" + str(i) +".dat"
            print("Global Controller Round is {}".format(i))
            sourceFile = DIR_PATH + DATASET + "/60000/outEdge_n60000_itr" + str(i) + ".txt"
            tmp = GlobalController(sourceFile)
            gcList.append(tmp)
    else:
        print("This dataset is {}.".format(DATASET))
        sourceFile = DIR_PATH + DATASET + "/edge.txt"
        gc = GlobalController(sourceFile)

    MEAN_TS1, MEAN_TS2, MEAN_TS3, MEAN_TS = [], [], [], []
    MEAN_T1, MEAN_T2, MEAN_T3, MEAN_T = [], [], [], []
    MEAN_L21, MEAN_L22, MEAN_L23, MEAN_L2 = [], [], [], []
    MEAN_RE1, MEAN_RE2, MEAN_RE3, MEAN_RE = [], [], [], []
    for e in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]:
    # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # for kAnony in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]:
        init_random(2024)
        close_logfile()
        # ERROR_FILE = "error_k" + str(kAnony) + ".log"
        # LOG_FILE = "run_k" + str(kAnony) + ".log"
        # RESULT_FILE = "result_k" + str(kAnony) + ".log"
        ERROR_FILE = "error_e" + str(e) + ".log"
        LOG_FILE = "run_e" + str(e) + ".log"
        RESULT_FILE = "result_e" + str(e) + ".log"
        init_logfile()
        TS1, TS2, TS3, TS = [], [], [], []
        T1, T2, T3, T = [], [], [], []
        L21, L22, L23, L2 = [], [], [], []
        RE1, RE2, RE3, RE = [], [], [], []
        for round in range(0, roundNumber):
            if DATASET =="IMDB":
                gc = gcList[round]
            outputInfo("Time is {}".format(datetime.datetime.now()), [LOG_FILE])
            outputInfo("Epsilon = {}".format(e), [LOG_FILE])
            outputInfo("kAnony = {}".format(kAnony), [LOG_FILE])
            outputInfo("Round = {}".format(round), [LOG_FILE])
            alg = Algorithm(gc, [e / 4, e / 4, e / 2], kAnony)
            if Version == 2:
                alg = Algorithm(gc, [e / 2, e / 2, 0], kAnony)
            alg.run()
            # T1
            t1 = alg.estimate1
            outputInfo("True T1 is {}, estimated T1 is {}, accuracy is {}".format(gc.triangleSum1, t1, abs(t1 / gc.triangleSum1 - 1)), [LOG_FILE])
            TS1.append(gc.triangleSum1)
            T1.append(t1)
            L21.append(math.pow(t1 - gc.triangleSum1, 2))
            RE1.append(abs(t1 / gc.triangleSum1 - 1))

            # T2
            t2 = alg.estimate2
            outputInfo("True T2 is {}, estimated T2 is {}, accuracy is {}".format(gc.triangleSum2, t2, abs(t2 / gc.triangleSum2 - 1)), [LOG_FILE])
            TS2.append(gc.triangleSum2)
            T2.append(t2)
            L22.append(math.pow(t2 - gc.triangleSum2, 2))
            RE2.append(abs(t2 / gc.triangleSum2 - 1))

            # T
            tSum = gc.triangleSum1 + gc.triangleSum2
            t = t1 + t2
            outputInfo("True T is {}, estimated T is {}, accuracy is {}".format(tSum, t, abs(t / tSum - 1)), [LOG_FILE])
            TS.append(tSum)
            T.append(t)
            L2.append(math.pow(t - tSum, 2))
            RE.append(abs(t / tSum - 1))


        print(TS1, file=RESULT_FILE)
        print(T1, file=RESULT_FILE)
        print(L21, file=RESULT_FILE)
        print(RE1, file=RESULT_FILE)
        print(TS2, file=RESULT_FILE)
        print(T2, file=RESULT_FILE)
        print(L22, file=RESULT_FILE)
        print(RE2, file=RESULT_FILE)
        print(TS, file=RESULT_FILE)
        print(T, file=RESULT_FILE)
        print(L2, file=RESULT_FILE)
        print(RE, file=RESULT_FILE)

        print(np.mean(TS1), file=ERROR_FILE)
        print(np.mean(T1), file=ERROR_FILE)
        print(np.mean(L21), file=ERROR_FILE)
        print(np.mean(RE1), file=ERROR_FILE)
        print(np.mean(TS2), file=ERROR_FILE)
        print(np.mean(T2), file=ERROR_FILE)
        print(np.mean(L22), file=ERROR_FILE)
        print(np.mean(RE2), file=ERROR_FILE)
        print(np.mean(TS), file=ERROR_FILE)
        print(np.mean(T), file=ERROR_FILE)
        print(np.mean(L2), file=ERROR_FILE)
        print(np.mean(RE), file=ERROR_FILE)

        MEAN_TS1.append(np.mean(TS1))
        MEAN_T1.append(np.mean(T1))
        MEAN_L21.append(np.mean(L21))
        MEAN_RE1.append(np.mean(RE1))
        MEAN_TS2.append(np.mean(TS2))
        MEAN_T2.append(np.mean(T2))
        MEAN_L22.append(np.mean(L22))
        MEAN_RE2.append(np.mean(RE2))
        MEAN_TS.append(np.mean(TS))
        MEAN_T.append(np.mean(T))
        MEAN_L2.append(np.mean(L2))
        MEAN_RE.append(np.mean(RE))
    
    close_logfile()
    ERROR_FILE = "error.log"
    LOG_FILE = "run.log"
    RESULT_FILE = "result.log"
    init_logfile()
    print(MEAN_TS1, file=RESULT_FILE)
    print(MEAN_T1, file=RESULT_FILE)
    print(MEAN_L21, file=RESULT_FILE)
    print(MEAN_RE1, file=RESULT_FILE)
    print(MEAN_TS2, file=RESULT_FILE)
    print(MEAN_T2, file=RESULT_FILE)
    print(MEAN_L22, file=RESULT_FILE)
    print(MEAN_RE2, file=RESULT_FILE)
    print(MEAN_TS, file=RESULT_FILE)
    print(MEAN_T, file=RESULT_FILE)
    print(MEAN_L2, file=RESULT_FILE)
    print(MEAN_RE, file=RESULT_FILE)
                 
    

            


