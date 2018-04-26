# encode 'utf-8'
import networkx as nx
import numpy as np
import scipy as sp
from scipy.sparse import linalg
from graph_matching import read_graph
from graph_matching import sample_graph
import matplotlib.pyplot as plt


def primary_eigenvec(G):
    # obtain the primary eigen vector which corresponds to the largest eigen value
    assert not G.is_directed()
    A = nx.adjacency_matrix(G)
    L, V = linalg.eigs(A, k=1)
    assert np.alltrue(V[:, 0] > 0) or np.alltrue(V[:, 0] < 0)
    privec = {}
    index = 0
    for node in G.nodes():
        privec[node] = abs(V[:, 0][index])
        index += 1
    return privec


def spectral_serialize(G):
    # serialize the nodes of G based on the spectral method
    privec = primary_eigenvec(G)
    n0 = max(privec.items(), key = lambda x:x[1])[0]
    S = [n0]
    privec.pop(n0)
    serialized = set()
    serialized.add(n0)
    C = [(node, privec.get(node, 0)) for node in G.neighbors(n0) if node not in serialized]
    while len(S) != G.order():
        if len(C) != 0:
            n0 = max(C, key = lambda x:[1])[0]
        else:
            n0 = max(privec.items(), key = lambda x:x[1])[0]
        S.append(n0)
        C = [(node, privec.get(node, 0)) for node in G.neighbors(n0) if node not in serialized]
        serialized.add(n0)
        privec.pop(n0)
    return S


def fiedler_vec(G, normalized=False):
    return nx.fiedler_vector(G, normalized=normalized)


def partition_super_cliques(G, alpha=0.5):
    """
    @auther: Chenxu Wang
    Partition the graph into super cliques. Each clique consists of a cneter node and its immediate neighbors
    Each node has a score equals alpha * deg_i / max_deg + beta * rank_i / N, where deg_i is the degree of the node,
    mat_deg is the maximum degree, rank_i is the rank of node according its corresponding component value in the
    Fiedler Vector, N is the number of nodes. alpha is a free parameter
    :param G: The graph
    :param alpha: weight of degree
    :return: Paritions, a list of lists
    """
    fvec = fiedler_vec(G)
    ords = np.argsort(fvec)[::-1]
    nodes = G.nodes()
    node_scores = {}
    deg = G.degree()
    maxD = max(deg.values())
    rank = 1
    for ord in ords:
        node = nodes[ord]
        node_scores[node] = alpha * deg[node] / maxD + (1 - alpha) * rank / G.order() # Calculate the score for each node
        rank += 1
    sorted_s = sorted(node_scores.items(), key=lambda x: x[1]) # sort the nodes by score in ascending order
    partitioned = set()
    partitions = []
    while len(partitioned) < G.order():
        index = 0
        for node, score in sorted_s:
            # finding the center node
            flag = True
            for neighbor in G.neighbors(node):
                if neighbor not in partitioned and score < node_scores[neighbor]:
                    flag = False
                    break
            if not flag:
                index += 1
            else:
                break
        node, score = sorted_s[index]
        partition = [node]
        partition.extend([neighbor for neighbor in G.neighbors(node) if neighbor not in partitioned])
        partitioned.update(partition)
        partitions.append(partition)
        for node in partition:
            sorted_s.remove((node, node_scores.get(node)))
    return partitions


def draw_partitions(G, partitions):
    color_map = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    color_index = 0
    partition_color = {}
    for partition in partitions:
        for node in partition:
            partition_color[node] = color_map[color_index]
        color_index += 1
        color_index = color_index % len(color_map)
    node_color = [partition_color.get(node, 'o') for node in G.nodes()]
    nx.draw_spring(G, node_color=node_color, with_labels=True)
    plt.show()


def main():
    G = read_graph('./data/karate.edgelist')
    # G = nx.erdos_renyi_graph(500, p=0.3)
    alpha = 0.5
    partitions = partition_super_cliques(G, alpha=alpha)
    for partition in partitions:
        print partition
    # draw_partitions(G, partitions)

    G2 = sample_graph(G, 0.7)
    partitions = partition_super_cliques(G2, alpha=alpha)
    for partition in partitions:
        print partition
    # draw_partitions(G2, partitions)


if __name__ == '__main__':
    main()