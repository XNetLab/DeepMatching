#coding=utf-8
from __future__ import division
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import datetime
import csv
import string
import random
import subprocess

from functools import partial
from scipy.io import loadmat
from cpdcore import DeformableRegistration, RigidRegistration, AffineRegistration
from sklearn.decomposition import PCA
from node2vec import learn_nodevec
from deepwalk import __main__ as deepwalk
#from graph_matching import sample_graph_real
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from bimatching.sparse_munkres import munkres


def visualize(X, Y, ax=None, words = None):
    pass

def nodes_embedding(G, p=1, q=1, dimensions=128, embedding='DeepWalk'):
    '''
    Extract the features for each node using Deepwalk or Node2vec.
    There are two parameters, p and q, which are used in Node2vec to control the balance of
        sampling strategies between breadth first search(BFS) and depth first search(DFS).
        Here, p=1 and q=1 means that Node2vec using a random walk.
    :param G: the graph which is need to embedding
    :param p: control the sampling strategies partial to breadth first search(BFS)
    :param q: control the sampling strategies partial to depth first search(DFS)
    :param dimensions: the number of nodes feature's dimension after embedding
    :param embedding: embedding method
    :return: the nodes' list and the list of nodes' vectors
    '''
    if embedding == 'DeepWalk' or embedding == 'deepwalk':
        model = deepwalk.process(G.edges(), dimensions=dimensions, number_walks = 30)
    elif embedding == 'Node2Vec' or embedding == 'node2vec':
        model = learn_nodevec(G, dimensions=dimensions, argp=p, argq=q, num_walks=100)
    nodes = [word for word, vcab in model.wv.vocab.iteritems()]
    inds = [vcab.index for word, vcab in model.wv.vocab.iteritems()]
    X = model.wv.syn0[inds]
    return nodes, X

def map_prob_maxtrix(G1, G2, p=1, q=1, dimensions=128, embedding='DeepWalk'):
    '''
    Match the nodes according to the node feature based on Coherent Point Drift
    :param G1: the matching graph
    :param G2: the matching graph
    :param p: control the sampling strategies partial to breadth first search(BFS)
    :param q: control the sampling strategies partial to depth first search(DFS)
    :param dimensions: the number of nodes feature's dimension after embedding
    :param embedding: embedding method
    :return: two nodes' list corresponding to two graph, and one matrix
    '''
    nodes1, X = nodes_embedding(G1, p=p, q=q, dimensions=dimensions, embedding=embedding)
    nodes2, Y = nodes_embedding(G2, p=p, q=q, dimensions=dimensions, embedding=embedding)
    reg = RigidRegistration(Y, X)
    callback = partial(visualize)
    reg.register(callback)
    P = reg.P
    return (nodes1, nodes2, P)

def bipartite_matching(G1, G2, p=1, q=1, dimensions=128, embedding='DeepWalk'):
    '''
    Match the nodes according to the node feature based on Coherent Point Drift
    :param G1: the matching graph
    :param G2: the matching graph
    :param p: control the sampling strategies partial to breadth first search(BFS)
    :param q: control the sampling strategies partial to depth first search(DFS)
    :param dimensions: the number of nodes feature's dimension after embedding
    :param embedding: embedding method
    :return: a list of tuple (v_i, u_i), where v_i \in G_1, u_i \in G_2
    '''
    node1, node2, proM = map_prob_maxtrix(G1, G2, p=p, q=q, dimensions=dimensions, embedding=embedding)
    M, N = proM.shape
    values = [(i, j, 1 - proM[i][j])
              for i in xrange(M)
              for j in xrange(N) if proM[i][j] > 1e-2]
    values_dict = dict(((i, j), v) for i, j, v in values)
    munkres_match = munkres(values)
    matches = []
    for p1, p2 in munkres_match:
        if p1 > len(node1) or p2 > len(node2):
            continue
        else:
            matches.append((int(node1[p1]), int(node2[p2]), 1 - values_dict[(p1, p2)]))
    return matches

def get_subgraph(real_G, nodes=500):
    '''
    Get a subgraph of a network graph which the number of node is assigned by user, and default value is 500.
    :param real_G: the graph
    :param nodes: the number of subgraph's nodes
    :return: the sub-graph of the graph, which the nodes are the top x, for example x=500, degree in the graph
    '''
    graph = real_G
    sub_nodes = []
    nodes_sort = sorted(graph.degree().items(), key=lambda item:item[1], reverse=True)
    sub_degree = nodes_sort[:nodes]
    for item in sub_degree:
        sub_nodes.append(item[0])
    G = graph.subgraph(sub_nodes)
    return G

def eavalute_accuracy_real(G1, G2, p=1, q=1, repeated=1, dimensions=64, embedding='DeepWalk'):
    '''
    Get the accuracies after 'repeated' times bipartite matching in a certain dimension
    :param G1: the matching graph
    :param G2: the matching graph
    :param p: control the sampling strategies partial to breadth first search(BFS)
    :param q: control the sampling strategies partial to depth first search(DFS)
    :param repeated: repeated times of bipartite matching
    :param dimensions: the number of nodes feature's dimension after embedding
    :param embedding: embedding method
    :return: a list include accuracies after each bipartite matching
    '''
    accuracies = []
    for i in range(repeated):
        count = 0
        matches = bipartite_matching(G1, G2, p=p, q=q, dimensions=dimensions, embedding=embedding)
        for match in matches:
            if match[0] == match[1]:
                count += 1
        accuracy = count * 1.0 / G1.order()
        accuracies.append(accuracy)
        print "Accuracy:", accuracy
    return accuracies

def load_file(file_path, undirected=True):
    '''
    Loading the graph from a txt or edgelist file
    :param file_path: file path
    :param undirected: indicates this will be an undirected graph
    :return: the graph
    '''
    G = nx.Graph()
    txt_reader = open(file_path, 'rb')
    for item in txt_reader:
        if "#" not in item:
            if " " in item:
                item = item.split(' ')
                G.add_edge(int(item[0]), int(item[1]))
                if undirected:
                    G.add_edge(int(item[1]), int(item[0]))
            else:
                item = item.split('\t')
                G.add_edge(int(item[0]), int(item[1]))
                if undirected:
                    G.add_edge(int(item[1]), int(item[0]))
    txt_reader.close()
    return G

def sample_graph(G, s):
    newG = nx.Graph()
    newG.add_edges_from(random.sample(G.edges(), int(len(G.edges())*s)))
    for edge in newG.edges():
        newG[edge[0]][edge[1]]['weight'] = 1.0
    return newG

def process(args):
    '''
    This is the main progress of DeepMatching.
    '''
    if args.input != "none":
        G = load_file(args.input)
        print "Graph: nodes:" + str(G.number_of_nodes()) + ", edges:" + str(G.number_of_edges()) + ", ratio:" + str(args.ratio)
        G1_sample = sample_graph(G, args.ratio)
        G2_sample = sample_graph(G, args.ratio)
        G1 = get_subgraph(G1_sample, args.nodes)
        G2 = get_subgraph(G2_sample, args.nodes)
        GC1 = max(nx.connected_component_subgraphs(G1), key=len)
        GC2 = max(nx.connected_component_subgraphs(G2), key=len)
        resultsfile = open('./results/' + args.embedding + '_dimensions' + str(args.dimension) + '_' + str(args.nodes) + '_sample' + str(args.ratio) + '.csv', 'w')
        results = eavalute_accuracy_real(G1=GC1, G2=GC2, dimensions=args.dimension, embedding=args.embedding)
        resultsfile.write(str(args.dimension))
        for result in results:
            resultsfile.write(",{:.5f}".format(result))
        resultsfile.write('\n')
        print
        resultsfile.close()

def main():
    parser = ArgumentParser("DeepMatching",
							formatter_class=ArgumentDefaultsHelpFormatter,
							conflict_handler='resolve')

    parser.add_argument('--input', default='data/Email-Enron.txt',
						help='Input file of graph')

    parser.add_argument('--embedding', default='DeepWalk',
						help='Embedding algorithm')

    parser.add_argument('--nodes', default=500, type=int,
						help='Number of sub-graphs nodes in network')

    parser.add_argument('--ratio', default=0.8, type=float,
						help='Sample ratio of each network')

    parser.add_argument('--dimension', default=50, type=int,
						help='Indicates nodes features dimension in embedding algorithm')

    args = parser.parse_args()

    process(args)

if __name__ == "__main__":
    main()

