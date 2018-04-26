from __future__ import division
import math
import datetime

from initialMatching import *
from refinement import *

def consistent_edges(matches, G1, G2):
    '''
    Extract all the consistent edge between the two matching graphs based on the matches
    :param matches: A list of tuple (v_i, u_i), where v_i \in G_1, u_i \in G_2
    :param G1: the matching graph
    :param G2: the matching graph
    :return: A dict, where the keys are the consistent edges in G_1, the values are the consistent edges in G_2
    '''
    matchmapping = dict([(match[0], match[1]) for match in matches])
    cedges = {}
    for edge in G1.edges():
        if G2.has_edge(matchmapping.get(edge[0], -1), matchmapping.get(edge[1], -1)):
            cedges[edge] = (matchmapping.get(edge[0]), matchmapping.get(edge[1]))
    return cedges


def match_consistent_degree(matches, G1, G2):
    '''
    Calculate the consistent degree for each match in matches, the consistent degree is define the number of 
    consistent edges connected to the matching node in each graph.
    :param matches: A list of tuple (v_i, u_i), where v_i \in G_1, u_i \in G_2
    :param G1: the matching graph
    :param G2: the matching graph
    :return: A dict, where the keys are the matches, the values are the consistent degree defined above
    '''
    cedges = consistent_edges(matches, G1, G2)
    mcdeg = {}
    for G1_edge, G2_edge in cedges.items():
        mcdeg[(G1_edge[0], G2_edge[0])] = mcdeg.get((G1_edge[0], G2_edge[0]), 0) + 1
        mcdeg[(G1_edge[1], G2_edge[1])] = mcdeg.get((G1_edge[1], G2_edge[1]), 0) + 1
    return mcdeg


def maximum_consistency_matches(matches, G1, G2, nodenum_limit=7, cth = 2.0):
    '''
    Extract a sublist of matches in order to maximize the consistency between the two subgraphs. The two subgraphs 
    are extracted from the two matching graphs according to the sublist of matches. The consistency between two 
    graphs is defined as the ratio of the number of consistent edges between the two graphs over the maximun number 
    of edges of the two graphs. 
    :param matches: A list of tuple (v_i, u_i), where v_i \in G_1, u_i \in G_2
    :param G1: the matching graph
    :param G2: the matching graph
    :param nodenum_limit: The minimum number of nodes in each subgraph. 
    :param cth: The consistency threshold, a consistency below this threshold suggests a failed matching. 
    :return: A sublist of matches. An empty list represents a failed matching. 
    '''
    mcdeg = match_consistent_degree(matches, G1, G2)
    seeds = []
    for match, deg in mcdeg.items():
        if deg > cth:
            seeds.append(match)
    if len(seeds) < nodenum_limit:
        seeds = []
    return seeds


def consistency_sequence(matches, G1, G2, nodenum_limit=7, cth=2.0):
    '''
    Obtain a sequence of consistencies between a series of subgraphs extracted from G_1 and G_2. Firstly we sort the 
    matches according to their consistent degree in descending order.  Add one node in the node list recursively in 
    the order of the sorted matches. Then we extract subgraphs based on the node list and calculate the consistency. 
    graphs.
    :param matches: A list of tuple (v_i, u_i), where v_i \in G_1, u_i \in G_2
    :param G1: the matching graph
    :param G2: the matching graph
    :return: A list of consistencies. 
    '''
    cons_edges = consistent_edges(matches, G1, G2)
    mcdeg = {}
    for G1_edge, G2_edge in cons_edges.items():
        mcdeg[(G1_edge[0], G2_edge[0])] = mcdeg.get((G1_edge[0], G2_edge[0]), 0) + 1
        mcdeg[(G1_edge[1], G2_edge[1])] = mcdeg.get((G1_edge[1], G2_edge[1]), 0) + 1
    mcdeg = sorted(mcdeg.items(), key=lambda x: x[1], reverse=True)
    nodes1 = []
    nodes2 = []
    graph_matching_consistent_ratio_list = []
    graph_matching_z_score_list = []
    seeds = []
    for match, cred in mcdeg:
        seeds.append(match)
        nodes1.append(match[0])
        nodes2.append(match[1])
        subG1 = G1.subgraph(nodes1)
        subG2 = G2.subgraph(nodes2)
        gcr = graph_consistency(cons_edges, subG1, subG2)
        graph_matching_consistent_ratio_list.append(gcr)
    return graph_matching_consistent_ratio_list, graph_matching_z_score_list

def graph_consistency(consistent_edges, G1, G2):
    '''
    Calculate the consistency between two graphs
    :param consistent_edges: A dict, composed of the consistent edges
    :param G1: the matching graph
    :param G2: the matching graph
    :return: the consistency, a float value between 0.0 to 1.0. 
    '''
    G1_size = G1.size()
    G2_size = G2.size()
    if G1_size == 0 or G2_size ==0:
        return 0.0
    consistent_edge_count = 0.0
    for edge in G1.edges():
        if edge in consistent_edges or (edge[1], edge[0]) in consistent_edges:
            consistent_edge_count += 1.0
    return consistent_edge_count*1.0/max(G2_size, G1_size)

def random_mapping_parameters_estimate(G1, G2):
    '''
    Calculate the mean and the standard error
    :param G1: the matching graph
    :param G2: the matching graph
    :return: the mean value and the standard error value
    '''
    nodes1 = G1.nodes()
    nodes2 = G2.nodes()
    edge_consistency_list = []
    for i in range(100):
        random.shuffle(nodes1)
        random.shuffle(nodes2)
        matches = [(nodes1[i], nodes2[i]) for i in range(len(nodes1))]
        edge_consistency_list.append(mapping_consistency(matches, G1, G2))
    edge_consistency_list = np.array(edge_consistency_list)
    return (np.mean(edge_consistency_list), np.std(edge_consistency_list))

def mapping_consistency(matches, G1, G2):
    '''
    Calculate the consistency value of matched node pairs
    :param matches: the matched node pairs, which is a list of tuple
    :param G1: the matching graph
    :param G2: the matching graph
    :return: the graph consistency value
    '''
    nodes1 = [n1 for n1, n2 in matches]
    nodes2 = [n2 for n1, n2 in matches]
    subG1 = G1.subgraph(nodes1)
    subG2 = G2.subgraph(nodes2)
    cedges = consistent_edges(matches, G1, G2)
    return graph_consistency(cedges, subG1, subG2)

def mapping_credibility(matches, G1, G2):
    '''
    Calculate the credibility value of matched node pairs.
    :param matches: the matched node pairs, which is a list of tuple
    :param G1: the matching graph
    :param G2: the matching graph
    :return: the graph credibility value
    '''
    nodes1 = [n1 for n1, n2 in matches]
    nodes2 = [n2 for n1, n2 in matches]
    subG1 = G1.subgraph(nodes1)
    subG2 = G2.subgraph(nodes2)
    seed_edge_consistency = mapping_consistency(matches, subG1, subG2)
    m, s = random_mapping_parameters_estimate(subG1, subG2)
    cred = (seed_edge_consistency - m) / s
    return cred

def z_score(gcr, subG1, subG2, k_c):
    '''
    Calculate the z score of matched node pairs.
    :param gcr: consistency value of matched node pairs
    :param subG1: the matching graph comprise of nodes which are matched between two graph
    :param subG2: the matching graph comprise of nodes which are matched between two graph
    :param k_c: the constraint on the number of consistent edges
    :return: z score of two matched sub-graph
    '''
    subG1_m = nx.number_of_edges(subG1)
    subG2_m = nx.number_of_edges(subG2)
    subG1_n = nx.number_of_nodes(subG1)
    subG2_n = nx.number_of_nodes(subG2)
    if subG1_n <= 1 or subG2_n <= 1 or subG1_m == 0 or subG2_m == 0:
        return 0.0
    density_subG1 = (2 * subG1_m * 1.0) / (subG1_n * (subG1_n - 1))
    density_subG2 = (2 * subG2_m * 1.0) / (subG2_n * (subG2_n - 1))
    density = density_subG1 if density_subG1 <= density_subG2 else density_subG2
    if density == 1:
        return 0.0
    z = 0.0
    p1 = (9 * density) / (1 - density)
    p2 = (9 * (1 - density)) / density
    p = p1 if p1 >= p2 else p2
    if k_c >= p:
        z = ((gcr - density) * 1.0) / (density * (1 - density))
    return z


def get_matches(G1, G2):
    '''
    Get matches of two network graphs using bipartite matching, and the result is marked
        'matches'. Then calculate the matches' edge consistency and z score, so we can
        extract a credible seed set which is marked 'matches_ms' here.
    :param G1: the matching graph
    :param G2: the matching graph
    :return: two lists(matches and matches_ms) of tuple (v_i, u_i), where v_i \in G_1,
        u_i \in G_2
    '''
    count = 0
    count1 = 0
    matches = bipartite_matching(G1, G2)
    for match in matches:
        if match[0] == match[1]:
            count += 1
    print 'Accuracy:', count*1.0/G2.order()
    matches_ms = maximum_consistency_matches(matches, G1, G2)

    edge_consistency = mapping_consistency(matches_ms, G1, G2)
    print "edge consistency:", edge_consistency
    cred = mapping_credibility(matches_ms, G1, G2)
    print "Z score:", cred
    
    matches_seed = []
    for match in matches_ms:
        if match[0] == match[1]:
            count1 += 1
            matches_seed.append(match)
    if len(matches_ms) != 0 and count != 0:
        print "Pr:", count1/len(matches_ms), "Re:", count1/count
    print
    return matches, matches_ms

def load_network(input1, input2):
    '''
    Loading the network graphs.
    :param input1: the matching graph's file path
    :param input2: the matching graph's file path
    :return: two network graphs
    '''
    G1 = load_file(input1)
    G2 = load_file(input2)
    GC1 = max(nx.connected_component_subgraphs(G1), key=len)
    GC2 = max(nx.connected_component_subgraphs(G2), key=len)
    return GC1, GC2

def process(args):
    '''
    This is the main progress of DeepMatching, and after DeepMatching, there will be a
        phase to calculate the matched node's credibility, then using these seeds to
        propagation in the graph.
    '''
    time_start = datetime.datetime.now()
    G1, G2 = load_network(args.input1, args.input2)
    sub_G1 = get_subgraph(G1, nodes=args.nodes)
    sub_G2 = get_subgraph(G2, nodes=args.nodes)
    matches, matches_ms = get_matches(sub_G1, sub_G2)

    print "-------Starting propagation on " + str(args.nodes) + " nodes sub-graph-------"
    refinement(matches_ms, sub_G1, sub_G2, G1, G2, args.propa_num)
    time_end = datetime.datetime.now()
    time = (time_end - time_start).seconds
    print "All time: " + str(time / 60) + "min."

def main():
    parser = ArgumentParser("DeepMatching",
							formatter_class=ArgumentDefaultsHelpFormatter,
							conflict_handler='resolve')

    parser.add_argument('--input1', default='data/Email-Enron.txt',
						help='Input file of graph')

    parser.add_argument('--input2', default='data/Email-Enron.txt',
                        help='Input file of graph')

    parser.add_argument('--nodes', default=500, type=int,
						help='Number of sub-graphs nodes in network')

    parser.add_argument('--propa_num', default=2000, type=int,
                        help='Number of sub-graphs nodes in network, which is used in propagation phase')

    args = parser.parse_args()

    process(args)

if __name__ == "__main__":
    main()