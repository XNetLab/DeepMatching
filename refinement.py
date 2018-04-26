#!/usr/bin/python
from __future__ import division
from DeepMatching import *

def match_propagation(matches, G1, G2):
    '''
    Start from the seeds, matching the remaining nodes of the two graph gradually.
    In each step, the top nodes whose degrees are greater than 2^i, i=\log n, ..., 1 are matched based on the seeds.
    If the two matching nodes share at least 3 matched neighbors, then the two nodes are also matched.
    :param matches: The initial seeds
    :param G1: the graph
    :param G2: the graph
    :return: a larger match list
    '''
    # Seeds = matches
    Seeds = [item for item in matches]
    deg1 = G1.degree()
    deg2 = G2.degree()
    maxD1 = max([deg for node, deg in deg1.items()])
    maxD2 = max([deg for node, deg in deg2.items()])
    maxD = max(maxD1, maxD2)

    rest_matches = []
    for i in range(1, int(np.log2(maxD)))[::-1]:
        limited_deg = 2 ** i
        newmatches = propagation_phase(Seeds, G1, G2, limited_deg)
        # print "limited_deg: %d\tlegth of newmatches: %d\tlength of Seeds: %d" %(limited_deg, len(newmatches), len(Seeds))
        Seeds.extend(newmatches)
        rest_matches.extend(newmatches)

    count = 0
    # Seeds = rest_matches
    for item in Seeds:
        if item[0] == item[1]:
            count += 1
    if len(Seeds) == 0:
        rate = 0.0
    else:
        rate = count * 1.0 / len(Seeds)

    return Seeds, count, rate


def propagation_phase(Seeds, G1, G2, limitdeg):
    '''
    Match the nodes whose degrees are greater than limitdeg. If the two matching nodes share at least 3 matched neighbors, then the two nodes are also matched.
    :param Seeds: The seeds
    :param G1: the graph
    :param G2: the graph
    :param limitdeg: the limited degree
    :return: A list of newly matched nodes, some matched nodes may also be included.
    '''
    G1matched = set([s1 for s1, s2 in Seeds])
    G2matched = set([s2 for s1, s2 in Seeds])
    WN = {}
    for seed in Seeds:
        N1 = G1.neighbors(seed[0])
        N2 = G2.neighbors(seed[1])
        for n1 in N1:
            # if G1.degree(n1)<limitdeg:#n1 in G1matched or
            if G1.degree(n1) < limitdeg or n1 in G1matched:
                continue
            for n2 in N2:
                # if G2.degree(n2) < limitdeg: # n2 in G2matched or
                if G2.degree(n2) < limitdeg or n2 in G2matched:
                    continue
                if n1 not in WN:
                    WN[n1] = {}
                    WN[n1][n2] = 1
                else:
                    WN[n1][n2] = WN[n1].get(n2, 0) + 1
    newmatches = []
    for key, cand in WN.items():
        maxWN = max([(n2, wn) for n2, wn in cand.items()], key=lambda x: x[1])
        if maxWN[1] > 2:
            newmatches.append((key, maxWN[0], maxWN[1]))
    dupnodes = {}
    for match in newmatches:
        # dupnodes.get(match[1], []).append(match)
        if match[1] not in dupnodes:
            dupnodes[match[1]] = [match]
        else:
            dupnodes[match[1]].append(match)
    newmatches = []
    for dupnode, matches in dupnodes.items():
        newmatches.append(max(matches, key=lambda x: x[2]))
    return [(n1, n2) for n1, n2, wits in newmatches]

def read_matches(filename):
    '''
    Read matches from a file
    :param filename: the filename of the file storing the matches
    :return: a list of matches
    '''
    matches = []
    for line in open(filename, 'r'):
        line = line.strip()
        nodes = line.split(',')
        matches.append((string.atoi(nodes[0]), string.atoi(nodes[1])))
    return matches

def read_graph(file_path):
    '''
    Read graph from a file
    :param filename: the filename of the file storing the matches
    :return: a list of nodes from a graph
    '''
    G = nx.Graph()
    txt_reader = open(file_path, 'rb')
    for item in txt_reader:
        if "#" not in item:
            if " " in item:
                item = item.split(' ')
                G.add_edge(int(item[0]), int(item[1]))
            else:
                item = item.split('\t')
                G.add_edge(int(item[0]), int(item[1]))
    txt_reader.close()
    return G

def accuracy_propagation(matches, G1, G2):
    '''
    Propagation using the matched node pairs(seeds) on the graph. In this phase, the seed
        set is expand gradually based on the principle that neighborhood preserves mapping.
        After propagation, the edge consistency of the matched node pairs will be calculated.
    :param matches: a list of tuple(v_i, u_i), where v_i \in G_1, u_i \in G_2
    :param G1: the matching graph
    :param G2: the matching graph
    :return: a list of tuple(v_i, u_i), where v_i \in G_1, u_i \in G_2
    '''
    refined_matches, count, rate = match_propagation(matches, G1, G2)
    count_match = 0
    count1_mismatch = 0
    matches_ms = maximum_consistency_matches(refined_matches, G1, G2)
    edge_consistency = mapping_consistency(matches_ms, G1, G2)
    print "edge consistency:", edge_consistency
    for match in refined_matches:
        if match[0] == match[1]:
            count_match += 1
        if match[0] != match[1]:
            count1_mismatch += 1
    print "refine accurate = %.2f" % rate
    count_consist = 0
    for match in matches_ms:
        if match[0] == match[1]:
            count_consist += 1
    if len(matches_ms) != 0 and count != 0:
        print "Pr:", count_consist/len(matches_ms), "Re:", count_consist/count_match
    print
    return refined_matches

def refinement(matches, sub_G1, sub_G2, G1, G2, propa_num):
    '''
    This is the main progress of propagation phase after calculating the matched node's
        credibility. Here we first using the matched node pairs to propagation on the
        sub-graph, and get a larger seed set. Then using the larger seed set continue to
        propagation on a larger graph to get more matched node pairs.
    :param matches: a list of tuple(v_i, u_i), where v_i \in G_1, u_i \in G_2
    :param sub_G1: the matching graph
    :param sub_G2: the matching graph
    :param G1: the matching graph
    :param G2: the matching graph
    :param propa_num: the nodes' number of larger matching graph
    '''
    refined_matches = accuracy_propagation(matches, sub_G1, sub_G2)
    sub_Gs1 = get_subgraph(G1, nodes=propa_num)
    sub_Gs2 = get_subgraph(G2, nodes=propa_num)
    print "Get " + str(propa_num) + " nodes sub-network done!"
    print "-------Starting propagation on " + str(propa_num) + " nodes sub-graph-------"
    accuracy_propagation(refined_matches, sub_Gs1, sub_Gs2)

if __name__ == '__main__':
    main()
