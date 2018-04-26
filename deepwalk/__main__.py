#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import random
from io import open
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import logging

from deepwalk import graph
from deepwalk import walks as serialized_walks
from gensim.models import Word2Vec
from skipgram import Skipgram

from six import text_type as unicode
from six import iteritems
from six.moves import range

import psutil
from multiprocessing import cpu_count


p = psutil.Process(os.getpid())
try:
    p.set_cpu_affinity(list(range(cpu_count())))
except AttributeError:
    try:
        p.cpu_affinity(list(range(cpu_count())))
    except AttributeError:
        pass

logger = logging.getLogger(__name__)
LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

def process(edge_list, undirected=True, number_walks=10, walk_length=40, window_size=5, workers=1, dimensions=64, max_memory_data_size=1000000000, seed=0, vertex_freq_degree=False):
    G = graph.load_edgelist(edge_list, undirected=undirected)

    num_walks = len(G.nodes()) * number_walks

    data_size = num_walks * walk_length

    if data_size < max_memory_data_size:
    #  print("Walking...")
      walks = graph.build_deepwalk_corpus(G, num_paths=number_walks,
                                        path_length=walk_length, alpha=0, rand=random.Random(seed))
    #  print("Training...")
      model = Word2Vec(walks, size=dimensions, window=window_size, min_count=0, workers=workers)
    else:
    #  print("Data size {} is larger than limit (max-memory-data-size: {}).  Dumping walks to disk.".format(data_size, max_memory_data_size))
    #  print("Walking...")

      walks_filebase = "karate.embeddings" + ".walks"
      walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase, num_paths=number_walks,
                                         path_length=walk_length, alpha=0, rand=random.Random(seed),
                                         num_workers=workers)

    #  print("Counting vertex frequency...")
      if not vertex_freq_degree:
        vertex_counts = serialized_walks.count_textfiles(walk_files, workers)
      else:
        # use degree distribution for frequency in tree
        vertex_counts = G.degree(nodes=G.iterkeys())

    #  print("Training...")
      model = Skipgram(sentences=serialized_walks.combine_files_iter(walk_files), vocabulary_counts=vertex_counts,
                     size=dimensions,
                     window=window_size, min_count=0, workers=workers)

    #model.save_word2vec_format("karate.embeddings")
    return model

