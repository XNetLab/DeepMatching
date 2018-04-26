--------------------
DeepMatching
--------------------
--------------------
A brief introduction
--------------------
If you want to test the effection of parameters such as 'embedding', 'ratio' and 'dimension'(you can see the detail usage below), please try 'Example Usage-1'. Or, if you want to test the whole process of DeepMatching(initial matching, extract a credibility seed set, and then using the seed set to propagation in the graph), please try 'Example Usage-2'.

-----
Usage
-----
-------------------
**Example Usage-1**
-------------------
	''python initialMatching.py --input ./data/Email-Enron.txt --embedding DeepWalk --nodes 500 --ratio 0.8 --dimension 64''
-----------
Parameters:
-----------
**--input**:	*network graph file path*

**--embedding**:	*embedding_algorithm*
	1. ''DeepWalk''
	2. ''Node2Vec''

**--nodes**:	*number of sub-graph's nodes*
	This parameter should be a number, e.g::
	300, 500, ...
	In our experiment, nodes=3000 will be more slow.

**--ratio**:	*sample_ratio*
	The sample ratio should be a number between 0 and 1, e.g::
	0.5, 0.55, 0.6, 0.65, ... ,0.9 ...

**--dimension**:	*assign the dimension used in the embedding algorithm*
	This parameter is used in the embedding algorithm, in order to generate a vector who has the specified dimension.
	e.g:: 20, 30, 40, 50, ..., 120 .... 
	In our experiment, we using dimension=64.

**Full Command List**
	The full list of command line options is available with ''initialMatching.py --help''

After running this algorithm, the results will be wrote in file and save in 'results' folder.

-------------------
**Example Usage-2**
-------------------
	''python DeepMatching.py --input1 ./data/Email-Enron.txt  --input2 ./data/Email-Enron.txt --nodes 500 --propa_num 2000''
-----------
Parameters:
-----------
**--input1**:	*network graph file path*

**--input2**:	*network graph file path*

**--nodes**:	*number of sub-graph's nodes*
	This parameter should be a number, e.g::
	300, 500, ...

**--propa_num**:	*number of a larger sub-graph's nodes*
	This parameter should be a number, and please note this number should be larger than the number of 'nodes'. e.g:: 
	300, 500, ...
	In our experiment, nodes=3000 will be more slow.

**Full Command List**
	The full list of command line options is available with ''DeepMatching.py --help''

After running this algorithm, the results will be wrote in file and save in './data/subgraphs/' folder. And there will generate three files names 'credibility_matches.txt', '1st_propagation.txt', '2nd_propagation.txt', respectively.

------------
Requirements
------------
numpy==1.13.3

scipy==1.0.0

gensim==3.1.0

networkx==1.11

sklearn==0.0

futures==3.1.1

psutil==5.4.1

matplotlib==1.13.1

------------
Installation
------------
#. pip install -r requirements.txt

-----------------
Important notice：
-----------------
1. Before launching DeepMatching, you must fix a bug in package 'gensim' by yourself::
    $python27/Lib/site-packages/gensim/models, then find 'word2vec.py' and open it. Find the function 'reset_weights(self)'. In this funtion, you will see the line 'self.wv.syn0[i] = self.seeded_vector(self.wv.index2word[i] + str(self.seed))'. Modify it as 'self.wv.syn0[i] = self.seeded_vector(str(self.wv.index2word[i]) + str(self.seed))'.

2. Please note that we STRONGLY recommend users to use the environment same as us, especially the version of  'networkx' package. Because after many times tests, we found that different version of this package will has a big influence to our algorithm(There are some changes in 'networkx' in different version).
