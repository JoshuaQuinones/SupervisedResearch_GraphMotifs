# Joshua Quinones U81851163
# Program was worked on during the University of South Florida Spring 2021 Semester for Supervised Research, under Paul Rosen
# This program reads a graph from a file and uses networkx to create n-neighborhood subgraphs for each node and then converts those subgraphs into dissimilarity matrices. These dissimilarity matrices are then used as input for ripser, whose output, a persistence diagram, will be used as a "fingerprint" for the graph.
# These fingerprints are then written to files, which are used as inputs for Hera. Hera, using Wasserstein distance, determines how similar the fingerprints are, and these measures of similarity are used to create a dissimilarity matrix for all nodes in the original graph. This dissimilarity matrix is then inputted
# into either the DBSCAN clustering algorithm, AgglomerativeClustering algorithm, or MDS. Outputs are then drawn on graphs, and the original graph is displayed with colored nodes, with nodes that were 'similar' sharing colors.

import networkx as nx
import matplotlib.pyplot as p
import subprocess as s
import os

#ripser documentation: https://ripser.scikit-tda.org/en/latest/
import numpy as np
from ripser import ripser
from persim import plot_diagrams

import sklearn.cluster
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import matplotlib.pyplot as plt
import random

#**********************************
filename = 'out.moreno_lesmis_lesmis' #file containing the graph to calculate the neighborhood graph of
weighted = True
neighbors = 2 #will calculate n-neighborhood graphs according to this value
drawGraph = True #draw the input graph at the start of running the program, after reading it from the input file
subgraph_to_visit = 0 #draw a specific neighborhood graph. Set to 0 to not draw a specific neighborhood graph
drawDiagrams = False #draw resulting diagrams from ripser, which computes the persistence cohomology of the neighborhood graphs.
                     #Will lead to errors later in the program.
drawNeighborhoodGraphs = False #draw all neighborhood graphs after calculating them.
                               #Will lead to errors later in the program
algorithmUsed = 1 #which algorithm to use to determine similar nodes. 1 is DBscan clustering algorithm, 2 is agglomerative clustering with specified number of clusters, 3 is MDS (whose output is currently unused)
epsValue = 2.7 #initial value 2.5, float value used for DBscan algorithm. The maximum distance between two samples that can be considered part of the same neighborhod. Samples are given as dissimilarity matrix from the distances between neighborhood graphs
               #also used with agglomerative clustering with distance threshold as the distance threshold above which clusters will not be merged
linkageToUse = 'average' #linkage value to input to AgglomerativeClustering. Can do 'average' which uses the average distances of the observations in the two sets, 'maximum' which uses the max distance between all observations of the two sets, and 'single' which
                      #uses the minimum distance betwen all observations
numClusters = 6 #number of clusters to try and find when using AgglomerativeClustering algorithm. Default for function used is 2
ndimensions = 2 #number of components input for MDS. Default for function is 2.
# **********************************

# SECTION 1: Read graph from file
try:
    f = open('graphs/' + filename)
    if (weighted):
        g = nx.parse_edgelist(f,  data=(("weight", float),))
    else:
        g = nx.parse_edgelist(f)
    if (drawGraph):
        nx.drawing.nx_pylab.draw_networkx(g)
        p.show()
    # Dont need file anymore. Close
    f.close()
    # Note to self: concat strings with other values when printing using commas https://www.codespeedy.com/how-to-print-string-and-int-in-the-same-line-in-python/
    print("Order: ",nx.Graph.order(g))
# Find neighbors of every node in the main graph, and store a subgraph for each node in a list called ngList (neighborhood graph list)
    # ngList will be a dictionary, where the keys are strings holding the values of the node that the neighborhood graph is built around
    ngList = dict()
    #iterate through every node in the graph G, find the neighborhood graph of that node, and place into a container
    for n in g:
        g2 = nx.generators.ego.ego_graph(g,n,neighbors)
        ngList.update({n : g2})
    #Will draw the n-th subgraph according to subgraph_to_visit, or none of them if the value entered is 0
    if (subgraph_to_visit > 0):
        count = 0
        for x in ngList:
            count += 1
            if (count == subgraph_to_visit):
                print("Drawing " + str(neighbors) + "-neighborhood graph for node " + x)
                nx.draw_networkx(ngList[x])
                p.show()
            if (count >= subgraph_to_visit):
                break;

# SECTION 2: Convert neighborhood graphs to dissimilarity matrices
    # Create dissimilarity matrix for each neighborhood graph, a 2D matrix where each element is a distance between two nodes in the graph
    dmDict = dict() # Where dissimilarity matrices will be stored once they are created
    dm = [] # dissimilarity matrix
    dmRow = [] # one row to be added to the dissimilarity matrix, has the shortest path distances of every node compared to the node this row's index represents
    rIndex = 0 # index of the current row being built in the dissimilarity matrix
    dmIndex = dict() # dict that will map the element a node represents to its index in the dissimilarity matrix
    count = 0 # How many times gone through the loop, and how many neighborhood graphs have been created
    for x in ngList:
        # Make a dissimilarity matrix for the current graph x in the neighborhood graph list. Go through row by row for each element 'y' and get shortest path lengths of all other elements 'z'
        for y in ngList[x]:
            # Add mapping of current element 'y' to its index rIndex in dmIndex. Index will be the key given to return the value
            dmIndex.update({rIndex : y})
            print({y: rIndex})
            for z in ngList[x]:
                if (y == z):
                    dmRow.append(0)
                else:
                    sp = nx.shortest_path_length(ngList[x],y,z, weight = 'weight')
                    dmRow.append(sp)
            # Have appended all values for z to dmRow. Append to dissimilarity matrix dm and clear dmRow, iterate row index
            dm.append(dmRow)
            dmRow = []
            rIndex += 1
    #print current dissimilarity matrix, along with the neighborhood graph if drawNeighborhoodGraphs = True
        if (drawNeighborhoodGraphs):
            print("Drawing current neighborhood graph for " + x)
            nx.drawing.nx_pylab.draw_networkx(ngList[x])
            p.show()
            print() #newline
        print("\t\t\t", end = "")
        for y in dm:
            print(dmIndex[dm.index(y)], end = ":\t\t")
        print() #newline
        for y in dm:
            print(dmIndex[dm.index(y)] + ":\t\t\t", end = "")
            for z in y:
                print(z, end = "\t\t")
            print() #newline
        print("\nNext Graph\n")
        dmDict.update({x : dm}) #add finished dm to list of dms for neighborhood graphs
        dm = [] #set dm equal to a new, empty list so the next iteration of this loop doesn't use the previous loop's information. Don't use clear, it erases previously added dm from dmList
        dmIndex = dict() #set dmIndex to an empty dictionary
        rIndex = 0 #reset rIndex to 0 for new graph that will be used in the next loop

# SECTION 3: Now have the dissimilarity matrices of all neighborhood graphs. Create output files of H0 values for all dissimilairy matrices.
    # Create folder to hold output files, if it doesn't already exist
    curr_dir = os.getcwd()
    h0dir = os.path.join(curr_dir, 'H0Values')
    if not os.path.exists(h0dir):
        os.mkdir(h0dir)
    count = 0
    for a in dmDict:
        # Need to convert data to numpy array
        dmpy = np.asarray(dmDict[a]) 
        diagrams = ripser(dmpy, distance_matrix = True)['dgms']
        if (drawDiagrams):
            plot_diagrams(diagrams, show=True)
    # Write diagram to file, get x and y coordinates of H0 and H1 values
        out = open('H0Values/H0-values-' + a + '.txt', 'w')
        #Get only H0 values first
        out.write("#H0\n")
        for x in diagrams[0]:
            for y in x:
                out.write('{0:.18f}'.format(y.item()) + " ") #specify precision to ensure the float isn't written in scientific notation, so it is easier to read in the accompanying C++ program
            out.write("\n")
        out.close()
        count += 1

# SECTION 4: Now have all the files written for all neighborhood graphs. Compare all neighborhood graphs to eachother to create another dissimilarity matrix of distance values
    distDM = [] #the dissimilarity matrix made up of distance values between neighborhood graphs of nodes in the input graph
    distDMRow = [] #one row of the dissimilarity matrix. Refers to all distance values for one neighborhood graph, compared to all others (including itself)
    nodeOrder = [] #the order of the nodes stored in dmDict, in case we need to know which row on the dissimilarity matrix corresponds to which node in the graph
    for a in dmDict:
        nodeOrder.append(a) #when this loop ends, will have all nodes in distDM in order in nodeOrder
        for b in dmDict:
            # documentation for subprocess .run() https://docs.python.org/3/library/subprocess.html
            stringToRun = 'wasserstein_dist.exe'
            arg1 =  'H0Values/H0-values-' + a + '.txt'
            arg2 =  'H0Values/H0-values-' + b + '.txt'
            wass = s.run([stringToRun, arg1, arg2], capture_output=True, text = True)
            distVal = float(wass.stdout)
            print('compare ' + a + ' to ' + b + ': ' + str(distVal))
            distDMRow.append(distVal)
        distDM.append(distDMRow) #add row for this node to the dissimilarity matrix
        distDMRow = [] #set distDMRow equal to a new empty list for the next node to use in the loop
# SECTION 5: After the loop, now have the dissimilarity matrix. Use as input for user-selected algorithm. Code for displaying clustering algs from: https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
    # first, convert dissimilarity matrix (list) to nparray
    dmpy = np.asarray(distDM)
    X = dmpy
    if algorithmUsed == 1: #compute DBSCAN
        cl = DBSCAN(metric = 'precomputed', eps = epsValue).fit(X)
        algorithmName = 'DBscan'
        core_samples_mask = np.zeros_like(cl.labels_, dtype=bool)
    elif algorithmUsed == 2: #compute AgglomerativeClustering using number of clusters
        cl = AgglomerativeClustering(affinity = 'precomputed', n_clusters = numClusters, linkage = linkageToUse).fit(X)
        algorithmName = 'AgglomerativeClustering'
        core_samples_mask = np.zeros_like(cl.labels_, dtype=bool)
    elif algorithmUsed == 3: #compute MDS. Reference used for computation, includes code for drawing that didn't work in this case https://towardsdatascience.com/visualize-multidimensional-datasets-with-mds-64d7b4c16eaa
        embedd = MDS(n_components=ndimensions, dissimilarity='precomputed', random_state=0)
        mdsOutput = embedd.fit_transform(X)
        print('MDS output is', mdsOutput.shape)
        for i in mdsOutput:
            print(i)
        print('press Enter to end program')
        input()
        exit(0) #stop program early, since not using output for MDS
    if algorithmUsed == 1:
        core_samples_mask[cl.core_sample_indices_] = True
        labels = cl.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)
        print("Silhouette Coefficient: %0.3f"
              % metrics.silhouette_score(X, labels))
    elif algorithmUsed == 2:
        labels = cl.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# SECTION 6: Plot result, showing the original graph with similar nodes having the same color
    if algorithmUsed == 1 or algorithmUsed == 2:
        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]
            class_member_mask = (labels == k)
            xy = X[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=14)
            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=6)
        plt.title(algorithmName + '.  Number of clusters: %d' % n_clusters_)
        for a in nodeOrder: 
            print(a, end = ' ')
        print()
        plt.show()
    if algorithmUsed == 3:
        print() #would place code for drawing MDS output here
    # Match labels to which clusters they are in, based on their indices in the labels list and nodes list
    # clusters will be a list of lists, each inner list holding the nodes in a cluster
    clusters = []
    # Add an empty list for each cluster
    for x in range(n_clusters_):
        clusters.append([])
    # Go through each label in labels, keeping track of its index, and if the label is not -1 (which means it is noise) then add it to the appropriate list in clusters
    index = 0
    for lab in labels:
        if not lab == -1:
            clusters[lab].append(nodeOrder[index])
        index += 1
    # Print each clustesr
    print('Nodes in each cluster:')
    index = 0
    for clu in clusters:
        print('[Cluster ' + str(index), end = ': ')
        for nod in clu:
            print(nod, end = ' ')
        print(']') 
# In each cluster, find and print the distance between each node to each other node (dissimilarity matrix)
    print('\nCluster distances:')
    index = 0
    for clu in clusters:
        print('Cluster ' + str(index), end=' consists of : ')
        for nod1 in clu:
            print(nod1, end = ' ')
        print('\nDistance matrix:')
        # print distance matrix of clusters
        for nod1 in clu:
            for nod2 in clu:
                # already computed in distDM, find the indices of these nodes in nodeOrder and use to get distance from distDm
                nod1Ind = nodeOrder.index(nod1)
                nod2Ind = nodeOrder.index(nod2)
                print(distDM[nod1Ind][nod2Ind], end = ' ')
            print() #newline
        index += 1
# Create a list of colors corresponding to nodes in graph 'g', color the nodes according to the cluster they are in
    # reference: https://stackoverflow.com/questions/27030473/how-to-set-colors-for-nodes-in-networkx
    nodeColors = [] # will hold the colors for each node in 'g' in order
    clustColors = [] # will hold the colors associated with each cluster
    # first, determine colors that will be associated with each cluster. Generate a list of colors in the order of the clusters they are associated with. Reference: https://www.kite.com/python/answers/how-to-generate-a-random-color-for-a-matplotlib-plot-in-python
    for clu in clusters:
        red = random.random()
        green = random.random()
        blue = random.random()
        nextColor = (red, green, blue)
        clustColors.append(nextColor)
    nonCluster = (0.9,0.9,0.9) #light grey color for nodes that aren't part of a cluster
    # now go through each node in g and determine which cluster its in, adding its color to nodeColors
    for nod in g:
        index = 0
        clusterFound = False
        for clu in clusters:
            if nod in clu:
                nodeColors.append(clustColors[index])
                clusterFound = True
            index += 1
        if not clusterFound:
                nodeColors.append(nonCluster)
    # display the graph along with the colors in nodeColors
    nx.draw(g, node_color=nodeColors, with_labels=True)
    p.show()
except IOError:
    print('Could not open file')