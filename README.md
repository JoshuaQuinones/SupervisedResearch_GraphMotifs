Joshua Quinones U81851163
Program was written during the University of South Florida Spring 2021 Semester for Supervised Research, under Paul Rosen
This program reads a graph from a file, calculates "fingerprints" for each node using their neighborhood graphs and topology, and then draws the graph with similar nodes
drawn with the same colors.

The program has various inputs that are hardcoded, and the python file must be edited with a text editor to change them. 
Graphs that are read as input for the program should be included in the folder 'graphs'

Graph data was obtained from:
http://snap.stanford.edu/data/
and
http://networkrepository.com/index.php

The code has been split into 7 sections:

* Section 1:
Attempt to open the user input file, and use networkx to read the graph from the file. If 'drawGraph' is selected, the program will draw the graph that was stored in the file.
The graph then uses networkx's "ego_graph" function to create the ego graphs, or n-neighborhood graphs of each node in the graph. These subgraphs show all neighbors of the chosen  
node up to a certain distance 'n', specified by the user as the input 'neighbors'. This section will also draw a specific neighborhood graph for the "n'th" node selected in 
"subgraph_to_visit", or none if it is set equal to 0.

* Section 2:
To use the neighborhood graphs in ripser, which will compute their persistence cohomology, the representation of each subgraph must first be converted to a dissimilarity matrix, as 
this is the input that ripser requires. The dissimilarity matrix is a matrix that shows the distance between each node in the graph. This section computes these distances
for each node in each subgraph using the networkx "shortest_path_length" function. This can take a while when there is a large number of nodes in the original graph, or when
a larger 'n' is selected for the neighborhood graphs. This section will also draw each of the neighborhood graphs if the option "drawNeighborhoodGraphs" is true.

* Section 3:
After converting all neighborhod graphs to dissimilarity matrices, they will be given to ripser as input. Ripser calculates the persistence cohomology, outputting
the H0 and H1 values. Currently, only the H0 values are used as input for the next sections, and will be used as the "fingerprints" for each neighborhood graph. 
These values are printed as they are calculated, and then written to files in the "H0Values" folder. This directory is created if it does not already exist. 

* Section 4:
This section uses Hera to compute the Wasserstein distance between the "fingerprints", aka the graphs of the H0 values for each neighborhood graph. The persistence diagram 
for each node is compared to each other node, and a dissimilarity matrix for the original graph is created with the Wasserstein distance from each node to each other node.
This take a while when there are a large number of nodes.

* Section 5:
The resulting dissimilarity matrix is used as input for either a clustering algorithm (from numpy) or multi-dimensional scaling (from sklearn), depending on user input. 
If multi-dimensional scaling is selected, the program will print its output in this section and then end the program, as I did not yet add the code to display the 
output on a graph. If one of two clustering algorithms is selected, agglomerativeClustering or DBscan, then the output of these algorithms will be created and used 
in section 6. An input can be given for each algorithm by the user in the options at the top of the program. The number of clusters to create for agglomerativeClustering 
can be selected, and the maximum distance between samples in a cluster can be given for DBscan. Other arguments are available for the functions used, but would need to be
added in this section instead of in the user input at the top of the program.

* Section 6:
The results of the program, the original graph with similar nodes having the same colors, is calculated and displayed. First, the output of the algorithm used is displayed, 
which will be presented in a way that doesn't match the original graph since a dissimilarity matrix was used as input (only has distances between nodes, not their positions). 
Then, a list of colors is created by first randomly generating a a color for each cluster that exists. After generating the list of colors to use, a sequence of colors is created 
by going through the original graph's nodes in the order they are stored, and adding the nodes' accompanying colors to a colors list in the same order. This is used as an input when 
displaying the graph so that each node is displayed with the correct color. Once the graph has been displayed, the program will close.

* What's Incomplete
The program has been tested with weighted, undirected graphs. I don't know if it will work with unweighted graphs or directed graphs, and don't know what kind of fixes or changes
may be required to make them work. The program also does not use the MDS output to color similar nodes, it only prints the output from the MDS calculation. 

* Issues I ran into while writing the program
When downloading graphs to test the program with, I initially had trouble with graphs that could not be read by the parsing function in networkx. It took me a little while to 
realize that these graphs were actually compressed archives with filetypes like .bz2 or .tar, which can be opened in Windows using 7zip.

The program was written and tested in Windows. Installing Hera, which uses C++, was confusing and getting its required libraries (like the C++ Boost libraries) were difficult
to get working. The easiest way to get Hera and its required packages working was to install the C++ build tools in Visual Studio, integrate vcpkg, and use CMake within
Visual Studio with the CMakeLists file included with Hera. Once this was done, an exe for the Wasserstein algorithm was automatically compiled which I moved to the same directory
as the python code to make it easier for the python program to call and use this exe. 
https://docs.microsoft.com/en-us/cpp/build/cmake-projects-in-visual-studio?view=msvc-160

The program was initially being written using unweighted graphs, but it was difficult to get meaningful results from these graphs, so I switched to writing the program
with weighted graphs in mind midway through.

Not all clustering algorithms can use a dissimilarity matrix as input. Instead of MDS, dimension reduction was initially considered but the method I wanted to use for it,
principal component analysis, could not take a dissimilarity matrix as input either. I am unsure if other dimension reduction algorithms can work with this type of input.

Running the program with large graphs or using a large 'n' for neighborhood graphs takes a long time, so the program was primarily tested with the 'out.moreno_lesmis_lesmis' graph
with an 'n' of 2. The neighborhood graphs and Wasserstein distances can be computed in under 5 minutes and provide meaningful results.
