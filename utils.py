
#
# Some functions to be used in the tutorial
#
# Developed by Debora Cristina Correa

import pandas as pd
import matplotlib.pyplot as plt # for 2D plotting
import numpy as np
import ordpy
import seaborn as sns # plot nicely =)
import networkx as nx
from tqdm import tqdm


def train_test_split( data, train_size =0.8):
    '''
    data: windowed dataset in DataFrame format
    train_size: size of the training dataset
    '''

    #n_train = round(0.8 * dataset.shape[0])
    #dataset_80 = dataset.iloc[:n_train,:]
    #dataset_20 = dataset.iloc[n_train:,:]

    #x_train = dataset_80.iloc[:,:-1]
    #y_train = dataset_80.iloc[:,-1]

    #x_test = dataset_20.iloc[:,:-1]
    #y_test = dataset_20.iloc[:,-1]

    nrow = round(train_size * data.shape[0])

    # iloc allows the using of slicing operation and returns
    # the related DataFrame. Note that, this is different of using 
    # data.values, in which the returned elements are numpy.array
    train = data.iloc[:nrow, :] # train dataset
    test = data.iloc[nrow:, :]  # test dataset

    train_X = train.iloc[:, :-1]
    test_X = test.iloc[:, :-1]

    train_Y = train.iloc[:, -1]
    test_Y = test.iloc[:, -1]

    return train_X, train_Y, test_X, test_Y

def get_net_properties_window(m,df,label,list_feat,win_len):
    
    df_feat = pd.DataFrame(columns=list_feat)
    for i in range(len(df)-win_len):
        data = np.array(df.iloc[i:i+win_len])
        vertices, edges, weights = ordpy.ordinal_network(data, dx=m,overlapping=True) #creating the ordinal network

        G = nx.Graph()
        G.add_nodes_from(vertices)
        G.add_edges_from(edges)

        measures, list_feat = calculate_basic_network_measures(G) 
        df_feat = df_feat.append({feat:measures[i] for i,feat in enumerate(list_feat)}, ignore_index=True)
    
    df_feat['Class'] = label
    
    return df_feat

def get_net_ordinal_properties_window(m,df,label,list_feat,win_len):
    
    df_feat = pd.DataFrame(columns=list_feat)
    for i in tqdm(range(len(df)-win_len)):
        data = np.array(df.iloc[i:i+win_len])
        data[-1] = data[0]
        
        dist = ordpy.ordinal_distribution(data,dx=m, return_missing=True)
        pe = ordpy.complexity_entropy(data, dx=m)[0]
        ord_sequence = ordpy.ordinal_sequence(data, dx=m)
        s_matrix = compute_stochastic_matrix(ord_sequence)
        c_pe = compute_conditional_pe(s_matrix,dist[1])
        g_node_e = ordpy.global_node_entropy(data,dx=m)
        fraction_missing_patterns = ordpy.missing_patterns(data,dx=m, return_missing=False)

        measures = [fraction_missing_patterns, g_node_e, c_pe, pe]
        df_feat = df_feat.append({feat:measures[i] for i,feat in enumerate(list_feat)}, ignore_index=True)
    
    df_feat['Class'] = label
    
    return df_feat

def calculate_basic_network_measures (G):

    cc = nx.average_clustering(G)
    aspl = nx.average_shortest_path_length(G)

    #closeness centrality
    closeness_centrality = nx.closeness_centrality(G)
    all_ = [closeness_centrality[k] for k in closeness_centrality]
    closeness_centrality = np.mean(all_)

    transit = nx.transitivity(G)
    self_loops = nx.number_of_selfloops(G)

    #cycle basis
    cycles = nx.cycle_basis(G)
    len_cycles = np.zeros(len(cycles))
    for i in np.arange(len(cycles)):
        len_cycles[i] = len(cycles[i])
    mean_cycle_length = np.mean(len_cycles)

    n_cliques = nx.graph_number_of_cliques(G)

    #V/diameter (Kostas' paper)
    net_diam = nx.diameter(G)
    if(net_diam):
        n_nodes = nx.number_of_nodes(G)
        net_inv_diam = n_nodes / net_diam
    else:
        net_inv_diam = 0

    n_nodes = G.number_of_nodes()
    
    #average degree
    degrees = G.degree
    av_degree = sum(deg for n, deg in degrees)/n_nodes

    #av degree centrality
    degree_centrality = nx.degree_centrality(G)
    all_centralities = [degree_centrality[k] for k in degree_centrality]
    av_degree_centrality = np.mean(all_centralities)

    #link density
    density = nx.density(G)

    list_feat = ('av_clust_coeff', 'av_shortest_path_length', 'closeness_centrality', 'transitivity', 'n_self_loops', 'mean_cycle_length', 'n_cliques', 'net_inv_diam', 'n_nodes', 'av_degree', 'av_degree_centrality', 'density')

    return [cc, aspl, closeness_centrality, transit, self_loops, mean_cycle_length, n_cliques, net_inv_diam, n_nodes, av_degree, av_degree_centrality, density], list_feat


def get_cycles_distribution(m,data):

    vertices, edges, weights = ordpy.ordinal_network(data, dx=m) #creating the ordinal network

    G = nx.Graph()
    G.add_nodes_from(vertices)
    G.add_edges_from(edges)

    cycles = nx.cycle_basis(G)
    len_cycles = np.zeros(len(cycles))
    for i in np.arange(len(cycles)):
        len_cycles[i] = len(cycles[i])
    mean_cycle_length = np.mean(len_cycles)

    return len_cycles, mean_cycle_length

#from ordpy
def logistic(a=4, n=100000, x0=0.4):
    x = np.zeros(n)
    x[0] = x0
    for i in range(n-1):
        x[i+1] = a*x[i]*(1-x[i])
    return(x[1000:])

def compute_stochastic_matrix(ord_sequence):

    ord_sequence = np.concatenate((ord_sequence,[ord_sequence[0]]),axis=0)
    
    unique_patterns = np.unique(ord_sequence, axis=0)
    n_patterns = len(unique_patterns)

    value2index = {}
    for i,p in enumerate(unique_patterns):
        value2index[np.str(p)] = i

    s_matrix = np.zeros((n_patterns,n_patterns))
    for p in unique_patterns:
        for j in range(1,len(ord_sequence)):
            p2 = ord_sequence[j-1]
            p3 = ord_sequence[j]
            if(np.array_equal(p,p2)):
                s_matrix[value2index[np.str(p)],value2index[np.str(p3)]]+=1
    
    s_matrix_norm = np.zeros((n_patterns,n_patterns))
    for i in range(np.shape(s_matrix)[0]):
        s_matrix_norm[i,:] = s_matrix[i,:]/np.sum(s_matrix[i,:])

    return s_matrix_norm

def compute_conditional_pe(s_matrix,p):

    c_pe_part = np.zeros(np.shape(s_matrix)[0])
    sum_prod = 0
    for i in range(np.shape(s_matrix)[0]):
        for j in range(np.shape(s_matrix)[1]):
            if(s_matrix[i,j]>0):
                sum_prod = sum_prod + s_matrix[i,j]*np.log(s_matrix[i,j])
        c_pe_part[i] = -p[i]*sum_prod
    
    conditional_pe = np.sum(c_pe_part)

    return conditional_pe