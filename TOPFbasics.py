import numpy as np
import gudhi as gd
from gudhi.datasets.generators import points
from pylab import *
from matplotlib import pyplot as plt
import scipy
import scipy.sparse
import sklearn
import pandas as pd
from Bio.PDB.MMCIFParser import MMCIFParser
import plotly
import plotly.express as px

def plot_persistance_diagram(points, min_range):
    alpha_complex = gd.AlphaComplex(points)
    st_alpha = alpha_complex.create_simplex_tree()
    Barcodes = st_alpha.persistence(min_persistence=min_range)
    gd.plot_persistence_diagram(Barcodes, legend=True)
    gd.plot_persistence_barcode(Barcodes, legend=True)


def num_k_simplices(simplicial_tree, k):
    if k > 0:
        n = len(list(simplicial_tree.get_skeleton(k))) - \
            len(list(simplicial_tree.get_skeleton(k-1)))
    else:
        n = simplicial_tree.num_vertices()
    return n


def sample(center, radius, n_per_sphere):
    r = radius
    ndim = center.size
    x = np.random.normal(size=(n_per_sphere, ndim))
    ssq = sqrt(np.sum(x**2, axis=1))
    p = (x.transpose()/ssq).transpose()*r+center
    return p


def sampleline(start, end, n_per_line):
    x = np.random.rand(n_per_line, 1)
    p = x*(end-start).transpose()+start
    return p


def get_simplices(simplicial_tree):
    maxdim = simplicial_tree.dimension()
    simplices = []
    for i in range(maxdim+1):
        simplices.append([])
    for simplextuple in simplicial_tree.get_simplices():
        simplex = simplextuple[0]
        simplices[len(simplex)-1].append(simplex)
    return simplices


def build_simplex_dict(simplicial_tree, simplices):
    maxdim = simplicial_tree.dimension()
    num_k_simplices_in_p = []
    simplexdict = []
    for i in range(maxdim+1):
        num = num_k_simplices(simplicial_tree, i)
        num_k_simplices_in_p.append(num)
        #print('Number of '+str(i)+'-simplices: '+str(num))
        simplexdict.append(dict(
            zip([str(simplex) for simplex in simplices[i]], range(num_k_simplices_in_p[i]))))
    return num_k_simplices_in_p, simplexdict


def extract_boundary_operators(simplices, simplexdict, num_k_simplices_in_p):
    maxdim = len(num_k_simplices_in_p)-1
    boundary_operators = []
    for k in range(maxdim):
        newmatrix = scipy.sparse.coo_matrix(
            (num_k_simplices_in_p[k], num_k_simplices_in_p[k+1]))
        coordi = []
        coordj = []
        entries = []
        for simplex in simplices[k+1]:
            simplex_index = simplexdict[k+1][str(simplex)]
            for i in range(k+2):
                new_simplex = simplex.copy()
                new_simplex.pop(i)
                new_simplex_index = simplexdict[k][str(new_simplex)]
                coordi.append(new_simplex_index)
                coordj.append(simplex_index)
                if i % 2 == 0:
                    entries.append(1)
                else:
                    entries.append(-1)
        boundary_operators.append(scipy.sparse.csc_matrix((np.array(entries), (np.array(coordi), np.array(
            coordj))), shape=(num_k_simplices_in_p[k], num_k_simplices_in_p[k+1]), dtype=float))
        #print("Shape of "+str(k)+"th Boundary operator: " +
       #       str(boundary_operators[k].shape))
    return boundary_operators


def extract_boundary_operators_new_idea(simplices, simplexdict, num_k_simplices_in_p, new_num_k_simplices_in_p,  old_simplex_dict):
    maxdim = len(num_k_simplices_in_p)-1
    boundary_operators = []
    for k in range(maxdim):
        newmatrix = scipy.sparse.coo_matrix(
            (num_k_simplices_in_p[k], num_k_simplices_in_p[k+1]))
        coordi = []
        coordj = []
        entries = []
        for simplex in simplices[k+1]:
            simplex_index = simplexdict[k+1][str(simplex)]
            for i in range(k+2):
                new_simplex = simplex.copy()
                new_simplex.pop(i)
                if str(new_simplex) in old_simplex_dict[k]:
                    new_simplex_index = old_simplex_dict[k][str(new_simplex)]
                    coordi.append(new_simplex_index)
                    coordj.append(simplex_index)
                    if i % 2 == 0:
                        entries.append(1)
                    else:
                        entries.append(-1)
        boundary_operators.append(scipy.sparse.csc_matrix((np.array(entries), (np.array(coordi), np.array(
            coordj))), shape=(num_k_simplices_in_p[k], new_num_k_simplices_in_p[k+1]), dtype=float))
        #print("Shape of "+str(k)+"th Boundary operator: " +
        #      str(boundary_operators[k].shape))
    return boundary_operators


def degree(boundary_operators, k):
    B = np.abs(boundary_operators[k])
    degrees = np.sum(B, axis=1)
    return degrees


def Adjacency_Matrix(boundary_operators, k):
    Bk = boundary_operators[k]
    A = -Bk@Bk.transpose()+scipy.sparse.diags(np.squeeze(np.asarray(degree(boundary_operators, k))))
    return A


def Hodge_Laplacian(boundary_operators, k):
    if k == len(boundary_operators):
        Bkm = boundary_operators[k-1]
        A = Bkm.transpose()@Bkm
    elif k > 0:
        Bk = boundary_operators[k]
        Bkm = boundary_operators[k-1]
        A = Bk@Bk.transpose()+Bkm.transpose()@Bkm
    else:
        Bk = boundary_operators[k]
        A = Bk@Bk.transpose()
    return A

def Hodge_Laplacian(boundary_operators, k):
    if k == len(boundary_operators):
        Bkm = boundary_operators[k-1]
        A = Bkm.transpose()@Bkm
    elif k > 0:
        Bk = boundary_operators[k]
        Bkm = boundary_operators[k-1]
        A = Bk@Bk.transpose()+Bkm.transpose()@Bkm
    else:
        Bk = boundary_operators[k]
        A = Bk@Bk.transpose()
    return A

def Weighted_1_Hodge_Laplacian(boundary_operators):
        Bk = boundary_operators[1]
        Bkm = boundary_operators[0]
        Weight_Edges = scipy.sparse.csc_array(np.maximum(np.diag(np.abs(Bk)@np.ones((Bk).shape[1])),np.diag(np.ones((Bk@Bk.transpose()).shape[0]))))
        Weight_Edges_Inverse = scipy.sparse.diags(1/(Weight_Edges@np.ones(Weight_Edges.shape[0])))
        Weight_Nodes = 2*scipy.sparse.diags(np.abs(Bkm)@Weight_Edges@np.ones(Weight_Edges.shape[0]))
        Weight_Nodes_Inverse = scipy.sparse.diags(1/(Weight_Nodes@np.ones(Weight_Nodes.shape[0])))
        Weight_Faces = scipy.sparse.diags(np.ones(Bk.shape[1])/3)
        A = Weight_Edges@Bkm.transpose()@Weight_Nodes_Inverse@Bkm+ Bk@Weight_Faces@Bk.transpose()@Weight_Edges_Inverse
        return A

def Hodge_Laplacian_new(boundary_operators, boundary_operators_new, k):
    if k == len(boundary_operators):
        Bkm = boundary_operators[k-1]
        A = Bkm.transpose()@Bkm
    elif k > 0:
        Bk = boundary_operators_new[k]
        Bkm = boundary_operators[k-1]
        A = Bk@Bk.transpose()+Bkm.transpose()@Bkm
    else:
        Bk = boundary_operators[k]
        A = Bk@Bk.transpose()
    return A


def sparsify(points, sparsity):
    psparse = list(points.copy())
    length = len(psparse)
    for i in range(int(floor(length*sparsity))):
        n = np.random.randint(0, len(psparse))
        psparse.pop(n)
    psparse = np.array(psparse)
    return psparse

def sparsify_two_lists(list1, list2, sparsity):
    list1sparse = list(list1.copy())
    list2sparse = list(list2.copy())
    length = len(list1sparse)
    for i in range(int(floor(length*sparsity))):
        n = np.random.randint(0, len(list1sparse))
        list1sparse.pop(n)
        list2sparse.pop(n)
    list1sparse = np.array(list1sparse)
    list2sparse = np.array(list2sparse)
    return list1sparse, list2sparse


def dist_mod_k(x, y, k):
    return np.absolute((x-y+k*0.5) % k-k*0.5)


def angle_to_colour(angle):
    value = angle/np.pi+1
    red = dist_mod_k(value, 0, 2)
    green = dist_mod_k(value, 2.0/3, 2)
    blue = dist_mod_k(value, 4.0/3, 2)
    return (red, green, blue)


def binary_sum(vector):
    sum = 0
    for i in range(len(vector)):
        sum += 2**i*vector[i]
    return sum


def sampleline(start, end, n_per_line):
    x = np.random.rand(n_per_line, 1)
    p = x*(end-start).transpose()+start
    return p


def randomcolour():
    return '#%06X' % np.random.randint(0, 0xFFFFFF)

def initialise_matlab_engine():
    eng = matlab.engine.start_matlab()
    clustering_path = eng.genpath('./MatLabClustering/DiSC')
    eng.addpath(clustering_path, nargout=0)
    clustering_path = eng.genpath('./MatLabClustering/prtools')
    eng.addpath(clustering_path, nargout=0)
    #clustering_path = eng.genpath('./rshlmnfnnbi')
    #eng.addpath(clustering_path, nargout=0)
    return eng


def subspace_clustering2(data_vectors, num_clusters, max_dim_subspace, eng, tuning_parameter=0.001):
    matlab_data_matrix = matlab.double(data_vectors.T.tolist())
    #print(matlab_data_matrix)
    matlab_clusters, I_ALL, ProjectionMatrix = eng.DiSC(
        matlab_data_matrix, float(max_dim_subspace), float(np.maximum(1, num_clusters)), 'quadrc', tuning_parameter, 50, nargout=3)
    results = np.array(matlab_clusters)-1
    return results


def subspace_clustering3(data_vectors, num_clusters, max_dim_subspace, eng, tuning_parameter=0.001):
    matlab_data_matrix = matlab.double(data_vectors.T.tolist())
    #print(matlab_data_matrix)
    [matlab_clusters, I_ALL, ProjectionMatrix] = eng.DiSC(
        matlab_data_matrix, float(max_dim_subspace), float(np.maximum(1, num_clusters)), 'quadrc', tuning_parameter, 50, nargout=3)
    results = np.array(matlab_clusters)-1
    return results


def subspace_clustering(num_clusters, data_vectors):
    matlab_data_matrix = matlab.double(data_vectors.T.tolist())
    #print(matlab_data_matrix)
    clustering = sklearn.cluster.SpectralClustering(num_clusters).fit(data_vectors)
    return clustering.labels_
