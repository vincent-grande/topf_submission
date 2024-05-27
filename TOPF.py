import numpy as np
import gudhi as gd
from pylab import *
from matplotlib import pyplot as plt
import matplotlib
import scipy
import scipy.sparse
from Bio.PDB.MMCIFParser import MMCIFParser
import sklearn
import pandas as pd
import plotly
import plotly.express as px
import seaborn as sns
import subprocess
import os
import sklearn.cluster
from sklearn.neighbors import KNeighborsClassifier
import csv
import ast
from plotly import graph_objects as go
#from Sampling import *
from TOPFbasics import *
from plotly.subplots import make_subplots


def make_weird_square():
    """
        Creates a square with a hole in it, with different densities in the different parts.
    """
    dens=50
    ul_sphere = thick_sphere(dens,0.,0.25)
    ur_sphere = thick_sphere(dens,0.,0.25)+[4,0]
    ll_sphere = thick_sphere(dens,0.,0.25)+[0,4]
    lr_sphere = thick_sphere(dens,0.,0.25)+[4,4]
    u_rect = filled_rectangle(dens/2, 3.5, 0.5)+[0.25,-0.25]
    d_rect = filled_rectangle(2*dens, 3.5, 0.5)+[0.25,3.75]
    l_rect = filled_rectangle(dens/4, 0.5, 3.5)+[-0.25,0.25]
    r_rect = filled_rectangle(dens, 0.5, 3.5)+[3.75,0.25]
    noise_rect = filled_rectangle(dens, 0.5, 3.5)+[1.75,4]
    ttotal = np.concatenate((ul_sphere, ur_sphere, ll_sphere, lr_sphere, u_rect, l_rect, r_rect, d_rect, noise_rect), axis=0)
    fig0 = plt.figure('3D Image of Points Used', figsize=(5, 5))
    ax0 = plt.axes()
    ax0.scatter(ttotal[:,0],ttotal[:,1])
    plt.show()
    return ttotal

def compute_curl_effective_resistance(Bkup):
    """
        Computes the curl effective resistance of a simplicial complex.
    """

def compute_weighted_harmonic_projection_upper_res(Bkm,Bk,vec,exponent = 1, mode = "weighted"):
    upper_res = compute_upper_effective_resistance(Bk)
    #print(tri_counts)
    weights = np.power(upper_res,exponent)
    if mode == "weighted":
        return compute_weighted_harmonic_projection(Bkm,Bk,vec,weights)
    else:
        return compute_harmonic_projection(Bkm,Bk,vec)/weights

def compute_upper_effective_resistance(Bk, exponent = 1):
    return np.sqrt(1-np.diag(scipy.linalg.pinv(Bk.T.toarray())@Bk.T)**(2*exponent))

def compute_weighted_harmonic_projection_lower_res(Bkm,Bk,vec,exponent = 1, mode = "weighted"):
    lower_res = compute_lower_effective_resistance(Bkm)
    weights = np.array(lower_res)**exponent
    if mode == "weighted":
        return compute_weighted_harmonic_projection(Bkm,Bk,vec,weights)
    else:
        return compute_harmonic_projection(Bkm,Bk,vec)/weights


def compute_lower_effective_resistance(Bkm, exponent = 1):
    return np.diag(scipy.linalg.pinv(Bkm.toarray())@Bkm)**exponent


def create_point_conversion(cur_points,original_points):
    """
        Creates a dictionary that converts points from an alpha complex ordering to the original ordering (?).
    """
    temp_string_dict = dict([[str(point),i] for i, point in enumerate(original_points)])
    point_indices = []
    for point in cur_points:
        point_indices.append(temp_string_dict[str(point)])
    return np.array(point_indices)

def topf_run(base_points, complex_type = 'alpha', thresh_julia = 0, max_hom_dim = 1.0,  max_rel_quot = 0.1, m = 0.3, weight_exponent = -1, simplex_threshs = (0.1,0.1), simplex_chances = (0.01,0.01), post_exp = 1, n_clusters = 4, draw_reps = False, draw_scaled_vecs = False, draw_final_clustering = False, draw_signatures = False, rep_chance = 1, damping = 0, eigenvector_tresholding = True, eigenvector_threshold = 0.07, clustering_method = 'spectral', clustering_sparseness = 0.0, exponential_interpolation = True, auto_num_clusters = True, max_total_quot = 0.1, quotient_life_times = False, use_eff_resistance = False, draw_signature_heatmaps = False, aggregation_mode = 'mean', scale_vecs = True, eff_resistance_mode = 'weighted', eff_resistance_exponent = 2, verbose = False, process_string = '', max_reps = 100,dim0min_pers_ratio=5, use_weight_tri_kernel = False,thresholding_type = 'linear', add_convex_hull = False, convex_hull_density_coeff =0.1, only_dims = [0,1,2,3], heatmaps_in_one = False):
    """
        One Run of the entire TOPF algorithm.
    """
    base_points = noisify_input_points(np.array(base_points)) #To prevent numerical instabilities with alpha filtration
    num_old_base_points = len(base_points)
    if add_convex_hull:
        base_points = np.concatenate((base_points, construct_convex_hull(base_points, density_coeff = convex_hull_density_coeff)), axis = 0)
    np.savetxt("JuliaCommunication/PointsForJulia" + process_string+ ".csv", base_points, delimiter=",")
    if verbose:
        print("Points saved. Starting Ripserer...")
    if process_string == '':
        os.system("julia --project=TestJuliaEnvironment HomologyGeneratorsMultiD.jl "+str(thresh_julia)+" "+str(max_hom_dim)+" "+complex_type+" "+process_string)
    else:
        os.system("julia --project=TestJuliaEnvironment HomologyGeneratorsMultiDMultiP.jl "+str(thresh_julia)+" "+str(max_hom_dim)+" "+complex_type+" "+process_string)
    if verbose:
        print("Ripserer finished. Reading Reps...")
    multi_reps, multi_inds, multi_life_times = extract_reps(complex_type, max_hom_dim, thresh_julia, max_rel_quot, damping=damping, max_total_quot = max_total_quot, quotient_life_times= quotient_life_times, process_string = process_string, verbose = verbose, max_reps = max_reps, dim0min_pers_ratio = dim0min_pers_ratio, only_dims = only_dims)
    if verbose:
        print(multi_inds)
    if auto_num_clusters:
        n_clusters = np.sum([len(cur_indices) for cur_indices in multi_inds])
    if verbose:
        print("Reading done. Computing Projections...")
    multi_eigen_vecs = [[],[],[]]
    multi_scaled_vecs = [[],[],[]]
    multi_multi_simplices =[[],[],[]]
    multi_multi_points = [[],[],[]]
    multi_scaled_vecs_save = []
    multi_all_num_k_simplices_in_p = [[],[],[]]
    for cur_d in range(len(multi_inds)):
        if verbose:
            print("Computing Projection for Dimension "+str(cur_d))
        eigen_vecs, scaled_vecs, all_num_k_simplices_in_p, multi_points, multi_simplices = compute_projection_from_reps(base_points, cur_d,multi_inds, multi_reps, multi_life_times, m = m, weight_exponent = weight_exponent, complex_type = complex_type, exponential_interpolation = exponential_interpolation, use_eff_resistance = use_eff_resistance, scale_vecs = scale_vecs, eff_resistance_exponent = eff_resistance_exponent, eff_resistance_mode = eff_resistance_mode, weight_tri_kernel = use_weight_tri_kernel)
        multi_eigen_vecs[cur_d] = eigen_vecs
        multi_scaled_vecs_save.append([np.copy(array) for array in scaled_vecs])
        if eigenvector_tresholding:
            for i,vec in enumerate(scaled_vecs):
                vec = vec
                scaled_vecs[i] = np.array([my_threshold(value = v,threshold=eigenvector_threshold, thresholding_type=thresholding_type) for v in vec])
        multi_scaled_vecs[cur_d] = [np.copy(array) for array in scaled_vecs]
        multi_all_num_k_simplices_in_p[cur_d] = all_num_k_simplices_in_p
        multi_multi_points[cur_d] = multi_points
        multi_multi_simplices[cur_d] = multi_simplices
    if verbose:
        print("Computation done. Plotting representatives...")
    if draw_reps:
        draw_representatives(base_points, max_hom_dim, multi_multi_points, multi_reps, multi_inds, chance = rep_chance)
        if verbose:
            print("Plotting representatives done. Plotting eigenvector components...")
    if draw_scaled_vecs:
        if verbose:
            print(multi_scaled_vecs_save)
        plot_vecs(base_points, multi_scaled_vecs_save, multi_all_num_k_simplices_in_p, multi_multi_points, multi_multi_simplices, multi_inds, max_hom_dim, simplex_threshs, simplex_chances)
    if verbose:
        print("Plotting eigenvector components done. Generating long point signatures...")
    long_point_signatures = generate_long_point_signatures(base_points, multi_inds, multi_multi_simplices, multi_scaled_vecs)
    if verbose:
        print("Long point signatures done. Generating short signatures...")
    short_flat_signatures = generate_short_point_signatures(long_point_signatures, aggregation_mode)
    if verbose:
        print("Short signatures done. Clustering...")
    short_flat_signatures = short_flat_signatures[:num_old_base_points]
    base_points = base_points[:num_old_base_points]
    labels = cluster_points(short_flat_signatures, base_points, n_clusters = n_clusters, sparseness = clustering_sparseness, post_exp = post_exp, show_plots = draw_final_clustering, clustering_method=clustering_method, verbose=verbose)
    if draw_signatures:
        draw_short_signatures(short_flat_signatures, labels)
    if draw_signature_heatmaps:
        if len(short_flat_signatures)>3000:
            base_points, short_flat_signatures_s = sparsify_two_lists(base_points, short_flat_signatures, 1-3000/len(short_flat_signatures))
        else:
            short_flat_signatures_s = short_flat_signatures
        if heatmaps_in_one:
            plot_signatures_in_one(base_points,short_flat_signatures_s)
        else:
            plot_signatures(base_points,short_flat_signatures_s)
    return labels, short_flat_signatures

def noisify_input_points(base_points, noise_level= 0.00001):
    """
        Adds noise to the points.
    """
    return base_points + noise_level * np.random.normal(size=base_points.shape) * np.std(base_points)

def my_colour_maps_strings():
    """
    Returns list of colour maps.
    """
    return ['Blues', 'Reds',  'Greens', 'Oranges', 'Purples', 'Greys', 'Greys', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', 'viridis', 'inferno', 'plasma', 'magma', 'cividis', 'twilight', 'twilight_shifted', 'gist_earth', 'terrain', 'ocean', 'gist_stern', 'gist_heat', 'gist_rainbow', 'gist_ncar']

def plot_signatures_in_one(base_points, signatures):
    points = base_points
    num_examples = signatures.shape[1]
    ambient_dim = points.shape[1]
    colour_maps_strings = my_colour_maps_strings()
    if ambient_dim>=3:
        axes_signatures = make_subplots(cols=1, rows=1, horizontal_spacing=0.0, vertical_spacing=0.0, specs = [[{'type': 'scene'}]])
    else:
        axes_signatures = make_subplots(cols=1, rows=1, horizontal_spacing=0.0, vertical_spacing=0.0)
    axes_signatures.update_layout(width = 1000, height = 1000)
    axes_signatures.update_layout(showlegend=False)
    axes_signatures.update_xaxes(visible=False)
    axes_signatures.update_yaxes(visible=False)
    largest_feature = [np.argmax(features_point) for features_point in signatures]
    relevant_indices = [[i for i in range(points.shape[0]) if largest_feature[i] == j] for j in range(num_examples)]
    for i in range(num_examples):
        cur_points = points[relevant_indices[i]]
        cur_signatures = signatures[relevant_indices[i]]
        if ambient_dim ==2:
            widths = 20*cur_signatures[:,i]+1
            opc = 1
            axes_signatures.add_scatter(x=cur_points[:,0], y=cur_points[:,1],row = 1, col =1, mode = 'markers',marker = dict( 
                                        size = widths, 
                                        color = cur_signatures[:,i], 
                                        colorscale =colour_maps_strings[i], 
                                        opacity = opc
                                    ) )
        else:
            widths = 12*cur_signatures[:,i]+4
            opc = 1
            axes_signatures.add_scatter3d(x=cur_points[:,0], y=cur_points[:,1], z = cur_points[:,2],row = 1, col =1, mode = 'markers',marker = dict( 
                                        size = widths, 
                                        color = cur_signatures[:,i], 
                                        colorscale =colour_maps_strings[i], 
                                        opacity = opc
                                    ) )
    axes_signatures.update_scenes(xaxis_visible=False, yaxis_visible=False)
    if ambient_dim>2:
        axes_signatures.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )
    axes_signatures.update_layout(showlegend=False)
    axes_signatures.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
    axes_signatures.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    #axes_signatures.write_image("SignatureHeatmaps.pdf")
    axes_signatures.show()

def plot_signatures(base_points, signatures):
    points = base_points
    num_examples = signatures.shape[1]
    ambient_dim = points.shape[1]
    colour_maps_strings = my_colour_maps_strings()
    if ambient_dim>=3:
        axes_signatures = make_subplots(cols=(num_examples+1)//2, rows=2, horizontal_spacing=0.0, vertical_spacing=0.0, specs = [[{'type': 'scene'}]*((num_examples+1)//2),[{'type': 'scene'}]*((num_examples+1)//2)])
    else:
        axes_signatures = make_subplots(cols=(num_examples+1)//2, rows=2, horizontal_spacing=0.0, vertical_spacing=0.0)
    axes_signatures.update_layout(width = 2000/13*num_examples, height = 600)
    axes_signatures.update_layout(showlegend=False)
    axes_signatures.update_xaxes(visible=False)
    axes_signatures.update_yaxes(visible=False)
    for i in range(num_examples):
        if ambient_dim >=3:
            axes_signatures.add_scatter3d(x=points[:,0], y=points[:,1], z = points[:,2],row = (i)%2+1, col =(i)//2+1, mode = 'markers',marker = dict( 
                                        size = 2, 
                                        color = signatures[:,i], 
                                        colorscale ='Viridis', 
                                        opacity = 0.8
                                    ) )
            axes_signatures.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )
        else:
            widths = 4*signatures[:,i]+2
            axes_signatures.add_scatter(x=points[:,0], y=points[:,1], row = (i)%2+1, col =(i)//2+1, mode = 'markers',marker = dict( 
                                        size = 2, 
                                        color = 'black', 
                                        opacity = 0.8
                                    ) )
            axes_signatures.add_scatter(x=points[:,0], y=points[:,1], row = (i)%2+1, col =(i)//2+1, mode = 'markers',marker = dict( 
                                        size = widths, 
                                        color = signatures[:,i], 
                                        colorscale =colour_maps_strings[i], 
                                        opacity = 0.8
                                    ) )
            axes_signatures.update_scenes(xaxis_visible=False, yaxis_visible=False)
            #axes.add_scatter(x=points[:,0], y=points[:,1], row = 2, col =i+1, mode = 'markers')
    axes_signatures.update_layout(showlegend=False)
    axes_signatures.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
    axes_signatures.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    #axes_signatures.write_image("SignatureHeatmaps.pdf")
    axes_signatures.show()


def sample_points_simplex(simplex, density, noise = 0.02):
    area_simplex = np.linalg.norm(np.cross(simplex[1]-simplex[0],simplex[2]-simplex[0]))/2
    num_samples = int(np.floor(density*area_simplex))
    samples = []
    for i in range(num_samples*5):
        r1 = np.random.rand()
        r2 = np.random.rand()
        if r1+r2>1:
            r1 = 1-r1
            r2 = 1-r2
        samples.append(simplex[0]+r1*(simplex[1]-simplex[0])+r2*(simplex[2]-simplex[0]))
    samples = np.array(samples)
    if num_samples>0:
        samples, boring = minmax_landmark_sampling(samples, num_samples)
    return noisify(samples, noise*np.sqrt(area_simplex))
    
def noisify (points, noise_level):
    return points + noise_level * np.random.normal(size=points.shape)

def construct_convex_hull(base_points, density_coeff, noise = 0.01):
    hull = scipy.spatial.ConvexHull(base_points)
    hull_points_list = []
    if base_points.shape[1]>3:
        return np.array([]).reshape(0,base_points.shape[0])
    if base_points.shape[1]==3:
        density = 0.2*(base_points.shape[0]/np.prod(np.std(base_points, axis=0)))**(2/3)
        for simplex in hull.simplices:
            new_points = sample_points_simplex(base_points[simplex], density, noise = noise)
            if new_points.shape[0]>0:
                hull_points_list.append(new_points)
        hull_points = np.concatenate(hull_points_list, axis=0)
    elif base_points.shape[1] == 2:
        density = 0.2*(base_points.shape[0]/np.prod(np.std(base_points, axis=0)))
        for simplex in hull.simplices:
            length = np.linalg.norm(base_points[simplex[1]]-base_points[simplex[0]])
            num_points = np.floor(density*length)    
            new_points = np.array(np.linspace(base_points[simplex[0]], base_points[simplex[1]], num_points+1))[1:-1]
            if new_points.shape[0]>0:
                hull_points_list.append(noisify(new_points, noise*length))
        hull_points = np.concatenate(hull_points_list, axis=0)
    return hull_points

def plot_clustered_points(base_points, labels):
    """
        Plots the points with the labels.
    """
    ambient_dim = base_points.shape[1]
    if ambient_dim>=3:
        fig = px.scatter_3d(x=base_points[:,0], y=base_points[:,1], z=base_points[:,2], opacity=0.5, color=labels,width=1000, height=1000)
    else:
        fig = px.scatter(x = base_points[:,0], y = base_points [:,1], color=[str(label) for label in labels], width=500, height=500)
        fig.update_scenes(xaxis_visible=False, yaxis_visible=False)
        fig.update_layout(showlegend=False)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig.update_traces(marker=dict(size=5))
        fig.show()
    fig.update_traces(marker=dict(size=5))
    fig.show()
    return labels

def cluster_points(short_flat_signatures, base_points, n_clusters,sparseness = 0.0, post_exp = 1, show_plots = True, clustering_method = 'kmeans', verbose = False):
    """
        Clusters the points using the short signatures.
    """
    sparsity = sparseness
    if len(short_flat_signatures)>5000:
        sparsity = 1-5000/len(short_flat_signatures)
        if verbose:
            print("Sparsifying to "+str(sparsity))
    base_points_s, sparse_signatures = sparsify_two_lists(base_points,short_flat_signatures,sparsity)
    if clustering_method == 'kmeans':
        clustering = sklearn.cluster.KMeans(n_clusters, n_init = 'auto').fit(sparse_signatures**post_exp)
        if verbose:
            print("KMeans Clustering")
    elif clustering_method == 'spectral':
        clustering = sklearn.cluster.SpectralClustering(n_clusters).fit(sparse_signatures**post_exp)
        if verbose:
            print("Spectral Clustering")
    elif clustering_method == 'agglomerative':
        clustering = sklearn.cluster.AgglomerativeClustering(n_clusters).fit(sparse_signatures**post_exp)
        if verbose:
            print("Agglomerative Clustering")
    labels_sparse = clustering.labels_
    if sparsity>0:
        neighbour_classifier = KNeighborsClassifier(n_neighbors=1)
        neighbour_classifier.fit(sparse_signatures, labels_sparse)
        labels = neighbour_classifier.predict(short_flat_signatures)
    else:
        labels = labels_sparse
    ambient_dim = base_points_s.shape[1]
    if verbose:
        print("shape base_points:"+str(base_points_s.shape))
    string_labels = np.array([str(label) for label in labels_sparse])
    if verbose:
        print("string_label length"+str(string_labels.shape))
    if show_plots:
        if verbose:
            print("string_label length"+str(len(string_labels)))
            print("shape base_points:"+str(base_points_s.shape))
        if ambient_dim>=3:
            fig_final = px.scatter_3d(x=base_points_s[:,0], y=base_points_s[:,1], z=base_points_s[:,2], opacity=0.5, color=string_labels,width=1000, height=1000)
            fig_final.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )
        else:
            fig_final = px.scatter(x = base_points_s[:,0], y = base_points_s [:,1], color=string_labels, width=500, height=500)
            fig_final.update_scenes(xaxis_visible=False, yaxis_visible=False)
            fig_final.update_layout(showlegend=False)
            fig_final.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig_final.update_traces(marker=dict(size=5))
        fig_final.show()
    return labels

def draw_short_signatures(short_flat_signatures, labels, post_exp = 1):
    """
        Plots the short signatures.
    """
    ambient_dim = short_flat_signatures.shape[1]
    if ambient_dim>=3:
        if ambient_dim > 3:
            fig13 = px.scatter_3d(x=short_flat_signatures[:,0]**post_exp, 
                y=short_flat_signatures[:,1]**post_exp,
                z=short_flat_signatures[:,2]**post_exp,
                opacity=0.9, color=[str(label) for label in labels],width=1000, height=1000)
            fig13.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
            #fig13.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False)
        else:
            fig13 = px.scatter_3d(x=short_flat_signatures[:,0]**post_exp, 
                y=short_flat_signatures[:,1]**post_exp,
                z=short_flat_signatures[:,2]**post_exp,
                opacity=0.5, color=labels,width=1000, height=1000)
    elif ambient_dim == 2:
        fig13 = px.scatter(x = short_flat_signatures[:,0]**post_exp, y = short_flat_signatures [:,1]**post_exp, color=labels)
    if ambient_dim == 1:
        fig13 = px.scatter(x = short_flat_signatures[:,0]**post_exp,
                            y = short_flat_signatures [:,0]**post_exp, color=labels)
    if ambient_dim > 0:
        fig13.update_traces(marker=dict(size=10))
        fig13.show()

def generate_short_point_signatures(long_point_signatures, aggregation_mode = 'mean'):
    """
        Shortens the long point signatures.
    """
    short_signatures =[]
    short_flat_signatures = []
    for single_point_signature in long_point_signatures:
        short_point_signature =[]
        short_flat_point_signature = []
        for dim_signature in single_point_signature:
            short_dim_signature = []
            for eigenvec_signature in dim_signature:
                #print(eigenvec_signature)
                if aggregation_mode == 'mean':
                    if len(eigenvec_signature) == 0:
                        new_entry = 0
                    else:
                        new_entry = np.mean(np.array(eigenvec_signature))
                else:
                    new_entry = np.max(np.array(eigenvec_signature+[0]))
                short_dim_signature.append(new_entry)
                short_flat_point_signature.append(new_entry)
            short_point_signature.append(short_dim_signature)
        short_signatures.append(short_point_signature)
        short_flat_signatures.append(short_flat_point_signature)
    return np.array(short_flat_signatures)

def generate_long_point_signatures(base_points, multi_inds, multi_multi_simplices, multi_scaled_vecs):
    """
        Generates the long point signatures from the eigenvector components on the higher-order simplices.
    """
    long_point_signatures = []
    for i in range(len(base_points)):
        cur_signature =[]
        for inds in multi_inds:
            cur_temp_signature = []
            for ind in inds:
                cur_temp_signature.append([])
            cur_signature.append(cur_temp_signature)
        long_point_signatures.append(cur_signature)
    for d,inds in enumerate(multi_inds):
        for i in range(len(inds)):
            for h,simplex in enumerate(multi_multi_simplices[d][i][d]):
                for vertex in simplex:
                    long_point_signatures[vertex][d][i].append(multi_scaled_vecs[d][i][h])
    return long_point_signatures

def plot_vecs(base_points, multi_scaled_vecs, multi_all_num_k_simplices_in_p, multi_multi_points, multi_all_simplices, multi_inds, max_hom_dim, threshold  = (0.1,0.1), chance = (0.01,0.01)):
    """
        Plots the eigenvector components of the harmonic projection on the simplices.
    """
    ambient_dim = base_points.shape[1]
    #print(ambient_dim)
    colour_maps = my_colour_maps()
    if ambient_dim >=3:
        figsimplices = px.scatter_3d(x=base_points[:,0], y=base_points[:,1], z=base_points[:,2])
    else:
        figsimplices = px.scatter(x=base_points[:,0], y=base_points[:,1], width = 600, height = 600)
    figsimplices.update_traces(marker=dict(size=0.5))
    figsimplices.update_xaxes(visible=False)
    figsimplices.update_yaxes(visible=False)
    figsimplices.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
    figsimplices.update_scenes(xaxis_visible=False, yaxis_visible=False)
    for i in range(len(multi_inds[0])):
        for index in range(multi_all_num_k_simplices_in_p[0][i][0]):
            if (multi_scaled_vecs[0][i][index]> 0):
                simplex = multi_all_simplices[0][i][0][index]
                starting_point = simplex[0]
                starting_point_coord = multi_multi_points[0][i][starting_point]
                cur_colour = matplotlib.colors.to_hex(matplotlib.cm.tab10(i))
                xline = [starting_point_coord[0]]
                yline = [starting_point_coord[1]]
                if ambient_dim>=3:
                    zline = [starting_point_coord[2]]
                    figsimplices.add_trace(go.Scatter3d(x=xline, y=yline, z=zline, mode='markers', marker=dict(color=cur_colour, size = 10)))
                else:
                    figsimplices.add_trace(go.Scatter3d(x=xline, y=yline,mode='markers', marker=dict(color=cur_colour, size = 10)))
    for i in range(len(multi_inds[1])):
        multi_scaled_vecs[1][i]=np.abs(multi_scaled_vecs[1][i])/np.max(np.abs(multi_scaled_vecs[1][i]))
        for index in range(multi_all_num_k_simplices_in_p[1][i][1]):
            if (multi_scaled_vecs[1][i][index]> threshold[0] and np.random.rand()<chance[0]):
                simplex = multi_all_simplices[1][i][1][index]
                starting_point = simplex[0]
                end_point = simplex[1]
                starting_point_coord = multi_multi_points[1][i][starting_point]
                end_point_coord = multi_multi_points[1][i][end_point]
                cur_colour = matplotlib.colors.to_hex(colour_maps[i](multi_scaled_vecs[1][i][index]**0.4))
                xline = [starting_point_coord[0], end_point_coord[0]]
                yline = [starting_point_coord[1], end_point_coord[1]]
                if ambient_dim>=3:
                    zline = [starting_point_coord[2], end_point_coord[2]]
                    figsimplices.add_trace(go.Scatter3d(x=xline, y=yline, z=zline, mode='lines', line=dict(color=cur_colour, width=2+multi_scaled_vecs[1][i][index]*10)))
                else:
                    figsimplices.add_trace(go.Scatter(x=xline, y=yline, mode='lines', line=dict(color=cur_colour, width=2+multi_scaled_vecs[1][i][index]*6)))
    if max_hom_dim == 2:
        for i in range(len(multi_inds[2])):
            for index in range(multi_all_num_k_simplices_in_p[2][i][2]):
                if (multi_scaled_vecs[2][i][index]>threshold[1] and np.random.rand()<chance[1]):
                    largesimplex = multi_all_simplices[2][i][2][index]
                    for simplex in [largesimplex[:-1],largesimplex[1:], [largesimplex[0],largesimplex[2]]]:
                        starting_point = simplex[0]
                        end_point = simplex[1]
                        starting_point_coord = multi_multi_points[2][i][starting_point]
                        end_point_coord = multi_multi_points[2][i][end_point]
                        cur_colour = matplotlib.colors.to_hex(colour_maps[i+len(multi_inds[2])](multi_scaled_vecs[2][i][index]**0.4))
                        xline = [starting_point_coord[0], end_point_coord[0]]
                        yline = [starting_point_coord[1], end_point_coord[1]]
                        if ambient_dim>=3:
                            zline = [starting_point_coord[2], end_point_coord[2]]
                            figsimplices.add_trace(go.Scatter3d(x=xline, y=yline, z=zline, mode='lines', line=dict(color=cur_colour, width=2)))
                        else:
                            figsimplices.add_trace(go.Scatter(x=xline, y=yline, mode='lines', line=dict(color=cur_colour, width=2)))
    figsimplices.update_xaxes(visible=False)
    figsimplices.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    figsimplices.update_yaxes(visible=False)
    figsimplices.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
    figsimplices.update_scenes(xaxis_visible=False, yaxis_visible=False)
    figsimplices.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )
    figsimplices.show()

def nonsense():
    print("hi")

def draw_representatives(points, max_hom_dim, multi_multi_points, multi_representations, multi_inds, chance):
    """
        Plots homology representatives
    """
    ambient_dim = points.shape[1]
    if ambient_dim >=3:
        figcolour = px.scatter_3d(x=points[:,0], y=points[:,1], z=points[:,2])
    else:
        figcolour = px.scatter(x=points[:,0], y=points[:,1], width = 600, height= 600)
    figcolour.update_traces(marker=dict(size=1, color = 'black', opacity= 0.5))
    while len(multi_inds)<3:
        multi_inds.append([])
    for i in range(len(multi_inds[0])):
        cur_points = points
        xline = []
        yline = []
        zline = []
        for j,rep in enumerate(multi_representations[0][i]):
            simplex = rep[0]
            starting_point = simplex[0]
            starting_point_coord = cur_points[starting_point]
            cur_colour = matplotlib.colors.to_hex(my_colour_maps()[i+len(multi_inds[1])+len(multi_inds[2])](0.5))
            xline.append(starting_point_coord[0])
            yline.append(starting_point_coord[1])
            if ambient_dim>=3:
                zline.append(starting_point_coord[2])
        if ambient_dim>=3:
            figcolour.add_trace(go.Scatter3d(x=xline, y=yline, z=zline, mode='markers', marker=dict(color=cur_colour, size = 10)))
        else:
            figcolour.add_trace(go.Scatter(x=xline, y=yline,mode='markers', marker=dict(color=cur_colour, size = 50)))
    for i in range(len(multi_inds[1])):
        cur_points = multi_multi_points[1][i]
        for j,rep in enumerate(multi_representations[1][i]):
            simplex = rep[0]
            starting_point = simplex[0]
            end_point = simplex[1]
            starting_point_coord = cur_points[starting_point]
            end_point_coord = cur_points[end_point]
            xline = [starting_point_coord[0], end_point_coord[0]]
            yline = [starting_point_coord[1], end_point_coord[1]]
            if ambient_dim >= 3:
                zline = [starting_point_coord[2], end_point_coord[2]]
            cur_colour = matplotlib.colors.to_hex(my_colour_maps()[i](0.5))
            if ambient_dim>=3:
                figcolour.add_trace(go.Scatter3d(x=xline, y=yline, z=zline, mode='lines', line=dict(color=cur_colour, width=8)))
            else:
                figcolour.add_trace(go.Scatter(x=xline, y=yline, mode='lines', line=dict(color=cur_colour, width=10)))
    if max_hom_dim >= 2:
        print("Printing 2D representatives")
        for i,index in enumerate(multi_inds[2]):
            cur_points = multi_multi_points[2][i]
            for j,rep in enumerate(multi_representations[2][i]):
                largesimplex = rep[0]
                if np.random.rand()<chance:
                    for simplex in [largesimplex[:-1],largesimplex[1:], [largesimplex[0],largesimplex[2]]]:
                        starting_point = simplex[0]
                        end_point = simplex[1]
                        starting_point_coord = cur_points[starting_point]
                        end_point_coord = cur_points[end_point]
                        xline = [starting_point_coord[0], end_point_coord[0]]
                        yline = [starting_point_coord[1], end_point_coord[1]]
                        if ambient_dim >= 3:
                            zline = [starting_point_coord[2], end_point_coord[2]]
                        cur_colour = matplotlib.colors.to_hex(matplotlib.cm.tab10(i+len(multi_inds[1])))
                        if ambient_dim>=3:
                            figcolour.add_trace(go.Scatter3d(x=xline, y=yline, z=zline, mode='lines', line=dict(color=cur_colour, width=10)))
                        else:
                            figcolour.add_trace(go.Scatter(x=xline, y=yline, mode='lines', line=dict(color=cur_colour, width=5)))
                #figcolour.add_trace(go.Scatter(x=xline, y=yline,mode='markers', marker=dict(color=cur_colour, size = 50)))
    figcolour.update_xaxes(visible=False)
    figcolour.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    figcolour.update_yaxes(visible=False)
    figcolour.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
    figcolour.update_scenes(xaxis_visible=False, yaxis_visible=False)
    figcolour.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )
    #figcolour.write_image("Representatives.pdf")
    figcolour.show()

def compute_projection_from_reps(base_points, cur_d, multi_inds, multi_reps, multi_life_times, m, weight_exponent, complex_type, exponential_interpolation, use_eff_resistance, scale_vecs, eff_resistance_exponent, eff_resistance_mode, verbose = False, weight_tri_kernel = False):
    """
        Computes projection of homology representatives to harmonic space.
    """
    eigen_vecs = []
    scaled_vecs = []
    all_num_k_simplices_in_p = []
    multi_points = []
    multi_simplices = []
    for i in range(len(multi_inds[cur_d])):
        cur_index = multi_inds[cur_d][i]
        if exponential_interpolation and multi_life_times[cur_d][cur_index][0]*(1-m)>0 and cur_d >0:
            max_edge_length = multi_life_times[cur_d][cur_index][1]**m*multi_life_times[cur_d][cur_index][0]**(1-m)
        else:
            if verbose:
                print("cur_d: "+str(cur_d))
                print("cur m: "+str(m))
                print("birth:"+str(multi_life_times[cur_d][cur_index][0])+ " death: "+str(multi_life_times[cur_d][cur_index][1]))
            if cur_d == 0:
                max_edge_length = multi_life_times[cur_d][cur_index][1]*0.995
                if verbose:
                    print("Max edge length in dim 0: "+str(max_edge_length))
            else:
                max_edge_length = multi_life_times[cur_d][cur_index][0]*(1-m)+multi_life_times[cur_d][cur_index][1]*m
                if verbose:
                    print("actual edge length:"+str(max_edge_length))
        boundary_operators,simplices, simplexdict, num_k_simplices_in_p, cur_points = construct_simplices(base_points, length = max_edge_length, complex_type = complex_type, maxdim=cur_d+1)
        eigen_vecs = []
        if verbose:
            print(cur_d)
        if verbose:
            print(num_k_simplices_in_p)
        char_vector = np.zeros(num_k_simplices_in_p[cur_d])
        if verbose:
            print("We are at cur_d = "+str(cur_d))
            print("We are at i = "+str(i))
            print("Total multi_reps_length: "+str(len(multi_reps)))
            print("Current multi_reps length: "+str(len(multi_reps[cur_d])))
        for rep in multi_reps[cur_d][i]:
            simplex = rep[0]
            value = rep[1]
            simplex_key = str((list(flip(simplex))))
            if simplex_key in simplexdict[cur_d]:
                char_vector[simplexdict[cur_d]
                            [simplex_key]] = (value+1%3)-1
        if cur_d == 0:
            if verbose:
                print(np.sum(char_vector))
            #plot_clustered_points(base_points, char_vector)
            vec = char_vector#compute_harmonic_projection_zero(
               # Bk=boundary_operators[cur_d], vec = char_vector)
        else:
            if verbose:
                print(boundary_operators)
            if use_eff_resistance:
                vec = compute_weighted_harmonic_projection_lower_res(
                    Bkm = boundary_operators[cur_d-1], Bk=boundary_operators[cur_d], vec = char_vector, exponent=eff_resistance_exponent, mode = eff_resistance_mode)
            elif weight_tri_kernel:
                weights_tri_kernel = [0 for i in range(3)]
                weights_tri_kernel[2] = compute_upper_weights(simplices = simplices[2], base_points = cur_points, threshold = max_edge_length)
                weights_tri_kernel[1] = compute_lower_weights(weights_tri_kernel[2], Bk = boundary_operators[1])
                weights_tri_kernel[0] = compute_lower_weights(weights_tri_kernel[1], Bk = boundary_operators[0])
                vec = compute_weighted_harmonic_projection(
                    Bkm = boundary_operators[cur_d-1], Bk=boundary_operators[cur_d], vec = char_vector,weights=1/(weights_tri_kernel[cur_d]+1e-13))
            else:
                if len(boundary_operators) < cur_d+2:
                    boundary_operators.append(np.zeros((num_k_simplices_in_p[cur_d],1)))
                    if verbose:
                        print("We are at cur_d = "+str(cur_d))
                        print("length boundary operator:" +str(len(boundary_operators)))
                vec = compute_weighted_harmonic_projection_num_tris(
                    Bkm = boundary_operators[cur_d-1], Bk=boundary_operators[cur_d], vec = char_vector, exponent = weight_exponent)
        eigen_vecs.append(vec)
        if scale_vecs:
            scaled_vecs.append(np.abs(vec)/np.max(np.abs(vec)+1e-13))
        else:
            scaled_vecs.append(np.abs(vec))
        all_num_k_simplices_in_p.append(num_k_simplices_in_p)
        multi_points.append(cur_points)
        multi_simplices.append(simplices)
    return eigen_vecs, scaled_vecs, all_num_k_simplices_in_p, multi_points, multi_simplices

def construct_simplices(base_points, length, complex_type = 'alpha', maxdim = 2):
    """
    Constructs the simplices of a simplicial complex from a set of points and a maximum edge length. Returns Boundary operators and useful information on simplices.
    """
    max_edge_length_barcodes = length
    if complex_type == 'alpha':
        test_komplex = gd.AlphaComplex(
            points = base_points, precision = 'exact')
        rips_simplex_tree_sample = test_komplex.create_simplex_tree(
            max_alpha_square=max_edge_length_barcodes**2)
    elif complex_type == 'rips':
        test_komplex = gd.RipsComplex(
            points = base_points, max_edge_length=max_edge_length_barcodes)
        rips_simplex_tree_sample = test_komplex.create_simplex_tree(
            max_dimension =maxdim)
    simplicial_tree = rips_simplex_tree_sample
    simplices = get_simplices(simplicial_tree)
    num_k_simplices_in_p, simplexdict = build_simplex_dict(
        simplicial_tree=simplicial_tree, simplices=simplices)
    boundary_operators = extract_boundary_operators(
        simplices, simplexdict, num_k_simplices_in_p)
    if complex_type == 'alpha':
        cur_points = np.array([test_komplex.get_point(i) for i in range(len(base_points))])
    else:
        cur_points = base_points
    return boundary_operators,simplices, simplexdict, num_k_simplices_in_p, cur_points


def extract_reps(complex_type, max_hom_dim, thresh_julia = 0, max_rel_quot = 0.1, damping = 0, max_total_quot = 0.1, quotient_life_times = False, verbose = False, process_string = '', max_reps = 100, dim0min_pers_ratio = 2, only_dims= [0,1,2,3]):
    """
    Extracts the representatives from the Julia output files and returns the most persistent representatives based on some threshold.
    """
    multi_representations = []
    multi_life_times = []
    multi_persistence = []
    multi_quot_persistence = []
    # if complex_type == 'rips':
    #     exp = 1
    # else:
    #     exp = 0.5
    #     if verbose:
    #         print("0.5exponent")
    exp = 1
    csv.field_size_limit(sys.maxsize)
    for dim in range(int(max_hom_dim)+1):    
        cur_reps, cur_life_times, cur_persistence = read_julia_reps_1_dim(dim,exp, process_string = process_string, verbose = verbose)
        multi_representations.append(cur_reps)
        if complex_type == 'alpha':
            cur_life_times = cur_life_times/2
            if verbose:
                print("Alpha complex life times halved")
        multi_life_times.append(cur_life_times)
        if verbose:
            print("Life times in dim "+str(dim)+":"+str(cur_life_times))
        #for life_time in cur_life_times:
        #    print(life_time)
        multi_persistence.append(cur_persistence)
        if dim > 0:
            multi_quot_persistence.append(cur_life_times[:,1]/cur_life_times[:,0])
        else:
            multi_quot_persistence.append(cur_persistence)
    multi_inds = []
    min_persistence = np.inf
    max_persistence = 0
    for d, persistence in enumerate(multi_persistence):
        quot_persistence = multi_quot_persistence[d]
        if d == 0:
            multi_inds.append([])
        else:
            if thresh_julia > 0:
                persistence = np.minimum(multi_persistence[2],2*thresh_julia)
            multi_inds.append(pick_features(persistence, quot_persistence, max_rel_quot = max_rel_quot, damping = damping, max_total_quot = max_total_quot, quotient_life_times = quotient_life_times))
            multi_inds[d] = multi_inds[d][:min(len(multi_inds[d]),max_reps)]
            min_persistence = np.minimum(min_persistence,min((multi_life_times[d][multi_inds[d]][:,1]-multi_life_times[d][multi_inds[d]][:,0]))* dim0min_pers_ratio)
            max_persistence = np.maximum(max_persistence,max((multi_life_times[d][multi_inds[d]][:,1]-multi_life_times[d][multi_inds[d]][:,0])))
            #min_persistence = np.minimum(min_persistence,min((multi_life_times[d][multi_inds[d]][:,1]+multi_life_times[d][multi_inds[d]][:,0])/2))
            if verbose:
                print("min persistence: "+str(min_persistence))
            if verbose:
                print("Picked life times in dim "+str(d)+":"+str(multi_life_times[d][multi_inds[d]]))
    new_multi_inds = []
    multi_reps = []
    inds0dim = pick_features_dim_0(multi_persistence[0], min_allowed_persistence = min_persistence, max_rel_quot = max_rel_quot, damping = damping, verbose = verbose)[:max_reps]
    multi_inds[0] = inds0dim
    for d, persistence in enumerate(multi_persistence):
        inds_now = []
        if d in only_dims:
            for i,ind in enumerate(multi_inds[d]):
                if multi_life_times[d][ind][1]-multi_life_times[d][ind][0]>max_persistence*max_total_quot:
                    inds_now.append(ind)
        new_multi_inds.append(inds_now)
        multi_reps.append([multi_representations[d][i] for i in inds_now])
    multi_inds = new_multi_inds
    if verbose:
        print("multi_reps:"+str(multi_reps))
    #print("Life times in dim 0:"+str(multi_life_times[0][multi_inds[0]]))
    return multi_reps, multi_inds, multi_life_times
    
    
def my_colour_maps():
    """
    Returns list of colour maps.
    """
    return [matplotlib.cm.Blues, matplotlib.cm.Reds,  matplotlib.cm.Greens, matplotlib.cm.Oranges, matplotlib.cm.Purples, matplotlib.cm.Greys, matplotlib.cm.Greys, matplotlib.cm.YlOrBr, matplotlib.cm.YlOrRd, matplotlib.cm.OrRd, matplotlib.cm.PuRd, matplotlib.cm.RdPu, matplotlib.cm.BuPu, matplotlib.cm.GnBu, matplotlib.cm.PuBu, matplotlib.cm.YlGnBu, matplotlib.cm.PuBuGn, matplotlib.cm.BuGn, matplotlib.cm.YlGn, matplotlib.cm.viridis, matplotlib.cm.inferno, matplotlib.cm.plasma, matplotlib.cm.magma, matplotlib.cm.cividis, matplotlib.cm.twilight, matplotlib.cm.twilight_shifted, matplotlib.cm.gist_earth, matplotlib.cm.terrain, matplotlib.cm.ocean, matplotlib.cm.gist_stern, matplotlib.cm.gist_heat, matplotlib.cm.gist_rainbow, matplotlib.cm.gist_ncar]

def pick_features(persistence, quot_persistence, max_rel_quot = 0.1, damping = 0, max_total_quot = 0.1, quotient_life_times = False, verbose = False):
    """
    Picks the most persistent representatives based on some threshold.
    """
    sort_life_times = sorted(persistence, reverse=True)
    if quotient_life_times:
        #sort_or_quots = [pers for _,pers in sorted(zip(quot_persistence,persistence), reverse= True)]
        sort_life_times = sorted(quot_persistence, reverse=True)
        relevance_list = quot_persistence
    else:
        sort_life_times = sorted(persistence, reverse=True)
        #sort_or_quots = sort_life_times
        relevance_list = persistence
    if quotient_life_times:
        damping = damping/4
    rel_quots = [sort_life_times[i+1]/sort_life_times[i]*(1+damping/(i+1)) for i in range(len(sort_life_times)-1)]
    min_pos_violated_total_quot = next(i for i,q in enumerate(sort_life_times+[0]) if q < max_total_quot*sort_life_times[0])+1
    pos_largest_relevance_drop = argmin(rel_quots[:min_pos_violated_total_quot])+1
    pos_first_rel_quot_violation = next(i for i,q in enumerate(rel_quots+[0]) if q < max_rel_quot)+1
    pos_first_inadmissible_element_combined = min(pos_largest_relevance_drop,pos_first_rel_quot_violation)
    interesting_indices = np.argpartition(relevance_list, -pos_first_inadmissible_element_combined)[-pos_first_inadmissible_element_combined:]
    indices_ordered = interesting_indices[np.argsort(-relevance_list[interesting_indices])]
    if verbose:
        print("Used rel quots + 1:")
    if verbose:
        print(rel_quots[:pos_largest_relevance_drop+1])
    return indices_ordered

def pick_features_dim_0(persistence, min_allowed_persistence = 0, max_rel_quot = 0.1, damping = 0, verbose = False):
    """
    Picks the most persistent representatives based on some threshold.
    """
    sort_life_times = sorted(persistence, reverse=True)
    for i,lf in enumerate(sort_life_times):
        if lf == 0:
            print("Zero life time in dim 0 at i="+str(i))
    rel_quots = [sort_life_times[i+1]/sort_life_times[i]+damping/(i+1) for i in range(len(sort_life_times)-1)][1:-(len(sort_life_times)-1)//2]
    minimal_rel_quot_pos = argmin(rel_quots)+2
    if verbose:
        print("Current minimal relative quotient position:"+str(minimal_rel_quot_pos))
    pos_first_rel_quot_violation = next(i for i,q in enumerate(rel_quots+[0]) if q < max_rel_quot)+2
    pos_first_min_persistence_violation = next(i for i,q in enumerate(sort_life_times[1:]+[0]) if q < min_allowed_persistence)+1
    min_admissible_pos = min(minimal_rel_quot_pos,pos_first_rel_quot_violation, pos_first_min_persistence_violation)
    relevant_indices = np.argpartition(persistence, -min_admissible_pos)[-min_admissible_pos:] 
    if verbose:
        print("Length ind2:"+str(len(relevant_indices)))
        print("value of min_admissible_pos:"+str(min_admissible_pos))
    indices_correct_order = relevant_indices[np.argsort(-persistence[relevant_indices])]
    if verbose:
        print("Used rel quots + 1 in dim 0:")
    if verbose:
        print(rel_quots[:min_admissible_pos+1])
    return indices_correct_order[1:]

def read_julia_reps_1_dim(dim,exponent, verbose = False, process_string = ''):
    """
    Reads the Julia output files for 1 dimension and returns data on homology reps and life times.
    """
    csv_file_path = "JuliaCommunication/JuliaOutputHomologyGenerators"+str(int(dim))+process_string+".csv"
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        data_list = []
        for row in csv_reader:
            data_list.append(row)
    if verbose:
        print("Cur exponent"+str(exponent))
    life_times_cur = np.array([[float(data[0])**exponent,float(data[1])**exponent] for data in data_list])
    representations_cur = [[ast.literal_eval(data[2]),ast.literal_eval(data[3])] for data in data_list]
    try:
        persistence_cur = life_times_cur[:,1]-life_times_cur[:,0]
    except:
        print(persistence_cur)
        sdfghke
    better_reps_cur = []
    for representations in representations_cur:
            cur_rep = []
            for i, rep in enumerate(representations[0]):
                cur_rep.append([np.array(rep)-1,(representations[1][i]+1)%3-1])
            better_reps_cur.append(cur_rep)
    return better_reps_cur, life_times_cur, persistence_cur

def compute_harmonic_projection(Bkm,Bk,vec, verbose = False):
    """
    Computes the harmonic projection of a vector onto the kernel of the boundary operators.
    """
    grad_gen = scipy.sparse.linalg.lsmr(Bkm.T, vec)[0]
    grad_component = Bkm.T@grad_gen
    curl_gen = scipy.sparse.linalg.lsmr(Bk, vec-grad_component)[0]
    curl_component = Bk@curl_gen
    if verbose:
        print("grad component norm: ",np.linalg.norm(grad_component))
    if verbose:
        print("curl component norm: ",np.linalg.norm(curl_component))
    return vec-grad_component-curl_component

def compute_harmonic_projection_zero(Bk,vec):
    """
    Computes the harmonic projection of a vector onto the kernel of the boundary operators.
    """
    curl_gen = scipy.sparse.linalg.lsmr(Bk, vec)[0]
    curl_component = Bk@curl_gen
    #print("grad component norm: ",np.linalg.norm(grad_component))
    #print("curl component norm: ",np.linalg.norm(curl_component))
    return vec-curl_component



def compute_weighted_harmonic_projection(Bkm,Bk,vec,weights, verbose = False):
    """
    Computes the weighted harmonic projection of a vector onto the kernel of the boundary operators.
    """
    exp = 0.5
    vec = vec/(weights**exp+e-13)
    weight_matrix = scipy.sparse.diags(weights**exp)
    inv_weight_matrix = scipy.sparse.diags(1/weights**exp)
    grad_gen = scipy.sparse.linalg.lsmr(weight_matrix@Bkm.T, vec)[0]
    grad_component = weight_matrix@Bkm.T@grad_gen
    curl_gen = scipy.sparse.linalg.lsmr(inv_weight_matrix@Bk, vec-grad_component)[0]
    curl_component = inv_weight_matrix@Bk@curl_gen
    if verbose:
        print("grad component norm: ",np.linalg.norm(grad_component))
    if verbose:
        print("curl component norm: ",np.linalg.norm(curl_component))
    return vec-grad_component-curl_component

def compute_weighted_harmonic_projection_num_tris(Bkm,Bk,vec,exponent=-1):
    """
    Computes the weighted harmonic projection by number of upper-adjacent simplices of a vector onto the kernel of the boundary operators.
    """
    tri_counts = np.array(np.sum(np.abs(Bk),axis=1)).flatten()+1
    #print(tri_counts)
    weights = np.power(tri_counts,exponent)
    return compute_weighted_harmonic_projection(Bkm,Bk,vec,weights)

def compute_weighted_harmonic_projection_tri_kernel(Bkm,Bk,vec,exponent = 1, band_width = 0):
    tri_kernel = np.array(np.sum(np.abs(Bk),axis=1)).flatten()+1
    #print(tri_counts)
    weights = np.power(tri_counts,exponent)
    return compute_weighted_harmonic_projection(Bkm,Bk,vec,weights)
#from re import A

def minmax_landmark_sampling(points, n_landmarks):
    """Minmax landmark sampling.
    Parameters
    ----------
    points : ndarray, shape (n_points, n_features)
        The points to sample landmarks from.
    n_landmarks : int
        The number of landmarks to sample.
    Returns
    -------
    landmarks : ndarray, shape (n_landmarks, n_features)
        The sampled landmarks.
    """
    n_points = points.shape[0]
    landmarks = np.zeros((n_landmarks, points.shape[1]))
    landmarks_index = np.zeros(n_landmarks)
    distances = np.zeros((n_landmarks, points.shape[0]))
    new_index = np.random.randint(n_points)
    landmarks[0] = points[new_index]
    landmarks_index[0] = new_index
    for i in range(1, n_landmarks):
        new_distances = sklearn.metrics.pairwise_distances(
            points, landmarks[i-1:i])
        distances[i-1] = np.squeeze(new_distances)
        min_distances = np.min(distances[:i], axis=0)
        new_index = np.argmax(min_distances)
        landmarks_index[i] = new_index
        landmarks[i] = points[new_index]
    return landmarks, landmarks_index


def plot_points():
    """
    Plots the points.
    """
    base_points = np.genfromtxt('JuliaCommunication/PointsForJulia.csv', delimiter=',')
    ambient_dim = base_points.shape[1]
    if ambient_dim>=3:
        fig = px.scatter_3d(x=base_points[:,0], y=base_points[:,1], z=base_points[:,2], opacity=0.5, width=1000, height=1000)
    else:
        fig = px.scatter(x = base_points[:,0], y = base_points [:,1])
    fig.update_traces(marker=dict(size=2))
    fig.show()
    return ambient_dim

def my_kernel(x, y, band_width):
    return np.exp(-np.linalg.norm(x-y, ord=2)**2/band_width**2)

def my_threshold(value,threshold, thresholding_type = 'linear'):
    if thresholding_type == 'linear':
        return np.minimum(value,threshold)/threshold
    elif thresholding_type == 'hard':
        if value < threshold:
            return 0
        else:
            return 1
    return np.minimum(value,threshold)


def compute_upper_weights(simplices, threshold, base_points):
    # Compute the weights of the triangles
    weights = np.zeros(len(simplices))
    for i, simplex in enumerate(simplices):
        weights[i] = np.product([my_kernel(base_points[simplex[i0]], base_points[simplex[i1]], threshold) for i0 in range(len(simplex)) for i1 in range(i0+1,len(simplex))])
    return np.array(weights)

def compute_lower_weights(upper_weights, Bk):
    abs_Bk = abs(Bk)
    lower_weights = abs_Bk@upper_weights
    return lower_weights # what happens with 0-values?

def compute_forman_curvature(Bkm,Bk,weights, exponent_mode = False, exponent = 1):
    """
    Computes the Forman curvature of a simplicial complex.
    """
    if exponent_mode:
        tri_counts = np.array(np.sum(np.abs(Bk),axis=1)).flatten()+1
        weights = np.power(tri_counts,exponent)
    inv_weights = 1/weights
    Bkm_pos = np.abs(Bkm)
    Bk_pos = np.abs(Bk)
    curv_tri_part = np.multiply(np.array((np.sum(Bk_pos.T, axis=0))).flatten(),weights)
    curv_point_part = np.multiply(np.array(np.sum(Bkm_pos, axis=0)).flatten(),inv_weights)
    L_up_pos = Bk_pos@Bk_pos.T
    A_up_pos = L_up_pos-scipy.sparse.diags(L_up_pos.diagonal())
    L_down_pos = Bkm_pos.T@Bkm_pos
    A_down_pos = L_down_pos-scipy.sparse.diags(L_down_pos.diagonal())
    curv_edges_up = scipy.sparse.diags(np.sqrt(weights))@A_up_pos@scipy.sparse.diags(np.sqrt(weights))
    curv_edges_down = scipy.sparse.diags(np.sqrt(inv_weights))@A_down_pos@scipy.sparse.diags(np.sqrt(inv_weights))
    curv_edges_total = np.array(np.sum(np.abs(curv_edges_up-curv_edges_down), axis=0)).flatten()
    return np.multiply((curv_tri_part+curv_point_part-curv_edges_total), weights)
