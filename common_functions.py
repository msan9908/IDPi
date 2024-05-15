"""
Created on Sep 13 2022
@author: IOB

Functions used for data prep in "A Catalyst-Agnostic Unsupervised Machine Learning Workflow for 
Assigning and Predicting Generality in Asymmetric Catalysis"
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from sklearn.cluster import KMeans

def remove_x(df, col, invalid_character='x'):
    """
    Remove invalid values (x) from a column
    df: DataFrame to work with
    col: Column that x is found
    """

    index_names = df[df[col]==invalid_character].index
    df.drop(index_names, inplace=True)
    df[col] = (pd.to_numeric(df[col]))  
    df = df.reset_index(drop=True)

    return(df)

def smi2RDKIT(smiles):
    """
    Convert list of SMILES strings to RDKIT descriptors
    """

    mols = []
    for i in smiles:
        x= str(i)
        mols.append(Chem.MolFromSmiles(x))

    descriptors = []
    calc = MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    for x in mols:
        descrip_x = calc.CalcDescriptors(x)
        gg = np.array(descrip_x)
        descriptors.append(gg)

        header = calc.GetDescriptorNames()
        descript_df = pd.DataFrame(descriptors, columns = header)

    return(descript_df)

def get_clusters(data,max_cluster):
    """
    Parameter
    data: input data for clustering
    max_cluster: highest cluster value to look for 
    """
    range_ = range(2,max_cluster)
    inertia = []
    for i in range_:
        kmeans = KMeans(
            init="k-means++",
            n_clusters=i,
            n_init=10,
            max_iter=300,
            random_state=25
        )
        cluster = kmeans.fit(data)
        inertia.append(cluster.inertia_)

    return(range_, inertia)

def k_cluster(data,num_clusters, init=25):
    """
    Obtain cluster labels for num_clusters with standard hyperparameters
    Parameter
    data: input data for clustering
    num_clusters: # of clusters
    init: random_state for clustering
    """
    kmeans = KMeans(
        init="k-means++",
        n_clusters=num_clusters,
        n_init=10,
        max_iter=300,
        random_state=init
    )
    
    fit = kmeans.fit(data)
    labels = fit.labels_

    return(labels)

def calc_generality(df, group, group_col, resp, cluster, cutoff):
    """
    Calculate generality scores (Proportion of clusters deemed succesful)
    for a group. Target values and groups must be in a pandas dataframe.

    Parameter
    df: DataFrame
    group: grouping (i.e. specific catalyst, catalyst class)
    group_col: Column name of groups
    resp: response column name (ee, yield, etc.)
    cluster: clustering labels
    cutoff: Value for success
    """
    
    cluster = pd.DataFrame({'Cluster':cluster})
    df = df.join(cluster).reset_index(drop=True)

    new_df = df[df[group_col] == group].reset_index(drop=True)


    num_clusters = len(cluster['Cluster'].unique())
    cluster_ = new_df['Cluster'].unique()

    cluster_ee = []
    cluster_number = []

    for cluster_val in cluster_:

        cluster_df = new_df[new_df['Cluster'] == cluster_val]
        avg_ee = np.average(cluster_df[resp])
        cluster_ee.append(avg_ee)
        cluster_number.append(cluster_val)

    cluster_ee_df = pd.DataFrame({'ee':cluster_ee})
    generality = np.sum(cluster_ee_df.ee >= cutoff)/num_clusters


    return(generality, cluster_number, cluster_ee)

def average_ee(cat_class, class_col, df, ee_name='ee'):
    """
    Determine average ee
    Parameter
    cat_class:  class under study
    class_col: column header of classes
    df: dataframe with data
    """

    new_df = df[df[class_col] == cat_class]
    avg_ee = np.average(new_df[ee_name])

    return avg_ee
