import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import tqdm
import pickle
import argparse
import os
from itertools import combinations

from utils.util import (createCharacteristicMatrix, get_user_significances, get_commmon_users, get_sim_item_item, get_topz_item_neighbors, 
calculate_similarity_left, get_pearson_correlation_Sui, get_cosine_similarity_Sui, get_topk_user_neighbors, get_prediction)

def parse_args():
    '''
    Parses the CF arguments.
    '''
    parser = argparse.ArgumentParser(description="Colloborative Filtering based on significances", formatter_class= argparse.ArgumentDefaultsHelpFormatter)

    
    parser.add_argument('--input', nargs='?', default='./input/ratings.dat',
                        help='Input ratings.dat path')

    parser.add_argument('--num_recomms', type=int, default= 10,
                        help="Number of recommendations to be made")

    parser.add_argument('--user', type=int, default = 10,
                        help="User id for whom the recommendation has to be done")
                        
    parser.add_argument('--z', type=int, default=20,
                        help='Number of items neighbors to be taken')

    parser.add_argument('--s_measures', type=str, default='pearson',
                        help='similarity measure between users, one of "pearson", "cosine"')

    parser.add_argument('--k', default=40, type=int,
                        help='Number of user neighbors to be taken')

    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()

    filename = args.input # location of ratings.dat

    N = args.num_recomms # number of recommendations to be made
    user = args.user #user_id of whom the recommendations has to be made
    z = args.z # Number of items neighbors to be taken
    s_measure = args.s_measures # similarity measure between users, one of "pearson", "cosine"
    k = args.k # Number of user neighbors to be taken
    
    ds, df=createCharacteristicMatrix(filename)
    ds = ds  = ds/ ds.max() # Normalising the data between [0.2 , 1.0]

    n_users = ds.shape[1] # -> 6040
    n_items = ds.shape[0] # -> 3706

    zero_indices = list(zip(*np.where(ds == 0))) # retreiving those indices where no rating has been given
    non_zero_indices = list(zip(*np.where(ds)))

    item_significances = ds.sum(axis = 1)/n_users # calculating item significances
    user_significances = np.array([get_user_significances(user, n_items) for user in ds.T]) # calculating user significances

    ######## CALCULATING SIMILARITY BETWEEN ITEMS ########
    combs_items = list(combinations(range(0, n_items), 2))
    
    sim_xy = np.zeros(shape = (n_items, n_items))
    for items_tuple in tqdm.tqdm(combs_items):
        sim_xy[items_tuple] = sim_xy[items_tuple[1], items_tuple[0]] = get_sim_item_item(ds, items_tuple)

    z_item_neighbors = np.array([get_topz_item_neighbors(item, z)[1] for item in sim_xy])
    

    ######## CREATING THE SIGNIFICANCE MATRIX FOR USERS u and ITEMS i ########
    S_ui = ds * np.array(list(map(lambda x: 1 if x == 0.0 else x, user_significances)))
    S_ui = (S_ui.T* np.array(list(map(lambda x: 1 if x == 0.0 else x, item_significances)))).T

    for item_user_tuple in tqdm.tqdm(zero_indices):
        S_ui[item_user_tuple] = calculate_similarity_left(item_user_tuple, ds, sim_xy, z_item_neighbors, user_significances, item_significances)

    ######## CALCULATING S-MEASURES AMONGSTS USERS ########

    combs_users = list(combinations(range(0, n_users), 2))
    sims_xy  = np.zeros(shape = (n_users, n_users))

    if s_measure == 'pearson':
        
        for i, users_tuple in enumerate(tqdm.tqdm(combs_users)):
            a = S_ui[:, users_tuple[0]]
            b = S_ui[:, users_tuple[1]]
            try:
                sims_xy[users_tuple] = sims_xy[users_tuple[1], users_tuple[0]] = get_pearson_correlation_Sui(a, b, S_ui)
            except:
                sims_xy[users_tuple] = np.nan
    
    elif s_measure == 'cosine':

        for i, users_tuple in enumerate(tqdm.tqdm(combs_users)):
            
            a = S_ui[:, users_tuple[0]]
            b = S_ui[:, users_tuple[1]]
            
            try:
                sims_xy[users_tuple] = sims_xy[users_tuple[1] ,users_tuple[0]] = get_cosine_similarity_Sui(a, b, S_ui)
            except:
                sims_xy[users_tuple] = np.nan
    
    else:
        print('The given similarity measure is not defined.')
        exit()

    # FINDING TOP-K USER NEIGHBORS
    k_user_neighbors = np.array([get_topk_user_neighbors(user_id, k)[1] for user_id in sims_xy])

    # Predictions
    temp = np.where(np.array(zero_indices)[:, 1] == user)[0]
    user_zero_indices = np.array([zero_indices[i] for i in temp])
    all_preds = np.array([get_prediction(item_id, user, ratings_data= ds, k_user_neighbors= k_user_neighbors, 
                                        sims_users_xy = sims_xy, aggregation_method = 'weighted') 
                        for item_id in user_zero_indices[:, 0]])
    recommended_items = [user_zero_indices[j][0] for j in np.argsort(all_preds)[::-1][:N]]
    
    print('Top {} items recommended for user {} are '.format(N, user), recommended_items)

    



    







