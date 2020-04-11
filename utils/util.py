import pandas as pd
import numpy as np

from scipy.stats import pearsonr
from numpy import dot
from numpy.linalg import norm



def createCharacteristicMatrix(filename):

	# convert the ratings data file to a pandas dataframe and and then convert the dataframe into the required numpy matrix and return the matrix
    data_frame = pd.read_csv(filename, sep="::", usecols = [0, 1, 2], names = ['userID', 'movieID', 'rating'], engine = 'python')
    data_mat = np.array(data_frame.pivot(index = 'movieID', columns = 'userID', values = 'rating'))
    data_mat_rev = np.nan_to_num(data_mat)
    return data_mat_rev.astype(int), data_frame

def get_user_significances(user_vector, n_items):
    
    W = {0, 0.2, 0.4, 0.6, 0.8, 1.0} # Rating values
    V = {0.6, 0.8, 1.0} # Relevant ratings
    Vc = W - V - {0} # Non relevant ratings

    C_Du = len(list(filter(lambda x: x in V, user_vector))) #cardinality set of items that user u has rated with a relevant value
    C_Eu = len(list(filter(lambda x: x in Vc, user_vector))) #cardinality set of items that user u has rated with a non-relevant value
    
    f1 = 1 - (C_Du/(C_Du + C_Eu))
    f2 = (C_Du + C_Eu)/n_items
    
    return f1 * f2

def get_commmon_users(dataset, items_tuple):
    
    i = items_tuple[0] #-> item_1 id
    j = items_tuple[1] #-> item_2 id
    
    item_1 = dataset[i].reshape(-1, 1) #-> size = n_users
    item_2 = dataset[j].reshape(-1, 1) #-> size = n_users
    
    a = set(np.where(item_1)[0]) # Finding the indices where item_1 vector has 0.0 values
    b = set(np.where(item_2)[0]) # Finding the indices where item_2 vector has 0.0 values
    
    return a.intersection(b) # return the user ids for item_1 and item_2 who have simultaneously rated both items

def get_sim_item_item(dataset, items_tuple):
    
    common_users = get_commmon_users(dataset, items_tuple)
    total = 0
    
    for user in common_users:
        total+= np.abs(dataset[:, user][items_tuple[0]] - dataset[:, user][items_tuple[1]])
    
    try:
        return 1 - ((1/len(common_users))*total)
    except ZeroDivisionError:
        return np.nan

def get_topz_item_neighbors(item_vector, z = 20):
    
    a = np.nan_to_num(item_vector)
    ind = np.argpartition(a, -z)[-z:]
    
    return a[ind[np.argsort(a[ind])]], ind[np.argsort(a[ind])] #->return top-z values and indices

def calculate_similarity_left(item_user_tuple, ds, sim_xy, z_item_neighbors, user_significances
                                , item_significances):
    
    item_id = item_user_tuple[0]
    user_id = item_user_tuple[1]

    S_k_item = z_item_neighbors[item_id]
    F_user_id = set(np.where(ds[:, user_id])[0]).intersection(set(S_k_item))

    mu = sum([sim_xy[nbr, item_id] for nbr in F_user_id])
    
    if not(mu):
        return 0.0
    else:
        return ((user_significances[user_id] * item_significances[item_id])/mu) * sum([item_significances[nbr] * ds[nbr, user_id] * sim_xy[nbr, item_id] for nbr in F_user_id])


def get_pearson_correlation_Sui(x, y, S_ui):

    a = set(np.where(x)[0])
    b = set(np.where(y)[0])
    
    B_xy = a.intersection(b)
    
    return pearsonr(np.array([x[i] for i in B_xy]), y = np.array([y[i] for i in B_xy]))[0]

def get_cosine_similarity_Sui(x, y, S_ui):
  return dot(x, y)/(norm(x)*norm(y))

def get_topk_user_neighbors(user_vector, k = 3):
    
    a = np.nan_to_num(user_vector)
    ind = np.argpartition(a, -k)[-k:]
    
    return a[ind[np.argsort(a[ind])]], ind[np.argsort(a[ind])] #->return top-k values and indices
    

def get_prediction(item_id, user_id, ratings_data, k_user_neighbors, sims_users_xy, 
                   aggregation_method = 'weighted'):
    
    Ku = k_user_neighbors[user_id] #get the top k neighbors of user
    G_ui = [Ku[j] for j in np.where([ratings_data[item_id, i] for i in Ku])[0]] #get those neighbors of user 
                                                                          #who have rated the item
    
    if aggregation_method == 'average':
        try:
            return round(sum([ratings_data[item_id, n] for n in G_ui])/len(G_ui), 3)
        except:
            return 0.0
    
    elif aggregation_method == 'weighted':
        try:
            mu = 1/sum([sims_users_xy[n, user_id] for n in G_ui])
            return mu * sum([ratings_data[item_id, n] * sims_users_xy[n, user_id] for n in G_ui])
        except:
            return 0.0
    else:
        return 0.0