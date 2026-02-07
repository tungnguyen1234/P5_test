
import numpy as np
from time import time
import random
import h5py
import os
from scipy.sparse import csc_matrix
from scipy import sparse
import pandas as pd

def load_big_data_Netflix(path='./', seed=1234, files=[1, 2]):
    tic = time()
    
    # Load and process data
    dfs = []
    movie_ids = []
    
    for i in files:
        path_file = os.path.join(path, f'combined_data_{i}.txt')
        df = pd.read_csv(path_file, header = None, names = ['userId', 'rating'], usecols = [0,1])
        df['rating'] = df['rating'].astype(float)
        dfs.append(df)
    
    data = pd.concat(dfs, ignore_index=True)
    data.index = np.arange(0,len(data))

    del dfs  # Free up memory

    data_nan = pd.DataFrame(pd.isnull(data.rating))
    data_nan = data_nan[data_nan['rating'] == True]
    data_nan = data_nan.reset_index()

    movie_np = []

    # Calculate the differences between consecutive indices
    diff = np.diff(data_nan['index'])

    # Create an array of movie IDs
    movie_ids = np.arange(1, len(diff) + 1)

    # Create the repeated movie IDs
    movie_np = np.repeat(movie_ids, diff - 1)

    # Account for the last record
    last_record = np.full(len(data) - data_nan.iloc[-1, 0] - 1, len(diff) + 1)
    movie_np = np.concatenate([movie_np, last_record])

    data = data[pd.notnull(data['rating'])]

    data['movieId'] = movie_np.astype(int)
    data['userId'] = data['userId'].astype(int)
    
    print(f'Data loaded in {time() - tic:.2f} seconds')
    
    # Create ratings matrix
    n_u = data['userId'].nunique()
    n_m = data['movieId'].nunique()
    
    # Create dictionaries for user and movie IDs
    udict = {id: i for i, id in enumerate(data['userId'].unique())}
    mdict = {id: i for i, id in enumerate(data['movieId'].unique())}
    
    np.random.seed(seed)
    data = data.sample(frac=1).reset_index(drop=True)

    # Convert user and movie IDs to their corresponding indices
    user_indices = data['userId'].map(udict).values
    movie_indices = data['movieId'].map(mdict).values

    # Create the ratings sparse matrix
    train_r = sparse.csr_matrix((data['rating'].values, (movie_indices, user_indices)), 
                                shape=(n_m, n_u), dtype='float32')

    # Create mask for non-zero entries (this will also be a sparse matrix)
    train_m = train_r.copy()
    train_m.data = np.ones_like(train_m.data, dtype='float32')
    
    return n_m, n_u, train_r, train_m

def load_big_data_ML_20M(path='./', seed=1234):
    tic = time()
    
    # MovieLens 20M uses comma as delimiter and has a header
    data = pd.read_csv(os.path.join(path, 'ratings.csv'), usecols=['userId', 'movieId', 'rating'], header=0)
    
    print(f'Data loaded in {time() - tic:.2f} seconds')

    n_u = data['userId'].nunique()  # num of users
    n_m = data['movieId'].nunique()  # num of movies
    n_r = len(data)  # num of ratings

    # Create dictionaries for user and movie IDs
    udict = {id: i for i, id in enumerate(data['userId'].unique())}
    mdict = {id: i for i, id in enumerate(data['movieId'].unique())}

    # Shuffle the data
    np.random.seed(seed)
    data = data.sample(frac=1).reset_index(drop=True)

    # Convert user and movie IDs to their corresponding indices
    user_indices = data['userId'].map(udict).values
    movie_indices = data['movieId'].map(mdict).values

    # Create the ratings sparse matrix
    train_r = sparse.csr_matrix((data['rating'].values, (movie_indices, user_indices)), 
                                shape=(n_m, n_u), dtype='float32')

    # Create mask for non-zero entries (this will also be a sparse matrix)
    train_m = train_r.copy()
    train_m.data = np.ones_like(train_m.data, dtype='float32')

    print('Data matrix loaded')
    print(f'Number of users: {n_u}')
    print(f'Number of movies: {n_m}')
    print(f'Number of ratings: {n_r}')
    return n_m, n_u, train_r, train_m

def load_data_100k(path='./', delimiter='\t'):

    train = np.loadtxt(path+'movielens_100k_u1.base', skiprows=0, delimiter=delimiter).astype('int32')
    test = np.loadtxt(path+'movielens_100k_u1.test', skiprows=0, delimiter=delimiter).astype('int32')
    total = np.concatenate((train, test), axis=0)

    n_u = np.unique(total[:,0]).size  # num of users
    n_m = np.unique(total[:,1]).size  # num of movies
    n_train = train.shape[0]  # num of training ratings
    n_test = test.shape[0]  # num of test ratings

    train_r = np.zeros((n_m, n_u), dtype='float32')


    for i in range(n_train):
        train_r[train[i,1]-1, train[i,0]-1] = train[i,2]

    for i in range(n_test):
        train_r[test[i,1]-1, test[i,0]-1] = test[i,2]

    train_m = np.greater(train_r, 1e-12).astype('float32')  # masks indicating non-zero entries

    print('data matrix loaded')
    print('num of users: {}'.format(n_u))
    print('num of movies: {}'.format(n_m))
    print('num of training ratings: {}'.format(n_train))
    return n_m, n_u, train_r, train_m


def load_data_100k_train_test(path='./', delimiter='\t'):

    train = np.loadtxt(path+'movielens_100k_u1.base', skiprows=0, delimiter=delimiter).astype('int32')
    test = np.loadtxt(path+'movielens_100k_u1.test', skiprows=0, delimiter=delimiter).astype('int32')
    total = np.concatenate((train, test), axis=0)

    n_u = np.unique(total[:,0]).size  # num of users
    n_m = np.unique(total[:,1]).size  # num of movies
    n_train = train.shape[0]  # num of training ratings
    n_test = test.shape[0]  # num of test ratings

    train_r = np.zeros((n_m, n_u), dtype='float32')
    test_r = np.zeros((n_m, n_u), dtype='float32')

    for i in range(n_train):
        train_r[train[i,1]-1, train[i,0]-1] = train[i,2]

    for i in range(n_test):
        test_r[test[i,1]-1, test[i,0]-1] = test[i,2]

    train_m = np.greater(train_r, 1e-12).astype('float32')  # masks indicating non-zero entries
    test_m = np.greater(test_r, 1e-12).astype('float32')

    print('data matrix loaded')
    print('num of users: {}'.format(n_u))
    print('num of movies: {}'.format(n_m))
    print('num of training ratings: {}'.format(n_train))
    print('num of test ratings: {}'.format(n_test))

    return n_m, n_u, train_r, train_m, test_r, test_m



def load_data_1m(path='./', delimiter='::', seed=1234):
    tic = time()
    print('reading data...')
    data = np.genfromtxt(path+'movielens_1m_dataset.dat', delimiter=delimiter)
    print('taken', time() - tic, 'seconds')

    n_u = np.unique(data[:,0]).size  # num of users
    n_m = np.unique(data[:,1]).size  # num of movies
    n_r = data.shape[0]  # num of ratings

    udict = {}
    for i, u in enumerate(np.unique(data[:,0]).tolist()):
        udict[u] = i
    mdict = {}
    for i, m in enumerate(np.unique(data[:,1]).tolist()):
        mdict[m] = i

    np.random.seed(seed)
    idx = np.arange(n_r)
    np.random.shuffle(idx)

    train_r = np.zeros((n_m, n_u), dtype='float32')

    for i in range(n_r):
        u_id = data[idx[i], 0]
        m_id = data[idx[i], 1]
        r = data[idx[i], 2]
        train_r[mdict[m_id], udict[u_id]] = r

    train_m = np.greater(train_r, 1e-12).astype('float32')  # masks indicating non-zero entries

    print('data matrix loaded')
    print('num of users: {}'.format(n_u))
    print('num of movies: {}'.format(n_m))

    return n_m, n_u, train_r, train_m


def load_data_1m_train_test(path='./', delimiter='::', frac=0.1, seed=1234):

    tic = time()
    print('reading data...')
    data = np.genfromtxt(path+'movielens_1m_dataset.dat', delimiter=delimiter)
    print('taken', time() - tic, 'seconds')

    n_u = np.unique(data[:,0]).size  # num of users
    n_m = np.unique(data[:,1]).size  # num of movies
    n_r = data.shape[0]  # num of ratings

    udict = {}
    for i, u in enumerate(np.unique(data[:,0]).tolist()):
        udict[u] = i
    mdict = {}
    for i, m in enumerate(np.unique(data[:,1]).tolist()):
        mdict[m] = i

    np.random.seed(seed)
    idx = np.arange(n_r)
    np.random.shuffle(idx)

    train_r = np.zeros((n_m, n_u), dtype='float32')
    test_r = np.zeros((n_m, n_u), dtype='float32')

    for i in range(n_r):
        u_id = data[idx[i], 0]
        m_id = data[idx[i], 1]
        r = data[idx[i], 2]

        if i < int(frac * n_r):
            test_r[mdict[m_id], udict[u_id]] = r
        else:
            train_r[mdict[m_id], udict[u_id]] = r

    train_m = np.greater(train_r, 1e-12).astype('float32')  # masks indicating non-zero entries
    test_m = np.greater(test_r, 1e-12).astype('float32')

    print('data matrix loaded')
    print('num of users: {}'.format(n_u))
    print('num of movies: {}'.format(n_m))
    print('num of training ratings: {}'.format(n_r - int(frac * n_r)))
    print('num of test ratings: {}'.format(int(frac * n_r)))

    return n_m, n_u, train_r, train_m, test_r, test_m

# ------------------------------
def load_matlab_file(path_file, name_field):
    
    db = h5py.File(path_file, 'r')
    ds = db[name_field]

    try:
        if 'ir' in ds.keys():
            data = np.asarray(ds['data'])
            ir   = np.asarray(ds['ir'])
            jc   = np.asarray(ds['jc'])
            out  = csc_matrix((data, ir, jc)).astype(np.float32)
    except AttributeError:
        out = np.asarray(ds).astype(np.float32).T

    db.close()

    return out

def load_data_monti(path='./'):
    # Load the 'M' matrix and 'Otraining' matrix from the MATLAB file
    M = load_matlab_file(path+'douban_monti_dataset.mat', 'M')

    # Calculate the number of users and movies from the 'M' matrix
    n_u = M.shape[0]  # num of users
    n_m = M.shape[1]  # num of movies

    # Count the number of training ratings
    n_train = M[np.where(M)].size

    # Transpose the training data to get the desired format (movies x users)
    train_r = M.T

    # Create a mask for non-zero entries in the training ratings matrix
    train_m = np.greater(train_r, 1e-12).astype('float32')

    print('data matrix loaded')
    print('num of users: {}'.format(n_u))
    print('num of movies: {}'.format(n_m))
    print('num of training ratings: {}'.format(n_train))

    return n_m, n_u, train_r, train_m


def load_data_monti_train_test(path='./'):

    M = load_matlab_file(path+'douban_monti_dataset.mat', 'M')
    Otraining = load_matlab_file(path+'douban_monti_dataset.mat', 'Otraining') * M
    Otest = load_matlab_file(path+'douban_monti_dataset.mat', 'Otest') * M

    n_u = M.shape[0]  # num of users
    n_m = M.shape[1]  # num of movies
    n_train = Otraining[np.where(Otraining)].size  # num of training ratings
    n_test = Otest[np.where(Otest)].size  # num of test ratings

    train_r = Otraining.T
    test_r = Otest.T

    train_m = np.greater(train_r, 1e-12).astype('float32')  # masks indicating non-zero entries
    test_m = np.greater(test_r, 1e-12).astype('float32')

    print('data matrix loaded')
    print('num of users: {}'.format(n_u))
    print('num of movies: {}'.format(n_m))
    print('num of training ratings: {}'.format(n_train))
    print('num of test ratings: {}'.format(n_test))

    return n_m, n_u, train_r, train_m, test_r, test_m

#----------------------------

##### Experiment 1
def check_no_repeats(samples_dict):
    # Concatenate all samples into a single list
    all_samples = sum(samples_dict.values(), [])
    # Check if the length of the list is the same as the length of the set made from the list
    return len(all_samples) == len(set(all_samples))


def filter_rows(A, sample_indices):
    valid_rows = []
    for i in range(A.shape[0]):
        if all(A[i, j] > 0 for j in sample_indices):
            valid_rows.append(i)
    return valid_rows

def shuffle_sublist(lst, start_index, end_index):
    """
    Shuffles elements of a list from start_index to end_index.

    Args:
    - lst: The list to shuffle.
    - start_index: The starting index for the shuffle (inclusive).
    - end_index: The ending index for the shuffle (inclusive).

    Returns:
    - The list with the specified sublist shuffled.
    """
    sublist = lst[start_index:end_index]
    random.shuffle(sublist)
    lst[start_index:end_index] = sublist
    return lst


def random_sampling_sequence(A, n, start = 20, end = 5):
    # Create a list of numbers from 0 to n
    numbers = list(range(n))

    # Shuffle the list to randomize the order of numbers
    random.shuffle(numbers)

    # Initialize an empty dictionary to store the samples
    samples_dict = {}

    # Initialize a variable to track the start index for slicing the shuffled list
    start_index = 0

    # Loop to generate samples with sizes decreasing from 20 to 5
    for sample_size in range(start, end - 1, -1):
        # Calculate the end index for slicing
        end_index = start_index + sample_size

        while True:
          # Directly shuffle the sublist without creating a new shuffled list every time
          numbers = shuffle_sublist(numbers, start_index, n)
          sample_indices = numbers[start_index:end_index]

          # Optimized filtering process (assuming filter_rows is efficient and necessary)
          users = filter_rows(A, sample_indices)

          if len(users) > 0:
              samples_dict[sample_size] = [sample_indices, users]
              break  # Exit the loop once a suitable sample is found

        # Update the start_index for the next slice
        start_index = end_index

    return samples_dict
##### Experiment 1


##### Experiment 2
def get_users_most_numbers(arr):
   # Target set of integers
    target_set = set(np.arange(0, 6, dtype=float))
                                                                                                                                 
    # Check for columns that match the target set exactly
    valid_columns_indices = np.array([col for col in range(arr.shape[1]) if set(arr[:, col]) == target_set])

    # Extract the relevant columns based on valid_columns_indices
    relevant_columns = arr[:, valid_columns_indices]

    # Count the number of non-zeros in each row of the selected columns
    nonzero_counts_per_row = np.count_nonzero(relevant_columns, axis=0)

    # Calculating median, mean
    median_val = np.median(nonzero_counts_per_row)
    mean_val = np.mean(nonzero_counts_per_row)

    # Finding mode and corresponding indices
    min_idx = valid_columns_indices[np.argmin(nonzero_counts_per_row)]
    max_idx = valid_columns_indices[np.argmax(nonzero_counts_per_row)] 
    # Finding index closest to median and mean
    median_idx = valid_columns_indices[np.abs(nonzero_counts_per_row - median_val).argmin()]
    mean_idx = valid_columns_indices[np.abs(nonzero_counts_per_row - mean_val).argmin()]

    # get mode idx
    # mode_idx = np.where(nonzero_counts_per_row == stats.mode(nonzero_counts_per_row)[0][0])[0]
   
    return min_idx, max_idx, median_idx, mean_idx

def get_list_of_products(user, A):
    # Initialize an empty list to store the indices
    indices = []
    
    # Loop through the numbers 1 to 5 to find matching column indices
    for j in range(1, 6):  # 1 to 5 inclusive
        # Find indices in row 'idx' where the value equals 'j'
        matched_indices = np.where(A.T[user] == j)[0]
        
        # If there are matching indices, randomly choose one and add to the list
        if matched_indices.size > 0:
            chosen_index = np.random.choice(matched_indices)
            indices.append(chosen_index)

    # Ensure we have found exactly 5 distinct indices
    if len(indices) == 5:
        return indices
    else:
        return []