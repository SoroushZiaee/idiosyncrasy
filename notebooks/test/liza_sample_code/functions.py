# Functions for the various files in this repository


# importing packages so that functions work
import numpy as np
import h5py
import pandas as pd
import sklearn as sk
import scipy.stats 
import math
import sys

# Reading HDF5 files
def print_hdf5_structure(file):
    def print_dataset(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}")
        elif isinstance(obj, h5py.Group):
            print(f"Group: {name}")

    with h5py.File(file, 'r') as f:
        f.visititems(print_dataset)

def print_hdf5_contents(file, contents):
    with h5py.File(file, 'r') as f:
        c = f[contents]
        if c.shape == ():
            c_val = c[()]
        else:
            c_val = c[:]
    return c_val

# Ko's shuffle method of getting indices for train and test splits for linear regressions (k-fold cross validation)

def regression_indices(number_of_data_values,number_of_groups,test_group,SEED):
    np.random.seed(SEED) # to ensure that we get the same shuffling everytime as long as the seed stays the same, change the seed and you reshuffle the indices
    index = np.arange(0,number_of_data_values) # prints a range from 0 to end-1, i.e., the input is the number of numbers to print starting from zero and in steps of one by default, need to specify the start to avoid multiple numbers in split arrays
    # the way that Ko uses this is as a way to list all the indices, so all of the things (i.e., 640 things in my case)
    np.random.shuffle(index) # shuffles the order of elements in the specified array
    groups= np.array_split(index,number_of_groups) # splits the indices into a specified number of individual arrays 
    test_indices = index[np.isin(index,groups[test_group])] # take the first array from the splits and find the indicies in that array in the group of overall indices, this creates a boolean index with the positions where the matching indices are as true, this is used to index all of the indices and create a test index array - these are the images that will be used for testing
    train_indices = index[np.logical_not(np.isin(index, test_indices))] # find the test indices in index and mark those positions with a boolean true, turn these values into boolean false, use this to index all the indices and get everything that is not a test index
    return train_indices, test_indices

# Function for running ridge regression with normal sampling
def ridge_normal_sample(labels, train_set_size):
    idx_train = np.random.normal(np.nanmean(labels),np.std(labels),size=train_set_size)
    return idx_train

# My neuron reliability function

def neuron_reliability(data,runs = 20):
    neurons_list = list(np.split(data, data.shape[1], axis=1))
    e_array = np.empty((20,data.shape[1]))
    for i in range(len(neurons_list)):
        np.random.seed(i)
        our_neuron = np.squeeze(neurons_list[i])
        for x in range(runs):
            random_indices = np.random.choice(data.shape[2], size=data.shape[2], replace=False)
            m_1 = np.nanmean(our_neuron[:,random_indices[0:(int((data.shape[2])/2))]], axis = 1)
            m_2 = np.nanmean(our_neuron[:,random_indices[int(((data.shape[2])/2)):int((data.shape[2]))]], axis = 1)
            R = scipy.stats.pearsonr(m_1,m_2)[0]
            e_array[x][i] = 2 * R / (1 + R) # applying Ko's spearman brown corrrelation splithalf reliabiluty correction found in consistency.py
    m_n = np.mean(e_array,axis = 0)
    s_n = np.std(e_array,axis=0)
    return m_n, s_n       


# My model reliability function, the setup is correct bc if you call the same thing by diff names you set up independent objects

def model_reliability (neural_data, position_data, regressor_type, n_groups):
    emp = np.empty(20)
    e_1 = np.empty(neural_data.shape[0])
    e_2 = np.empty(neural_data.shape[0])
    for x in range(20):
        np.random.seed(x)
        random_indices_2 = np.random.choice(neural_data.shape[2], size=neural_data.shape[2], replace=False)
        g_1 = np.nanmean(neural_data[:, :, random_indices_2[0:(int((neural_data.shape[2])/2))]], axis = 2)
        g_2 = np.nanmean(neural_data[:, :, random_indices_2[int(((neural_data.shape[2])/2)):int((neural_data.shape[2]))]], axis = 2)
        for i in range(n_groups):
            train_m, test_m = regression_indices(neural_data.shape[0], number_of_groups=n_groups, test_group= i, SEED=x)
            X_train_1 = g_1[train_m, :]
            X_train_2 = g_2[train_m, :]
            y_train = position_data[train_m]
            X_test_1 = g_1[test_m, :]
            X_test_2 = g_2[test_m, :]
            regressor_1 = regressor_type
            regressor_2 = regressor_type
            regressor_1.fit(X_train_1,y_train)
            regressor_2.fit(X_train_2,y_train)
            y_pred_1 = regressor_1.predict(X_test_1)
            y_pred_2 = regressor_2.predict(X_test_2)
            e_1[test_m] = np.squeeze(y_pred_1)
            e_2[test_m] = np.squeeze(y_pred_2)
        R_m = scipy.stats.pearsonr(e_1,e_2)[0]
        emp[x] = 2 * R_m / (1 + R_m)
    rel = np.mean(emp)
    return rel 

# Regressor reliability function that tests the split-half correlation of trained regressors across trials of the neural data (i.e., how reliable are the regressor predictions)
def model_reliability_2(neural_data, regressor, test_idx): 
    emp = np.empty(20)
    for x in range(20):
        np.random.seed(x)
        random_indices_2 = np.random.choice(neural_data.shape[2], size=neural_data.shape[2], replace=False)
        g_1 = np.nanmean(neural_data[:, :, random_indices_2[0:(int((neural_data.shape[2])/2))]], axis = 2)
        g_2 = np.nanmean(neural_data[:, :, random_indices_2[int(((neural_data.shape[2])/2)):int((neural_data.shape[2]))]], axis = 2)
        y_pred_1 = regressor.predict(g_1[test_idx,:])
        y_pred_2 = regressor.predict(g_2[test_idx,:])
        R_m = scipy.stats.pearsonr(np.squeeze(y_pred_1),np.squeeze(y_pred_2))[0]
        emp[x] = 2 * R_m / (1 + R_m)
    rel = np.mean(emp)
    return rel 


# My reliability filtering function

def reliability_filtering(neural_data, mean_neuron_correlation, metric = 0.5):
    b_index = mean_neuron_correlation > metric # find all the neurons with mean reliability > 0.2 and create a boolean array with True for all these neurons
    reliable_data = neural_data[:, b_index, :] # create a 3D matrix with only the data for the reliable neurons by indexing all the data with the positions of the reliable neurons in along the neuron axis
    return reliable_data

# My reliability filtering with time bins
def reliability_filtering_tb(neural_data, mean_neuron_correlation, metric = 0.5):
    b_index = mean_neuron_correlation > metric # find all the neurons with mean reliability > 0.2 and create a boolean array with True for all these neurons
    reliable_data = neural_data[:, :, :, b_index] # create a 3D matrix with only the data for the reliable neurons by indexing all the data with the positions of the reliable neurons in along the neuron axis
    return reliable_data


# My prediction accuracy as a function of neurons function, this one took too long (ran for 7 mins and never finished) -> trying one with less iterations

def prediction_accuracy(neural_data, position_data, host_regressor):
    # first get the index of how many neurons to test the prediction accuracy on
    e = np.array([])
    num_neurons = neural_data.shape[1]
    print(type(num_neurons))
    while num_neurons > 2:
        if int(num_neurons) % 2 == 0:
            e = np.append(e, int(num_neurons)) # concerned with making it even
            num_neurons = num_neurons / 2 # concerned with making the next num-neurons about half of the previous
        else: 
            e = np.append(e, int(num_neurons) - 1)
            num_neurons = (num_neurons - 1) / 2
    e = np.append(e, 2)
    print(f"End = {len(e)}")
    # set up to run the % explained variaince calculations 36 times for each group of neurons 
    e_array = np.empty((len(e), 36)) # the array where the exp_var for each run will be stored
    for i in range(len(e)):
        print(f"i = {i}")
        length_index = int(e[i]) # ensuring that the number of neurons sampled follows the calculated order
        for x in range(2):
            print(f"x = {x}")
            np.random.seed(x)
            our_neurons = np.mean(neural_data[:, np.random.choice(neural_data.shape[1], size=length_index, replace=False), :], axis = 2) # getting a certain number of random neurons from the number of neurons
            em_array = np.empty((len(position_data),36)) # create an empty vector that can hold the predictions for all 640 images
            for r in range(36):
                print(f"r = {r}")
                seed = r # change the seed (i.e., shuffle the random indices) 36 times 
                for p in range(10):
                   print(f"p = {p}")
                   train, test = regression_indices(len(position_data), test_group= p, SEED= seed) # use the function in functions.py to get the indices (i.e., the identifier of the images) in the train and test set 
                   X_train = our_neurons[train,:] # index the neural data for the images used for training from the neural variable (i.e., neural data = the X variable)
                   y_train = position_data[train] # index the position data for the images used for training from the x_pos variable (i.e., position data = the y variable)
                   X_test = our_neurons[test,:] # index the neural data for the images used for testing from the neural variable
                   regressor = host_regressor # retrieve the linear regression function from sklearn
                   regressor.fit(X_train, y_train) # the regressor trains itself based on the X and y data provided to it
                   y_pred = regressor.predict(X_test) # make predictions based on the data set aside for testing
                   em_array[test, seed] = np.squeeze(y_pred) # assign the prediction values for each image to the test indices and seed column
            y_pred_all = np.mean(em_array, axis= 1) # get the mean predictions for each image
            corr = scipy.stats.pearsonr(np.squeeze(position_data),y_pred_all)[0]
            reliability = model_reliability(neural_data[:, np.random.choice(neural_data.shape[1], size=length_index, replace=False), :], position_data, host_regressor) # get the model reliability
            exp_var = np.power(corr / np.sqrt(reliability), 2)
            e_array[i][x] = exp_var
    return e, e_array


# prediction accuracy function take 2, no iterations, only one exp_var for each group of neurons
def prediction_accuracy_2(neural_data, position_data, host_regressor):
    # first get the index of how many neurons to test the prediction accuracy on
    e = np.array([])
    num_neurons = neural_data.shape[1]
    while num_neurons > 2:
        if int(num_neurons) % 2 == 0:
            e = np.append(e, int(num_neurons)) # concerned with making it even
            num_neurons = num_neurons / 2 # concerned with making the next num-neurons about half of the previous
        else: 
            e = np.append(e, int(num_neurons) - 1)
            num_neurons = (num_neurons - 1) / 2
    e = np.append(e, 2)
    # set up to run the % explained variaince calculations 36 times for each group of neurons 
    e_array = np.empty(len(e)) # the array where the exp_var for each run will be stored
    for i in range(len(e)):
        length_index = int(e[i]) # ensuring that the number of neurons sampled follows the calculated order
        np.random.seed(i)
        our_neurons = np.mean(neural_data[:, np.random.choice(neural_data.shape[1], size=length_index, replace=False), :], axis = 2) # getting a certain number of random neurons from the number of neurons
        em_array = np.empty((len(position_data),36)) # create an empty vector that can hold the predictions for all 640 images
        for r in range(36):
            seed = r # change the seed (i.e., shuffle the random indices) 36 times 
            for p in range(10):
                train, test = regression_indices(len(position_data), test_group= p, SEED= seed) # use the function in functions.py to get the indices (i.e., the identifier of the images) in the train and test set 
                X_train = our_neurons[train,:] # index the neural data for the images used for training from the neural variable (i.e., neural data = the X variable)
                y_train = position_data[train] # index the position data for the images used for training from the x_pos variable (i.e., position data = the y variable)
                X_test = our_neurons[test,:] # index the neural data for the images used for testing from the neural variable
                regressor = host_regressor # retrieve the linear regression function from sklearn
                regressor.fit(X_train, y_train) # the regressor trains itself based on the X and y data provided to it
                y_pred = regressor.predict(X_test) # make predictions based on the data set aside for testing
                em_array[test, seed] = np.squeeze(y_pred) # assign the prediction values for each image to the test indices and seed column
        y_pred_all = np.mean(em_array, axis= 1) # get the mean predictions for each image
        corr = scipy.stats.pearsonr(np.squeeze(position_data),y_pred_all)[0]
        reliability = model_reliability(neural_data[:, np.random.choice(neural_data.shape[1], size=length_index, replace=False), :], position_data, host_regressor) # get the model reliability
        exp_var = np.power(corr / np.sqrt(reliability), 2)
        e_array[i] = exp_var
    return e, e_array


# prediction accuracy take 3, less for loops, more vectorization
def prediction_accuracy_3(neural_data, position_data, host_regressor):
    # first get the index of how many neurons to test the prediction accuracy on
    e = np.array([])
    num_neurons = neural_data.shape[1]
    while num_neurons > 2:
        if int(num_neurons) % 2 == 0:
            e = np.append(e, int(num_neurons)) # concerned with making it even
            num_neurons = num_neurons / 2 # concerned with making the next num-neurons about half of the previous
        else: 
            e = np.append(e, int(num_neurons) - 1)
            num_neurons = (num_neurons - 1) / 2
    e = np.append(e, 2)
    # set up to run the % explained variaince calculations 36 times for each group of neurons 
    n_reps = 36 # the number of times that we want to get the exp_var for each group of neurons
    e_array = np.empty(len(e)*n_reps) # the array where the exp_var for each run will be stored
    length_indices = np.sort(np.tile(e, n_reps)) # all of the groups, this change avoids another for loop
    # now we are getting the indices for the neurons in all the neuron groups
    np.random.seed(42)
    all_indices = np.random.choice(neural_data.shape[1], size=int(length_indices.sum()), replace=True)
    grouped_indices = {}
    for x in range(len(length_indices)):
        grouped_indices[x] = all_indices[:int(length_indices[x])]
        del(all_indices[:int(length_indices[x])])
    return length_indices,grouped_indices
    # indexing system done, we have all the indices for all the tests, but now we need to look at using this to actually run the model
    for x in range(len(e_array)):
        our_neurons = np.mean(neural_data[:, np.random.choice(neural_data.shape[1], size=int(length_indices[x]), replace=False), :], axis = 2) # getting a certain number of random neurons from the number of neurons
        em_array = np.empty((len(position_data),36)) # create an empty vector that can hold the predictions for all 640 images
        # this will change, but for now my thinking is we still need some way to run this group-by-group. moving away from the for loop model may require a rewrite of the entire model running paradigm
        for r in range(36):
            seed = r # change the seed (i.e., shuffle the random indices) 36 times 
            for p in range(10):
                train, test = regression_indices(len(position_data), test_group= p, SEED= seed) # use the function in functions.py to get the indices (i.e., the identifier of the images) in the train and test set 
                X_train = our_neurons[train,:] # index the neural data for the images used for training from the neural variable (i.e., neural data = the X variable)
                y_train = position_data[train] # index the position data for the images used for training from the x_pos variable (i.e., position data = the y variable)
                X_test = our_neurons[test,:] # index the neural data for the images used for testing from the neural variable
                regressor = host_regressor # retrieve the linear regression function from sklearn
                regressor.fit(X_train, y_train) # the regressor trains itself based on the X and y data provided to it
                y_pred = regressor.predict(X_test) # make predictions based on the data set aside for testing
                em_array[test, seed] = np.squeeze(y_pred) # assign the prediction values for each image to the test indices and seed column
        y_pred_all = np.mean(em_array, axis= 1) # get the mean predictions for each image
        corr = scipy.stats.pearsonr(np.squeeze(position_data),y_pred_all)[0]
        reliability = model_reliability(neural_data[:, np.random.choice(neural_data.shape[1], size=int(length_indices[x]), replace=False), :], position_data, host_regressor) # get the model reliability
        exp_var = corr / np.sqrt(reliability)
        e_array[x] = exp_var
    return e, e_array


# Logarithmic function for curve of best-fit for prediction accuracy function
def logarithmic_func(x, a, b):
    return a * np.log(x) + b

# Function for generating null distributions
def null_dist(data_1, data_2, regressor, n_reps, n_neurons):
    null_1 = np.empty((data_1.shape))
    null_2 = np.empty((data_2.shape))
    null_mean_delta_dis = np.empty(n_reps)
    regressor = regressor
    for r in range(n_reps):
        np.random.seed(r)
        shuffle_index = np.random.choice(n_neurons, size=n_neurons, replace=False)
        null_1[:,shuffle_index[0:int(n_neurons/2)]] = data_1[:,shuffle_index[0:int(n_neurons/2)]]
        null_2[:,shuffle_index[int(n_neurons/2):int(n_neurons)]] = data_1[:,shuffle_index[int(n_neurons/2):int(n_neurons)]]
        null_2[:,shuffle_index[0:int(n_neurons/2)]] = data_2[:,shuffle_index[0:int(n_neurons/2)]]
        null_1[:,shuffle_index[int(n_neurons/2):int(n_neurons)]] = data_2[:,shuffle_index[int(n_neurons/2):int(n_neurons)]]
        y_pred_null_1 = regressor.predict(null_1)
        y_pred_null_2 = regressor.predict(null_2)
        null_delta = np.squeeze(y_pred_null_2 - y_pred_null_1)
        null_mean_delta_dis[r]= np.nanmean(null_delta)
    return null_mean_delta_dis

def check_nan(isnan):
    for value in isnan.values():
        if isinstance(value, float) and np.isnan(value):
            return True
        elif isinstance(value, np.ndarray) and np.isnan(value).any():
            return True
    return False

