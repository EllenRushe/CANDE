import os
import numpy as np
from sklearn.utils import shuffle




def relabel_on_norm_context(y,  normal_key, context_dict):
    '''Relabel data depending on the normal classes of each context. 
       We look at the true label and assign a context to it depending on this label. 
       A context is assigned based on whether the label is in the class set for a different context. 
       Note that the sets for contexts must be pairwise disjoint. 
       For decriminative part of algorithm.
    Args:
        y (1-D numpy array of ints): Array of target values. 
        normal_key (str): Key of element with set of normal classes as value. 
        context_dict (dict): Dictionary defining normal classes for each context. 
        This dictionary should be formatted as follows:        {
            0 : 
                {
                     normal_key: {class_1,..., class_m}
                },
                 .
                 .
                 . 
            k : 
                {
                     normal_key': {class_1,..., class_m}
                }
            }
        
    Returns:
        1-D numpy array: Relabelled target values. 
    '''
    # Check that labels are any type of integer to prevent type problems down the line. 
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError('label values must be integers')
    # Take normal sets 
    normal_sets = [context_dict[c]['normal'] for c in context_dict.keys()]
    # Test to ensure that classes are pairwise disjoint. 
    # https://stackoverflow.com/questions/22432814/check-if-a-collection-of-sets-is-pairwise-disjoint
    # Get the length of the union of all sets.
    merged_sets_len = len(set().union(*normal_sets))
    # Get the length of all individual elements (with any duplicate values)
    all_elem_len = sum([len(s) for s in normal_sets])
    # If sets are pairwise disjoint, then the length individual 
    # elements should be the same as the union of sets. 
    if merged_sets_len != all_elem_len:
        raise ValueError('Classes in context sets must be pairwise disjoint.')
    # Initialise new labels all as numpy.nan.
    # We don't do this in place because it is likely that the context 
    # class will be the same integer value as one of the original classes. 
    new_y = np.full(y.shape, np.nan)
    for c in context_dict.keys():
        # For numpy.isin to give the expected answer, we need to cast set to list. 
        new_y[np.isin(y, list(context_dict[c][normal_key]))] = c
    return new_y


def shuffle_and_split(X, y, k, seed=None): 
    '''
    Shuffle training data and corresponding targets, then split into k parts. 
    Parts are returned in a dictionary, with the segment index (int - indexed from zero) as the the key.
    Args:
        X (numpy array): Dataset to be shuffled and split. 
        y (numpy array): Corresponding labels for X. 
        k (int) : Number of splits for data. 
        seed (int): Seed for deterministic shuffling of data. 

        
    Returns:
       Dictionary of segments resulting from split. 
            key (int): index as the the key. 
            value : dictionary with X and y values for split at each index. 
    
    '''
    # Create context
    split_dict = dict()
    # Set seed for shuffle if argument is passed.
    random_state = seed if seed is not None else False
    # Shuffle: Sample data and corresponding labels without replacement.
    X_shuffle, y_shuffle = shuffle(X, y, random_state=random_state)
    # Create array of indices that can be used to split training.
    indices = np.arange(len(X_shuffle))
    # np.array_split will allow for an uneven split. 
    splits = np.array_split(indices, k)
    # Get data at each set of split indices and assign to corresponding context. 
    for idx in range(k):
        split_dict[idx] = {
            'X': X_shuffle[splits[idx]], 
            'y': y_shuffle[splits[idx]]

            }
        unique, counts = np.unique(split_dict[idx]['y'], return_counts=True)
        print("{}\tX shape: {}\ty shape:{}".format(idx, split_dict[idx]['X'].shape, split_dict[idx]['y'].shape))
        print("{}\t")
        print(unique)
        print(counts)
    return split_dict




def relabel_norm_anom(split_dict, context_dict, norm_key, anom_key, keep_anom=False):
    '''
    Removes examples that don't fall into either of the normal or anomalous classes. 
    Args:
        split_dict (dict): Data split into different sections, indexed from 0 to k. 
            0 : {
                    X : [...]
                    y : [...]
                            
            }
            .
            .
            .
            k : {
                    X : [...],
                    y : [...]
            }
        context_dict (dict): Dictionary containing classes to be included in normal and
                            anomalous classes.          
            {
            0 : 
                {
                     normal_key: {class_1,..., class_m},
                     anom_key: {class_1,..., class_m}
                },
                 .
                 .
                 . 
            k : 
                {
                     normal_key': {class_1,..., class_m},
                     anom_key: {class_1,..., class_m}

                }
            }
        norm_key (str or int): Key name corresponding to normal classes in context_dict.
        anom_key (str or int): Key name corresponding to anomalous classes in context_dict.
        keep_anom (bool): Boolean value indicating whether anomalous classes are kept or not.
    Return: 
        Dictionary with data filtered by normal/anomalous classes and relabelled with binary 
        labels indicating normal/anomalous data. Index indicates a particular context. 
            0 : {
                    X : [...]
                    y : [...]
                            
            }
            .
            .
            .
            k : {
                    X : [...],
                    y : [...]
            }
    
    '''
    # Initialise dictionary to add data for each context. 
    context_split_dict = dict()
    # Process each context.  
    for split in split_dict.keys():
        # Get labels or data to be relabelled along with corresponding data. 
        split_X = split_dict[split]['X']
        split_y = split_dict[split]['y']
        # Get normal classes for context. 
        norm_classes = context_dict[split][norm_key]
        # Get anomalous classes for context.
        # We don't want anomalies in the training set so return empty set if keep anom is False. 
        anom_classes = context_dict[split][anom_key] if keep_anom else {}
        
        # Set examples to True if they are in either the normal class or the anomalous class. 
        # Cast set to list or results will not be as expected. np.idx eeds 'array_like' data structure. 
        y_bool_all = np.in1d(split_y, list(norm_classes.union(anom_classes)))
        
        # Filter examples based on whether they are in the passed sets/ 
        X = split_X[y_bool_all==True]
        # Class labels are not yet binary. 
        y_raw = split_y[y_bool_all==True]
        # If there are anomalies, relabel the data as normal or anomaly. 
        y_bool_filtered = np.in1d(y_raw, list(norm_classes))
        y = np.where(
            y_bool_filtered==True,
            np.zeros(len(y_bool_filtered), dtype=np.int64),
            np.ones(len(y_bool_filtered), dtype=np.int64)
        )
        context_split_dict[split] = {
            'X': X,
            'y': y
        }
    return context_split_dict


def write_data_supervised(X, y, context_dict, norm_key, filepath=""):
    '''
    Splits data into different sections based on the contexts defined in context_dict. 
    Each context in the context_dict defines the classes which are in each context. 
    The data is split into len(context_dict) classes and data is labelled with the 
    context name. Data is saved to path specified by filepath. 
    Args:
        X (numpy array): Data array. 
        y (numpy): Corresponding labels for data. 
        context_dict (dict): Dictionary containing classes to be included in normal and anomalous classes. 
            0 : 
                {
                     normal_key: {class_1,..., class_m},
                     anom_key: {class_1,..., class_m}
                },
                 .
                 .
                 . 
            k : 
                {
                     normal_key': {class_1,..., class_m},
                     anom_key: {class_1,..., class_m}

                }
            }
        norm_key(str): String used in dictionary as key.
        filepath (str): String to save data, labels and context array. 
    Return: 
        None
    '''
    if X.shape[0] != y.shape[0]:
        raise ValueError('Shapes of X and y at dimension 0 must be equal. There must be as many labels as data points.')
    if not os.path.exists(filepath): 
        os.makedirs(filepath)
    # relabel data depending on context.
    relabelled_on_context = relabel_on_norm_context(y, norm_key, context_dict)
    not_nan_indices = np.argwhere(~np.isnan(relabelled_on_context))
    contexts = relabelled_on_context[not_nan_indices]
    new_X = X[not_nan_indices]
    # Save data and labels. 
    np.save(os.path.join(filepath, 'X'), new_X)
    np.save(os.path.join(filepath, 'contexts'), np.squeeze(contexts))
    print("{} X shape: {}".format(filepath, new_X.shape))
    print("{} contexts shape: {}".format(filepath, contexts.shape))

def write_data_sep_context(X, y, context_dict, norm_key, anom_key, keep_anom=False, filepath="", seed=None):
    '''
    Splits data into different sections, labels them as normal or an anomaly depending on the
    context. The number of splits is dictated by the number of contexts passed in the context_dict.
    Data is saved to path specified by filepath. 
 
    Args:
        X (numpy array): Data array. 
        y (numpy): Corresponding labels for data. 
        context_dict (dict): Dictionary containing classes to be included in normal and anomalous classes. 
            0 : 
                {
                     normal_key: {class_1,..., class_m},
                     anom_key: {class_1,..., class_m}
                },
                 .
                 .
                 . 
            k : 
                {
                     normal_key': {class_1,..., class_m},
                     anom_key: {class_1,..., class_m}

                }
            }
        norm_key (str or int): Key name corresponding to normal classes in context_dict.
        anom_key (str or int): Key name corresponding to anomalous classes in context_dict.
        keep_anom (bool): True if anomalies should be kept in the dataset, False otherwise.
        filepath (str): String to save data, labels and context array. 
        seed (int): Seed used to initialise pseudorandom numbers. 
    Return: 
        None
    '''
    if X.shape[0] != y.shape[0]:
        raise ValueError('Shapes of X and y at dimension 0 must be equal. There must be as many labels as data points. ')
    # Get the number of required dicts. 
    k = len(context_dict.keys())
    # Create corresponding number of datasets
    split_dict = shuffle_and_split(X, y, k, seed=seed)
    context_split_dict = relabel_norm_anom(split_dict,context_dict, norm_key, anom_key,  keep_anom=keep_anom)
    for context in context_split_dict.keys():
        context_dir = os.path.join(filepath, str(context))
        if not os.path.exists(context_dir):
            os.makedirs(context_dir)
        np.save(os.path.join(context_dir, 'X.npy'), context_split_dict[context]['X'])
        np.save(os.path.join(context_dir, 'y.npy'), context_split_dict[context]['y'])
        print("{} X context {} shape: {}".format(filepath, context, context_split_dict[context]['X'].shape))
        print("{} y context {} shape: {}".format(filepath, context, context_split_dict[context]['y'].shape))

def write_data_all_contexts(X, y, context_dict, norm_key, anom_key, keep_anom=False, filepath="", seed=None):
    '''
    Splits data into different sections, labels them as normal or an anomaly depending on the
    context. The number of splits is dictated by the number of contexts passed in the context_dict.
    Data is then recombined so contextual anomalies will be present in the data and saved to path 
    specified by filepath. 
    Args:
        X (numpy array): Data array. 
        y (numpy): Corresponding labels for data. 
        context_dict (dict): Dictionary containing classes to be included in normal and anomalous classes. 
            0 : 
                {
                     normal_key: {class_1,..., class_m},
                     anom_key: {class_1,..., class_m}
                },
                 .
                 .
                 . 
            k : 
                {
                     normal_key': {class_1,..., class_m},
                     anom_key: {class_1,..., class_m}

                }
            }
        norm_key (str or int) : Key name corresponding to normal classes in context_dict.
        anom_key (str or int) : Key name corresponding to anomalous classes in context_dict.
        keep_anom (bool) : True if anomalies should be kept in the dataset, False otherwise.
        filepath (str): String to save data, labels and context array. 
        seed (int) : Seed used to initialise pseudorandom numbers. 
    Return: 
        None
    '''
    if X.shape[0] != y.shape[0]:
        raise ValueError('Shapes of X and y at dimension 0 must be equal. There must be as many labels as data points.')
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    # Get the number of required dicts. 
    k = len(context_dict.keys())
    # Create corresponding number of datasets
    split_dict = shuffle_and_split(X, y, k, seed=seed)
    context_split_dict = relabel_norm_anom(split_dict, context_dict, norm_key, anom_key, keep_anom=keep_anom)
    
    Xs = []
    ys = []
    context_list = []
    for context in context_split_dict.keys():
        context_dir = os.path.join(filepath, str(context))
        Xs.append(context_split_dict[context]['X'])
        ys.append(context_split_dict[context]['y'])
        context_list.append(np.full(context_split_dict[context]['y'].shape, context))
    # Remove individual numpy arrays. 
    # Concatenate all examples together. 
    X_all = np.concatenate(Xs)
    y_all = np.concatenate(ys)
    contexts_all = np.concatenate(context_list)
    # Save all examples to disk. 
    np.save(os.path.join(filepath, "X.npy"), X_all)
    np.save(os.path.join(filepath, "y.npy"), y_all)
    np.save(os.path.join(filepath, "contexts.npy"), contexts_all)

    print("{} X shape: {}".format(filepath, X_all.shape))
    print("{} y shape: {}".format(filepath, y_all.shape))
    print("{} Contexts shape: {}".format(filepath, contexts_all.shape))



