import shutil
import numpy as np
from data_utils import *
from numpy.testing import assert_array_equal, assert_raises, assert_, run_module_suite


class TestDataUtils:

	def test_relabel_on_norm_context(self):
	    context_dict = {  
	        0: 
	            {
	                'normal':{0,1,2}, 
	            }, 
	        1: 
	            {
	                'normal':{3,4,5}, 
	            }, 
	        2: 
	            {
	                'normal':{6,7,8,9}, 
	            }
	        
	        }
	    # Check type is being correctly checked. Typecast array to float.
	    wrong_test_y = np.array([0,9,1,8,2,7,3,6,4,5,0,2,4,6,8,1,3,5,7,9], dtype=np.float32)
	    assert_raises(ValueError, relabel_on_norm_context, wrong_test_y,  'normal', context_dict)
	    # Check that relabelling as expected. 
	    test_y = np.array([0,9,1,8,2,7,3,6,4,5,0,2,4,6,8,1,3,5,7,9])
	    expected_y= np.array([0,2,0,2,0,2,1,2,1,1,0,0,1,2,2,0,1,1,2,2])
	    relabelled_y = relabel_on_norm_context(test_y,  'normal', context_dict)
	    assert_array_equal(expected_y, relabelled_y)
	    # Testing that expection is raised when sets are not pairwise disjoint
	    context_dict_wrong = {  
	        0: 
	            {
	                'normal':{0,1,2}, 
	            }, 
	        1: 
	            {
	                'normal':{2,4,5}, 
	            }, 
	        2: 
	            {
	              'normal':{6,7,8,9}, 
	            }
	        }
	    assert_raises(ValueError, relabel_on_norm_context, test_y,  'normal', context_dict_wrong)
		
	def test_shuffle_and_split(self):
	    X = np.arange(0,100)
	    X = X.reshape(4, 5, 5)
	    y = np.arange(0,4)
	    random_state = 123
	    # Expected with random seed as 123
	    expected = {
	        0:
	                {
	                    'X': np.array([[[75, 76, 77, 78, 79],
	                                    [80, 81, 82, 83, 84],
	                                    [85, 86, 87, 88, 89],
	                                    [90, 91, 92, 93, 94],
	                                    [95, 96, 97, 98, 99]],

	                                   [[ 0,  1,  2,  3,  4],
	                                    [ 5,  6,  7,  8,  9],
	                                    [10, 11, 12, 13, 14],
	                                    [15, 16, 17, 18, 19],
	                                    [20, 21, 22, 23, 24]]]),

	                    'y': np.array([3, 0]) 
	            },
	        1:
	                {
	                    'X' :  np.array([[[25, 26, 27, 28, 29],
	                                    [30, 31, 32, 33, 34],
	                                    [35, 36, 37, 38, 39],
	                                    [40, 41, 42, 43, 44],
	                                    [45, 46, 47, 48, 49]],

	                                   [[50, 51, 52, 53, 54],
	                                    [55, 56, 57, 58, 59],
	                                    [60, 61, 62, 63, 64],
	                                    [65, 66, 67, 68, 69],
	                                    [70, 71, 72, 73, 74]]]),
	                    'y': np.array([1, 2]) 
	                }
	           }

	    k = 2
	    context_dict_ex = shuffle_and_split(X, y, k, seed=random_state)
	    assert_(sorted(context_dict_ex.keys()) == sorted(expected.keys()))
	    for key in sorted(expected.keys()):
	        assert_array_equal(expected[key]['X'],context_dict_ex[key]['X'])
	        assert_array_equal(expected[key]['y'],context_dict_ex[key]['y'])
	
	def test_relabel_norm_anom_keep_anom(self):      
	    context_dict = {  
	        0: 
	            {
	                'normal':{0,1,2}, 
	                'anom': {3}
	            }, 
	        1: 
	            {
	                'normal':{3,4,5}, 
	                'anom': {2}
	            }

	        }

	    split_dict = {
	        0:
	            {
	                'X': np.array([[0,  1,  2,  3,  4],
	                               [5,  6,  7,  8,  9],
	                               [20, 21, 22, 23, 24]]),

	                'y': np.array([3, 0, 5]) 
	            },
	        1:
	            {
	                'X' :  np.array([[10, 11, 12, 13, 14],
	                                 [15, 16, 17, 18, 19]]),
	                'y': np.array([4, 2]) 
	            }
	           }


	    expected = {
	        0 : 
	            {
	                'X': np.array([[0,  1,  2,  3,  4],
	                               [5,  6,  7,  8,  9]]),

	                'y': np.array([1, 0])   
	            },
	        1:
	            {
	                'X' :  np.array([[10, 11, 12, 13, 14],
	                                 [15, 16, 17, 18, 19]]),
	                'y': np.array([0, 1]) 
	            }    

	    }  

	    context_split_dict = relabel_norm_anom(split_dict, 
	                                           context_dict, 
	                                           'normal', 
	                                           'anom', 
	                                           keep_anom=True)
	    assert_(sorted(context_split_dict.keys()) == sorted(expected.keys()))
	    for key in sorted(expected.keys()):
	        assert_array_equal(expected[key]['X'],context_split_dict[key]['X'])
	        assert_array_equal(expected[key]['y'],context_split_dict[key]['y'])

	def test_relabel_norm_anom(self):      
	    context_dict = {  
	        0: 
	            {
	                'normal':{0,1,2}, 
	                'anom': {3}
	            }, 
	        1: 
	            {
	                'normal':{3,4,5}, 
	                'anom': {2}
	            }

	        }

	    split_dict = {
	        0:
	            {
	                'X': np.array([[0,  1,  2,  3,  4],
	                               [5,  6,  7,  8,  9],
	                               [20, 21, 22, 23, 24]]),

	                'y': np.array([3, 0, 5]) 
	            },
	        1:
	            {
	                'X' :  np.array([[10, 11, 12, 13, 14],
	                                 [15, 16, 17, 18, 19]]),
	                'y': np.array([4, 2]) 
	            }
	           }


	    expected = {
	        0 : 
	            {
	                'X': np.array([[5,  6,  7,  8,  9]]),

	                'y': np.array([0])   
	            },
	        1:
	            {
	                'X' :  np.array([[10, 11, 12, 13, 14]]),
	                'y': np.array([0]) 
	            }    

	    }  

	    context_split_dict = relabel_norm_anom(split_dict, 
	                                           context_dict, 
	                                           'normal', 
	                                           'anom')

	    assert_(sorted(context_split_dict.keys()) == sorted(expected.keys()))
	    for key in sorted(expected.keys()):
	        assert_array_equal(expected[key]['X'],context_split_dict[key]['X'])
	        assert_array_equal(expected[key]['y'],context_split_dict[key]['y'])


	def test_write_data_supervised(self):
	    context_dict = {  
	            0: 
	                {
	                    'normal':{0,1,2}, 
	                    'anom': {3}
	                }, 
	            1: 
	                {
	                    'normal':{3,4,5}, 
	                    'anom' : {0}
	                }, 
	            2: 
	                {
	                    'normal':{6,7,8,9}, 
	                    'anom':{3}
	                }

	            }

	    X = np.arange(1, 21)
	    X = X.reshape(10, 2)
	    # Check if wrong number of labels are passed. 
	    y_wrong = np.array([3,1,2,4,8,5,7,9,0])
	    y = np.array([3,1,2,4,8,5,7,9,0,6])
	    test_dir = "temp"
	    assert_raises(ValueError, write_data_supervised, X, y_wrong, 'normal',  context_dict, filepath=test_dir)
	    
	    write_data_supervised(X, y, context_dict, 'normal', filepath=test_dir)
	    X_loaded = np.load(os.path.join(test_dir, "X.npy"))
	    y_loaded = np.load(os.path.join(test_dir, "y.npy"))
	    y_expected = np.array([1,0,0,1,2,1,2,2,0,2])
	    assert_array_equal(X_loaded, X)
	    assert_array_equal(y_loaded, y_expected)
    

	def test_write_data_sep_context_anom_true(self):
	    context_dict = {  
	            0: 
	                {
	                    'normal':{0,1,2}, 
	                    'anom': {3}
	                }, 
	            1: 
	                {
	                    'normal':{3,4,5}, 
	                    'anom' : {0}
	                }, 
	            2: 
	                {
	                    'normal':{6,7,8,9}, 
	                    'anom':{3}
	                }

	            }

	    X = np.arange(1, 21)
	    X = X.reshape(10, 2)
	    # Check if wrong number of labels are passed. 
	    y_wrong = np.array([3,1,2,4,8,5,7,9,0])
	    y = np.array([3,1,2,4,8,5,7,9,0,6])
	    assert_raises(
	        ValueError, write_data_sep_context, X, y_wrong, context_dict, 'normal', 'anom', keep_anom=False, filepath="temp", seed=123)

	    X1_expected, y1_expected = np.array([[ 1,  2]]), np.array([1])
	    X2_expected, y2_expected = np.array([[17, 18],[7,  8]]), np.array([1, 0])
	    X3_expected, y3_expected = np.array([[13, 14], [19, 20]]), np.array([0, 0])
	    X_expected, y_expected = [X1_expected,X2_expected,X3_expected], [y1_expected,y2_expected,y3_expected]


	    temp_dir = "temp"
	    write_data_sep_context(X, y, context_dict, "normal", "anom", keep_anom=True, filepath=temp_dir, seed=123)

	    X1, y1 = np.load(os.path.join(temp_dir,"0/X.npy")), np.load(os.path.join(temp_dir,"0/y.npy"))
	    X2, y2 = np.load(os.path.join(temp_dir,"1/X.npy")), np.load(os.path.join(temp_dir,"1/y.npy"))
	    X3, y3 = np.load(os.path.join(temp_dir,"2/X.npy")), np.load(os.path.join(temp_dir,"2/y.npy"))
	    Xs, ys = [X1, X2, X3], [y1, y2, y3]
	    for i in range(len(Xs)):
	        assert_array_equal(Xs[i], X_expected[i])
	        assert_array_equal(ys[i], y_expected[i])
	    shutil.rmtree(temp_dir)

	def test_write_data_sep_context_anom_false(self):
	    context_dict = {  
	        0: 
	            {
	                'normal':{0,1,2}, 
	                'anom': {3}
	            }, 
	        1: 
	            {
	                'normal':{3,4,5}, 
	                'anom' : {0}
	            }, 
	        2: 
	            {
	                'normal':{6,7,8,9}, 
	                'anom':{3}
	            }

	        }

	    X = np.arange(1, 21)
	    X = X.reshape(10, 2)
	    # Check if wrong number of labels are passed. 
	    y_wrong = np.array([3,1,2,4,8,5,7,9,0])
	    y = np.array([3,1,2,4,8,5,7,9,0,6])
	    assert_raises(
	        ValueError, write_data_sep_context, X, y_wrong, context_dict, 'normal', 'anom', keep_anom=False, filepath="temp", seed=123)

	    # With seed 123
	    # X = [[ 9, 10],  y = [8, 3, 9, 5, 0, 4, 1, 7, 6, 2]
	    #      [ 1,  2],
	    #      [15, 16],
	    #      [11, 12],
	    #      [17, 18],
	    #      [ 7,  8],
	    #      [ 3,  4],
	    #      [13, 14],
	    #      [19, 20],
	    #      [ 5,  6]]

	    # Need to cast as int as otherwise the value automatically become float when it is an
	    # empty array, we use ints for all data in this case. 
	    X1_expected, y1_expected = np.array([], dtype=np.int64), np.array([], dtype=np.int64)
	    X2_expected, y2_expected = np.array([[7,  8]]), np.array([0])
	    X3_expected, y3_expected = np.array([[13, 14], [19, 20]]), np.array([0, 0])
	    X_expected, y_expected = [X1_expected,X2_expected,X3_expected], [y1_expected,y2_expected,y3_expected]

	    temp_dir = "temp"
	    write_data_sep_context(X, y, context_dict, "normal", "anom", keep_anom=False, filepath=temp_dir, seed=123)

	    X1, y1 = np.load(os.path.join(temp_dir,"0/X.npy")), np.load(os.path.join(temp_dir,"0/y.npy"))
	    X2, y2 = np.load(os.path.join(temp_dir,"1/X.npy")), np.load(os.path.join(temp_dir,"1/y.npy"))
	    X3, y3 = np.load(os.path.join(temp_dir,"2/X.npy")), np.load(os.path.join(temp_dir,"2/y.npy"))
	    Xs, ys = [X1, X2, X3], [y1, y2, y3]
	    for i in range(len(Xs)):
	        # Implied checking for empty arrays (empty arrays can apparently have different shapes in numpy!)
	        if Xs[i].shape[0] !=0 and X_expected[i].shape[0] !=0:
	            assert_array_equal(Xs[i], X_expected[i])
	        if ys[i].shape[0] != 0 and y_expected[i].shape[0] !=0:
	            assert_array_equal(ys[i], y_expected[i])

	    shutil.rmtree(temp_dir)

    
	def test_write_data_all_contexts(self):
	    context_dict = {  
	            0: 
	                {
	                    'normal':{0,1,2}, 
	                    'anom': {3}
	                }, 
	            1: 
	                {
	                    'normal':{3,4,5}, 
	                    'anom' : {0}
	                }, 
	            2: 
	                {
	                    'normal':{6,7,8,9}, 
	                    'anom':{3}
	                }

	            }

	    X = np.arange(1, 21)
	    X = X.reshape(10, 2)
	    # Check if wrong number of labels are passed. 
	    y_wrong = np.array([3,1,2,4,8,5,7,9,0])
	    y = np.array([3,1,2,4,8,5,7,9,0,6])
	    temp_dir = "temp"
	    assert_raises(
	        ValueError, write_data_all_contexts, X, y_wrong, context_dict, 'normal', 'anom', keep_anom=True, filepath=temp_dir, seed=123)
	    
	    X_expected = np.array([[ 1,  2], [17, 18] ,[7,  8], [13, 14], [19, 20]])
	    y_expected = np.array([1, 1, 0, 0, 0])
	    context_expected = np.array([0, 1, 1, 2, 2])
	    
	    write_data_all_contexts(X, y, context_dict, "normal", "anom", keep_anom=True, filepath=temp_dir, seed=123)
	    X_all = np.load(os.path.join(temp_dir,"X.npy"))
	    y_all = np.load(os.path.join(temp_dir,"y.npy"))
	    context_all = np.load(os.path.join(temp_dir,"contexts.npy")) 
	    
	    
	    assert_array_equal(X_expected, X_all)
	    assert_array_equal(y_expected, y_all)
	    assert_array_equal(context_expected, context_all)

    
	def test_write_data_all_contexts(self):
	    context_dict = {  
	            0: 
	                {
	                    'normal':{0,1,2}, 
	                    'anom': {3}
	                }, 
	            1: 
	                {
	                    'normal':{3,4,5}, 
	                    'anom' : {0}
	                }, 
	            2: 
	                {
	                    'normal':{6,7,8,9}, 
	                    'anom':{3}
	                }

	            }

	    X = np.arange(1, 21)
	    X = X.reshape(10, 2)
	    # Check if wrong number of labels are passed. 
	    y_wrong = np.array([3,1,2,4,8,5,7,9,0])
	    y = np.array([3,1,2,4,8,5,7,9,0,6])
	    temp_dir = "temp"
	    assert_raises(
	        ValueError, write_data_all_contexts, X, y_wrong, context_dict, 'normal', 'anom', keep_anom=True, filepath=temp_dir, seed=123)
	    
	    X_expected = np.array([[ 1,  2], [17, 18] ,[7,  8], [13, 14], [19, 20]])
	    y_expected = np.array([1, 1, 0, 0, 0])
	    context_expected = np.array([0, 1, 1, 2, 2])
	    
	    write_data_all_contexts(X, y, context_dict, "normal", "anom", keep_anom=True, filepath=temp_dir, seed=123)
	    X_all = np.load(os.path.join(temp_dir,"X.npy"))
	    y_all = np.load(os.path.join(temp_dir,"y.npy"))
	    context_all = np.load(os.path.join(temp_dir,"contexts.npy")) 
	    
	    
	    assert_array_equal(X_expected, X_all)
	    assert_array_equal(y_expected, y_all)
	    assert_array_equal(context_expected, context_all)
	    shutil.rmtree(temp_dir)

	def test_write_data_all_contexts_false(self):
	    context_dict = {  
	            0: 
	                {
	                    'normal':{0,1,2}, 
	                    'anom': {3}
	                }, 
	            1: 
	                {
	                    'normal':{3,4,5}, 
	                    'anom' : {0}
	                }, 
	            2: 
	                {
	                    'normal':{6,7,8,9}, 
	                    'anom':{3}
	                }

	            }

	    X = np.arange(1, 21)
	    X = X.reshape(10, 2)
	    # Check if wrong number of labels are passed. 
	    y_wrong = np.array([3,1,2,4,8,5,7,9,0])
	    y = np.array([3,1,2,4,8,5,7,9,0,6])
	    temp_dir = "temp"
	    assert_raises(
	        ValueError, write_data_all_contexts, X, y_wrong, context_dict, 'normal', 'anom', keep_anom=True, filepath=temp_dir, seed=123)
	    
	    X_expected = np.array([[7,  8], [13, 14], [19, 20]])
	    y_expected = np.array([0, 0, 0])
	    context_expected = np.array([1, 2, 2])
	    
	    write_data_all_contexts(X, y, context_dict, "normal", "anom", keep_anom=False, filepath=temp_dir, seed=123)
	    X_all = np.load(os.path.join(temp_dir,"X.npy"))
	    y_all = np.load(os.path.join(temp_dir,"y.npy"))
	    context_all = np.load(os.path.join(temp_dir,"contexts.npy")) 
	    
	    
	    assert_array_equal(X_expected, X_all)
	    assert_array_equal(y_expected, y_all)
	    assert_array_equal(context_expected, context_all)
	    shutil.rmtree(temp_dir)
    
if __name__ == "__main__":
    run_module_suite()
