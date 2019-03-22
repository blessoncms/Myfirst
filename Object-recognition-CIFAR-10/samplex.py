
import numpy as np
import matplotlib.pyplot as plt

def balanced_sample_maker(X, y, sample_size, random_seed=42):
    uniq_levels = np.unique(y)
    uniq_counts = {level: sum(y == level) for level in uniq_levels}

    if not random_seed is None:
        np.random.seed(random_seed)

    # find observation index of each class levels
    groupby_levels = {}
    for ii, level in enumerate(uniq_levels):
        obs_idx = [idx for idx, val in enumerate(y) if val == level]
        groupby_levels[level] = obs_idx
    # oversampling on observations of each label
    balanced_copy_idx = []
    for gb_level, gb_idx in groupby_levels.items():
        over_sample_idx = np.random.choice(gb_idx, size=sample_size, replace=True).tolist()
        balanced_copy_idx+=over_sample_idx
    np.random.shuffle(balanced_copy_idx)
    data_train=X[balanced_copy_idx]
    labels_train=y[balanced_copy_idx]
    newX = X[np.setdiff1d(np.arange(X.shape[0]), balanced_copy_idx)]
    newy = y[np.setdiff1d(np.arange(y.shape[0]), balanced_copy_idx)]
    print('DIFFID', newX.shape,newy.shape)
    if  ((len(data_train)) == (sample_size*len(uniq_levels))):
        print('number of sampled example ', sample_size*len(uniq_levels), 'number of sample per class ', sample_size, ' #classes: ', len(list(set(uniq_levels))))
    else:
        print('number of samples is wrong ')
    print('data train',data_train.shape, labels_train)	
    return (data_train,labels_train,newX,newy)


