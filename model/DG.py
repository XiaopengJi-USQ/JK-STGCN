import numpy as np


class kFoldGenerator():
    '''
    Data Generator
    '''
    k = -1      # the fold number
    x_list = [] # x list with length=k
    y_list = [] # x list with length=k

    # Initializate
    def __init__(self, x, y):
        if len(x) != len(y):
            assert False, 'Data generator: Length of x or y is not equal to k.'
        self.k = len(x)
        self.x_list = x
        self.y_list = y


    # Get i-th fold
    def getFold(self, i):
        isFirst = True
        val_id_list = [2,5,0,1,7,1,7,3,4,8]
        for p in range(self.k):
            if p != i:
                if p == val_id_list[i]:
                    val_data = self.x_list[p]
                    val_targets = self.y_list[p]
                else:
                    if isFirst:
                        train_data = self.x_list[p]
                        train_targets = self.y_list[p]
                        isFirst = False
                    else:
                        train_data = np.concatenate((train_data, self.x_list[p]))
                        train_targets = np.concatenate((train_targets, self.y_list[p]))
            else:
                test_data = self.x_list[p]
                test_targets = self.y_list[p]
        return train_data, train_targets, val_data, val_targets,test_data,test_targets
