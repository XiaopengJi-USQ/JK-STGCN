from Utils import *
import os
import numpy as np
import scipy.io as sio
from scipy import signal
from preprocess.Preprocess import *



class Preprocess_ISRUC_S3(Preprocess):

    def __init__(self):
        super().__init__()
        u = Utils()
        self.dataset_config = u.ReadConfig('Dataset_ISRUC_S3')
        self.preprocess_config = u.ReadConfig('Preprocess_ISRUC_S3')
        self.data_file_list = u.GetFileList(self.dataset_config['original_data_path'], '.mat',
                                                self.preprocess_config['exclude_subjects_data'])
        self.label_file_list = u.GetFileList(self.dataset_config['label_path'], '.txt',
                                                 self.preprocess_config['exclude_subjects_label'])
        self.data_file_list.sort()
        self.label_file_list.sort()


    def Read1DataFile(self,file_name):
        file_path = os.path.join(self.dataset_config['original_data_path'],file_name)
        mat_data = sio.loadmat(file_path)
        resample = 3000
        psg_use = list()
        for each_channel in self.preprocess_config['channels_to_use']:
            psg_use.append(
                np.expand_dims(signal.resample(mat_data[each_channel], resample, axis=-1), 1))
        psg_use = np.concatenate(psg_use, axis=1)
        return psg_use


    def Read1LabelFile(self,file_name):
        file_path = os.path.join(self.dataset_config['label_path'], file_name)
        original_label = list()
        ignore = 30
        with open(file_path, "r") as f:
            for line in f.readlines():
                if (line != '' and line != '\n'):
                    label = int(line.strip('\n'))
                    original_label.append(label)
        return np.array(original_label[:-ignore])



if __name__ == '__main__':
    isruc_s3_process = Preprocess_ISRUC_S3()

    fold_label = []
    fold_data = []
    fold_len = []

    data_dir = isruc_s3_process.dataset_config['original_data_path']
    label_dir = isruc_s3_process.dataset_config['label_path']

    for i in range(0,len(isruc_s3_process.data_file_list)):
        print('Read data file:', isruc_s3_process.data_file_list[i],' label file:',isruc_s3_process.label_file_list[i])
        data_path = os.path.join(data_dir, isruc_s3_process.data_file_list[i])
        data = isruc_s3_process.Read1DataFile(data_path)
        label_path =  os.path.join(label_dir, isruc_s3_process.label_file_list[i])
        label = isruc_s3_process.Read1LabelFile(label_path)
        print('data shape:', data.shape, ', label shape', label.shape)
        assert len(label) == len(data)
        # in ISRUC, 0-Wake, 1-N1, 2-N2, 3-N3, 5-REM
        label[label == 5] = 4  # make 4 correspond to REM
        fold_label.append(np.eye(5)[label])
        fold_data.append(data)
        fold_len.append(len(label))
    print('Preprocess over.')
    np.savez(os.path.join(isruc_s3_process.preprocess_config['save_path'], 'ISRUC_S3.npz'),
        Fold_data = fold_data,
        Fold_label = fold_label,
        Fold_len = fold_len
    )
    print('Saved to', os.path.join(isruc_s3_process.preprocess_config['save_path'], 'ISRUC_S3.npz'))



'''
output:
    save to $path_output/ISRUC_S3.npz:
        Fold_data:  [k-fold] list, each element is [N,V,T]
        N:subject, V:node, T:data points
        Fold_label: [k-fold] list, each element is [N,C]
        N:subject, C:label
        Fold_len:   [k-fold] list
'''







