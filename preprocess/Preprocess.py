#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import numpy as np

class Preprocess():


    def __init__(self):
        None


    def Read1DataFile(self,file_name):
        #output shape: (channels, all data points in one channels)
        None

    def Read1LabelFile(self,file_name):
        None

    def SavePreprocessedData(self, path_to_save, file_name, data_to_save, label_to_save):
        try:
            np.savez(os.path.join(path_to_save,file_name), data = data_to_save, label = label_to_save)
            print("Save data", file_name, "successfully!")
        except BaseException:
            print("Save data", file_name, "fail")
