# -*- coding: utf-8 -*-
import configparser
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs
import sklearn.metrics as metrics
import json
import keras


class Utils:

    def ReadConfig(self, file_name):
        work_dir = os.path.split(os.path.realpath(__file__))[0]
        config_path = os.path.join(work_dir,'config')+os.sep
        config_abs_path = os.path.join(config_path,file_name)+'.json'
        with open(config_abs_path, 'r', encoding='utf8') as fp:
            json_data = json.load(fp)
        return json_data

    def GetFileList(self, path, filter_words=None, exclude_files=list()):
        all_files = os.listdir(path)
        rs = list()

        if len(exclude_files) > 0 and filter_words:
            exclude_files = [i + filter_words for i in exclude_files]

        for each_file in all_files:
            if filter_words:
                if (filter_words in each_file) and (each_file not in exclude_files):
                    rs.append(each_file)
            else:
                if len(exclude_files) > 0:
                    exclude = False
                    for j in exclude_files:
                        if j in each_file:
                            exclude = True
                            break
                    if exclude == False:
                        rs.append(each_file)
                else:
                    rs.append(each_file)
        rs.sort()
        return rs

def AddContext_MultiSub(x, y, Fold_Num, context, i):
    '''
    input:
        x       : [N,V,F];
        y       : [N,C]; (C:num_of_classes)
        Fold_Num: [kfold];
        context : int;
        i       : int (i-th fold)
    return:
        x with contexts. [N',V,F]
    '''
    cut = context // 2
    fold = Fold_Num.copy()
    fold = np.delete(fold, -1)
    id_del = np.concatenate([np.cumsum(fold) - i for i in range(1, context)])
    id_del = np.sort(id_del)

    x_c = np.zeros([x.shape[0] - 2 * cut, context, x.shape[1], x.shape[2]], dtype=float)
    for j in range(cut, x.shape[0] - cut):
        x_c[j - cut] = x[j - cut:j + cut + 1]

    x_c = np.delete(x_c, id_del, axis=0)
    y_c = np.delete(y[cut: -cut], id_del, axis=0)
    return x_c, y_c

def AddContext_SingleSub(x, y, context):
    cut = int(context / 2)
    x_c = np.zeros([x.shape[0] - 2 * cut, context, x.shape[1], x.shape[2]], dtype=float)
    for i in range(cut, x.shape[0] - cut):
        x_c[i - cut] = x[i - cut:i + cut + 1]
    y_c = y[cut:-cut]
    return x_c, y_c


def VariationCurve(fit, val, yLabel, savePath, figsize=(9, 6)):
    plt.figure(figsize=figsize)
    plt.plot(range(1, len(fit) + 1), fit, label='Train')
    plt.plot(range(1, len(val) + 1), val, label='Val')
    plt.title('Model ' + yLabel)
    plt.xlabel('Epochs')
    plt.ylabel(yLabel)
    plt.legend()
    plt.savefig(savePath + 'Model_' + yLabel + '.png')
    plt.show()
    return

def PrintScore( true, pred, fold_acc, savePath=None, average='macro'):
    if savePath == None:
        saveFile = None
    else:
        saveFile = open(savePath + "Result.txt", 'a+')
    # Main scores
    F1 = metrics.f1_score(true, pred, average=None)
    print("Main scores:")
    print('Acc\tF1S\tKappa\tF1_W\tF1_N1\tF1_N2\tF1_N3\tF1_R', file=saveFile)
    print('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f' % (metrics.accuracy_score(true, pred),
                                                                  metrics.f1_score(true, pred, average=average),
                                                                  metrics.cohen_kappa_score(true, pred),
                                                                  F1[0], F1[1], F1[2], F1[3], F1[4]),
              file=saveFile)
    # Classification report
    print("\nClassification report:", file=saveFile)
    print(metrics.classification_report(true, pred, target_names=['Wake', 'N1', 'N2', 'N3', 'REM']), file=saveFile)
    # Confusion matrix
    print('Confusion matrix:', file=saveFile)
    print(metrics.confusion_matrix(true, pred), file=saveFile)
    # Overall scores
    print('\n    Accuracy\t', metrics.accuracy_score(true, pred), file=saveFile)
    print(' Cohen Kappa\t', metrics.cohen_kappa_score(true, pred), file=saveFile)
    print('    F1-Score\t', metrics.f1_score(true, pred, average=average), '\tAverage =', average, file=saveFile)
    print('   Precision\t', metrics.precision_score(true, pred, average=average), '\tAverage =', average,
              file=saveFile)
    print('      Recall\t', metrics.recall_score(true, pred, average=average), '\tAverage =', average,
              file=saveFile)
    # Results of each class
    print('\nResults of each class:', file=saveFile)
    print('    F1-Score\t', metrics.f1_score(true, pred, average=None), file=saveFile)
    print('   Precision\t', metrics.precision_score(true, pred, average=None), file=saveFile)
    print('      Recall\t', metrics.recall_score(true, pred, average=None), file=saveFile)
    print('\nACC of all fold:', file=saveFile)
    print(fold_acc, file=saveFile)
    print('\nACC of each fold:', file=saveFile)
    print(np.mean(fold_acc), file=saveFile)
    print(64 * '==', file=saveFile)
    if savePath != None:
        saveFile.close()
    return

def ConfusionMatrix( y_true, y_pred, classes, savePath, title=None, cmap=plt.cm.Blues):
    if not title:
        title = 'Confusion matrix'
        # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    cm_n = cm
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Confusion matrix")
    print(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')
        # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j] * 100, '.2f') + '%\n' + format(cm_n[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    plt.savefig(savePath + title + ".png")
    plt.show()
    return ax