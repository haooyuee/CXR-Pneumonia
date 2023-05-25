#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated on 2023-0119$

@author: Jonathan Beaulieu-Emond
"""


import numpy as np
import sklearn.metrics
#import timm.utils.metrics
from sklearn import metrics
from sklearn.metrics import (
    roc_curve,
    auc,
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize

import logging


class Metrics:
    def __init__(self, num_classes, names, threshold, true, pred):
        self.num_classes = num_classes
        self.names = names
        self.true = true
        self.pred = pred

        self.convert = lambda x: x

        #We store the threshold values from a default list or from the parameter threshold
        if len(threshold) != len(self.names):      
            self.thresholds = np.zeros((num_classes)) + 0.5
        else:
            self.thresholds = threshold
       
    def perf_measure(self, true, pred):
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(len(self.names)):
            if true[i]==pred[i]==1:
                TP += 1
            if pred[i]==1 and true[i]!=pred[i]:
                FP += 1
            if true[i]==pred[i]==0:
                TN += 1
            if pred[i]==0 and true[i]!=pred[i]:
                FN += 1

        return(TP, FP, TN, FN)

    ###################################################################
    def accuracy(self):
        #We do not want to change the reference content provided
        pred2 = np.copy(self.pred)
        pred2 = self.convert(pred2)
       
        for idx, thresh in enumerate(self.thresholds):
            pred2[:, idx] = np.where(pred2[:, idx] > thresh, 1, 0)
         
        #accuracy = 0
        #for x, y in zip(self.true, pred2):
        #    if (x == y).all():
        #        accuracy += 1
           
        return accuracy_score(self.true, pred2)
        #return accuracy/pred2.shape[0]

    ###################################################################
    def f1(self):
        #We do not want to change the reference content provided in the input parameters
        pred2 = np.copy(self.pred)
       
        for idx, thresh in enumerate(self.thresholds):
            pred2[:, idx] = np.where(pred2[:, idx] > thresh, 1, 0)

        f1 = f1_score(self.true, pred2, zero_division=0, average=None)
        f1Globale = f1_score(self.true, pred2, average='weighted')
       
        f1_dict = {name: item for name, item in zip(self.names, f1)}
        f1_dict["mean"] = np.mean(f1)
       
        f1_dict["globale"] = f1Globale
       
        return f1_dict

    ###################################################################
    def precision(self):
        #We do not want to change the reference content provided
        pred2 = np.copy(self.pred)
        pred2 = self.convert(pred2)
        #print(pred2)
        #print('####')
        #pred = np.where(pred > 0.5, 1, 0)
        #print(self.thresholds)
        for idx, name in enumerate(self.names):
            pred2[:,idx] = np.where(pred2[:,idx] > self.thresholds[idx], 1, 0)
            #print(pred2[:,idx])
       
        results = precision_score(self.true, pred2, average=None, zero_division=0)
        precisionGlobale = precision_score(self.true, pred2, average='weighted', zero_division=0)

        results_dict = {}
        for item, name in zip(results, self.names):
            results_dict[name] = item
           
        results_dict["globale"] = precisionGlobale
       
        return results_dict

    ###################################################################
    def recall(self):

        #We do not want to change the reference content provided
        pred2 = np.copy(self.pred)
        pred2 = self.convert(pred2)
       
        #pred = np.where(pred > 0.5, 1, 0)

        for idx, thresh in enumerate(self.thresholds):
            pred2[:, idx] = np.where(pred2[:, idx] > thresh, 1, 0)
       
        results = recall_score(self.true, pred2, average=None, zero_division=0)
        recallGlobale = recall_score(self.true, pred2, average='weighted', zero_division=0)
       
        results_dict = {}
        for item, name in zip(results, self.names):
            results_dict[name] = item
           
        results_dict['globale'] = recallGlobale
       
        return results_dict

    ###################################################################
    def computeAUROC(self):

        fpr = dict()
        tpr = dict()
        outAUROC = dict()
        classCount = self.pred.shape[1]
       
         #We do not want to change the reference content provided
        pred2 = np.copy(self.pred)
        pred2 = self.convert(pred2)
 
        for i in range(classCount):

            # fpr[i], tpr[i], thresholds = roc_curve(true[:, i], pred[:, i],pos_label=1)
            #
            # threshold = thresholds[np.argmax(tpr[i] - fpr[i])]
            # logging.info(f"threshold {self.names[i]} : ",threshold)
            # self.thresholds[i] =threshold
            # try :
            #     auroc =  auc(fpr[i], tpr[i])
            # except :
            #     auroc=0
            try:
                #Modif JFS on Feb 14th 2023, before average='macro' instead of multi_class='ovr'
                auroc = roc_auc_score(self.true[:, i], pred2[:, i], multi_class='ovr')
            except ValueError:
                auroc = 0
           
            outAUROC[self.names[i]] = auroc
            if np.isnan(outAUROC[self.names[i]]):
                outAUROC[self.names[i]] = 0

       
        outAUROC["mean"] = np.mean(list(outAUROC.values()))

        return outAUROC

    ###################################################################
    def mmc(self):
       
        #We do not want to change the reference content provided
        pred2 = np.copy(self.pred)
        pred2 = self.convert(pred2)
       
        for idx, thresh in enumerate(self.thresholds):
            pred2[:, idx] = np.where(pred2[:, idx] > thresh, 1, 0)
       
        # Convertir les étiquettes en un format binaire
        #y_true_bin = label_binarize(true, classes=range(self.num_classes))
       
        results = {}
        mmc_mean = 0.0
        for i in range(self.num_classes):
            results[self.names[i]] = matthews_corrcoef(self.true[:, i], pred2[:, i])
            mmc_mean += results[self.names[i]]

        # Calculer le MCC moyen (moyenne macro)
        results['mean'] = mmc_mean/self.num_classes

        return results

    ###################################################################
    def g_mean(self):
       
        #We do not want to change the reference content provided
        pred2 = np.copy(self.pred)
        pred2 = self.convert(pred2)

        for idx, thresh in enumerate(self.thresholds):
            pred2[:, idx] = np.where(pred2[:, idx] > thresh, 1, 0)
       
        g_mean={}

        #Initialize the g_mean
        for i in range(self.num_classes):
            g_mean[self.names[i]] = 0.0
       
        #Check if we can calculate gmean    
        if self.true.shape != pred2.shape or self.true.shape[0] == pred2.shape[0] == 1:
            return g_mean
 
        for i in range(self.num_classes):
            tn, fp, fn, tp = confusion_matrix(self.true[:, i], pred2[:, i]).ravel()
            specificity = tn / (tn + fp)
            sensitivity = tp / (tp + fn)
            g_mean[self.names[i]] = np.sqrt(specificity * sensitivity)

        return g_mean
   
    ###################################################################
    def metrics(self):
        dict = {
            "auc":      self.computeAUROC,
            "f1":       self.f1,
            "recall":   self.recall,
            "MMC" :     self.mmc, #TODO : Fix MMC for multi-label
            "precision":self.precision,
            "accuracy": self.accuracy,
            "gmean":    self.g_mean,
        }
        return dict

   
###################################################################



if __name__ == "__main__":
    #from radia import names
    names = ['a','b','c','d','e']
    num_classes = len(names)
    np.random.seed(0)
    label = np.random.randint(0, 2, (10, num_classes))
    pred = np.random.random(size=(10, num_classes))
    metric = Metrics(
        num_classes=num_classes, names=names, threshold=np.zeros((num_classes)) + 0.5, true=label, pred=pred
    )
    #print(label)
    #print(pred)
    res = {}
    #print(label)
    #print('#################')
    #print(pred)
    for key, item in metric.metrics().items():
        print(key)
        res[key] = item()
        
    print(res)