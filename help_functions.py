import pandas as pd
import numpy as np
import datetime, sys, pickle
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D,BatchNormalization, Dropout
from keras import optimizers


def preprocess_notNN(dat):

  dat_array = np.array(list(dat))

  labels = np.array(dat_array[0][1]).astype('float32')
  dataset = dat_array[0][0]

  ind_rem = []
  sequences_list = []
  for i,seq in enumerate(dataset):
    if 'N' not in str(seq):
      sequences_list.append(list(str(seq))[2:-1])
    else:
      ind_rem.append(i)  

  channels = {'A' : 0,'T' : 1,'C' : 2,'G' : 3}
  
  return np.array(pd.DataFrame(sequences_list).replace(channels)), np.delete(labels,ind_rem)




def preprocess_NN(dat):

  dat_array = np.array(list(dat))

  labels = np.array(dat_array[0][1]).astype('float32')
  dataset = dat_array[0][0]

  ind_rem = []
  sequences_list = []
  for i,seq in enumerate(dataset):
    if not 'N' in str(seq):
      sequences_list.append(list(str(seq))[2:-1])
    else:
      ind_rem.append(i)


  samples_size = len(sequences_list)
  sequence_size = min([len(x) for x in sequences_list])
  ohe = np.zeros((samples_size, sequence_size, 4))
  channels = {'A' : 0,'T' : 1,'C' : 2,'G' : 3}

  for index, sequence in enumerate(sequences_list):
    for pos, nucleotide in enumerate(sequence):
        ohe[index, pos, channels[nucleotide]] = 1
  
  return ohe, np.delete(labels,[ind_rem]), sequence_size


