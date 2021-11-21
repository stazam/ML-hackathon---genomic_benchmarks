from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPooling1D, BatchNormalization, Conv1D
from keras.layers.experimental.preprocessing import Rescaling
from keras import optimizers
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
############################
# This i can create an empty file in colab
#   %%writefile functions.py
#   def function(x):
#       return(x)
   
def complement(x):
  """
  return a complement of sequence
  """
  if x == 'A':
    return('T')
  elif x == 'T':
    return('A') 
  elif x == 'C':
    return('G')
  return('C')   


def reverse_complement(x):
  """
  return a reverse complement of sequence
  """
  x.reverse()
  return(list(map(complement, x)))  


def parse_object(url):
  """
  return a tuple of two lists: parsed sequence and strand_sign 
  """
  content = []
  with open(url) as f:
      for line in f:
          content.append(line.strip().split())

  sequences = []
  strand_sign = []

  for i in range(len(content)-1):

    if i % 2 == 0:
      strand_sign.append(content[i][0][len(content[i][0])-2]) 
    else:
      sequences.append(list(content[i][0])) 

  return (sequences, strand_sign) 


def correction_parsing(strand, sign):
  """
  1. reverse negative strands to positive
  2. if sequence contains N, remove her from the list
  """

  for i,seq in enumerate(strand):
      if sign[i] == '-':
          strand[i] = reverse_complement(seq)

  strand = [seq for seq in strand if 'N' not in seq]        

  return strand 


def plot_loss(model):

  plt.figure(1)
  plt.plot(model.history['loss'], label='loss')
  plt.plot(model.history['val_loss'], label='val_loss')
  plt.ylim([0, 1])
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()
  plt.grid(True)

  plt.figure(2)
  plt.plot(model.history['accuracy'], label='accuracy')
  plt.plot(model.history['val_accuracy'], label='val_accuracy')
  plt.ylim([0, 2])
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.grid(True)  


def build_model_complex(sequence_size = 200):
  """
  return model building with complex architecture
  """

  model = Sequential()

  model.add(Conv1D(
        filters=16,
        kernel_size=8,
        padding='same',
        data_format="channels_last",
        activation='relu',
        input_shape=(sequence_size, 4)))

  model.add(BatchNormalization())
  model.add(MaxPooling1D())
  model.add(Dropout(0.3))

  model.add(Conv1D(
        filters=8,
        kernel_size=8,
        padding='same',
        activation='relu'))

  model.add(BatchNormalization())
  model.add(MaxPooling1D())
  model.add(Dropout(0.3))

  model.add(Conv1D(
        filters=4,
        kernel_size=8,
        padding='same',
        activation='relu'))

  model.add(BatchNormalization())
  model.add(MaxPooling1D())
  model.add(Dropout(0.3))

  model.add(Conv1D(
        filters=3,
        kernel_size=8,
        padding='same',
        activation='relu'))

  model.add(BatchNormalization())
  model.add(MaxPooling1D())
  model.add(Dropout(0.3))

  model.add(Flatten())

  model.add(Dense(512, activation="relu"))
  model.add(BatchNormalization())
  model.add(Dropout(0.3))

  model.add(Dense(1, activation='sigmoid'))

  model.summary()

  return model


def build_model_simple(sequence_size = 200):
  """
  return model building with simple architecture
  """
  
  model = Sequential([
      Conv1D(filters = 32, kernel_size=8, padding='same', activation = 'relu', input_shape=(sequence_size, 4)),
      Dropout(0.5),
      Flatten(),
      Dense(128, activation='relu'),
      Dense(128, activation='relu'),
      Dense(64, activation='relu'),
      Dropout(0.5),
      Dense(1, activation='sigmoid')
  ])

  model.summary()

  return model


def one_hot(seq_file):
  """
  return one hot ecnoded .np object as an input into model
  """

  samples_size = len(seq_file) 
  sequence_size = len(seq_file[0])
  ohe_dataset = np.zeros((samples_size, sequence_size, 4))
  channels = {'A' : 0,'T' : 1,'C' : 2,'G' : 3}

  for index, sequence in enumerate(seq_file):
     for pos, nucleotide in enumerate(sequence):
            ohe_dataset[index, pos, channels[nucleotide]] = 1

  return ohe_dataset  


def create_labels(len_pos, len_neg):
  """
  return labels as .np object as type float32
  """

  return np.array([0] * len_pos  + [1] * len_neg).astype('float32')


def create_rseq(sequence, sequence1, length, ratio):
  """
  return list object as input to one_hot function and labels as .np object:
  1. length is parameter which gives the length of reference sequence
  2. ratio should be in the form a:b
  """
  r = [ int(s)  for s in ratio.split(":")]
  (a,b) = (length,round(r[0] / r[1] * length))

  seq = sequence[0:a] + sequence1[0:b]

  print("The length of returned sequence is %d and the lengths of positive/negative sets  are %d,%d" %(len(seq),a,b))

  return (seq, np.array([0] * a  + [1] * b).astype('float32'))


def parse_final(url1, url2, length_set, length_ev, ratio):
  """
  1. parse objects
  2. correct parsing
  3. create numpy sequence
  4. create evaluate sequecne
  5. one hot encoding

  RETURN: one hot encoded sequences, and also labels
  """

  sequence, strand_sign = parse_object(url1)
  sequence1, strand_sign1 = parse_object(url2)

  sequence = correction_parsing(sequence, strand_sign)
  sequence1 = correction_parsing(sequence1, strand_sign1)

  seq_final,labels = create_rseq(sequence, sequence1, length_set,ratio)


  seq_final_ev = sequence[length_set + 1: length_set + length_ev + 1 ] + sequence1[length_set + 1: length_set + length_ev + 1 ]
  labels_ev =  np.array([0] * length_ev  + [1] * length_ev).astype('float32')

  ohe_dataset_ev = one_hot(seq_final_ev)
  ohe_dataset = one_hot(seq_final)

  return (ohe_dataset, labels, ohe_dataset_ev, labels_ev)
  