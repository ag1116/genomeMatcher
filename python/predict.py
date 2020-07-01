from Bio import SeqIO
import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from collections import Counter
from sklearn import model_selection, linear_model
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
import sys

filename = sys.argv[1]
mutated_locations = np.genfromtxt('C:\\myvenv\\genome\\mutated.csv',delimiter=',')

sequences = [r for r in SeqIO.parse(filename, 'fasta')]
sequence_num =  0
print(sequences[sequence_num])

mutation_df = pd.DataFrame()
n_bases_in_seq = len(sequences[0])
for location in tqdm.tqdm(mutated_locations): 
  print(location)
  bases_at_location = np.array([s[location] for s in sequences])
  #print(bases_at_location)
  #if len(set(bases_at_location))==1: continue # If
  for base in ['A', 'T', 'G', 'C', '-']:
    feature_values = (bases_at_location==base)
    #print(feature_values)
    
    # Set the values of any base that equals 'N' to np.nan.
    feature_values[bases_at_location=='N'
                   ] = np.nan
    
    # Convert from T/F to 0/1.
    feature_values  = feature_values*1
    #print(feature_values)
    # Make the column name look like <location>_<base> (1_A, 2_G, 3_A, etc.)
    column_name = str(location) + '_' + base
    mutation_df[column_name] = feature_values
    #print(mutation_df)

X = mutation_df
print(X)

Z = X.iloc[[0]]
print(Z)

loaded_model = pickle.load(open('C:\\myvenv\\genome\\finalized_model.sav', 'rb'))
pred = loaded_model.predict(Z)
print(pred)
