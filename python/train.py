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
import pickle 


sequences = [r for r in SeqIO.parse("C:\\myvenv\\genome\\trainingsequences.fasta", 'fasta')]
mutation_df = pd.DataFrame()
n_bases_in_seq = len(sequences[0])

for location in (range(n_bases_in_seq)):
  bases_at_location = np.array([s[location] for s in sequences])

  skipped_locations = np.array([])
  mutated_locations = np.array([])

  if len(set(bases_at_location))==1:
    skipped_locations = np.append(skipped_locations, location)
    continue
  
  mutated_locations = np.append(mutated_locations, location)
 
  for base in ['A', 'T', 'G', 'C', '-']:
    feature_values = (bases_at_location==base)
    feature_values[bases_at_location=='N'
                   ] = np.nan
    feature_values  = feature_values*1
    column_name = str(location) + '_' + base
    mutation_df[column_name] = feature_values
countries = [(s.description).split('|')[-1] for s in sequences]
countries_to_regions_dict = {
         'Australia': 'Oceania',
         'China': 'Asia',
         'Hong Kong': 'Asia',
         'India': 'Asia',
         'Nepal': 'Asia',
         'South Korea': 'Asia',
         'Sri Lanka': 'Asia',
         'Taiwan': 'Asia',
         'Thailand': 'Asia',
         'USA': 'North America',
         'Viet Nam': 'Asia'
}
regions = [countries_to_regions_dict[c] if c in 
           countries_to_regions_dict else 'NA' for c in countries]
mutation_df['label'] = regions

X = mutation_df.drop('label', 1)
Y = mutation_df.label

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
    X, Y, train_size=.8, random_state=42)

lam = 0.8
l1m = linear_model.LogisticRegression(
    multi_class="multinomial", max_iter=1000,
    fit_intercept=False, tol=0.001, C=1/lam,
    penalty='l1', solver='saga', random_state=42)
l1m.fit(X_train, Y_train)

filename = 'C:\\myvenv\\genome\\finalized_model.sav'
pickle.dump(l1m, open(filename, 'wb'))
print(filename)

np.savetxt("skipped.csv", skipped_locations, delimiter=",")
np.savetxt("mutated.csv", mutated_locations, delimiter=",")
