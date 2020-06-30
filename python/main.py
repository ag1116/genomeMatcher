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

filename = ""

originals = [] # Setup an empty list
for record in SeqIO.parse("globalsequences.fasta", 'fasta'):
        originals.append(record)
        
input_genome = SeqIO.read("input.fasta.txt", "fasta")
print(input_genome)
originals.append(input_genome)

SeqIO.write(originals,'SARS_CoV_2_sequences_global.fasta', 'fasta')

sequences = [r for r in SeqIO.parse("SARS_CoV_2_sequences_global.fasta", 'fasta')]
sequence_num =  100
print(sequences[sequence_num])

n_sequences = len(sequences)
print("There are %f sequences" % n_sequences)

mutation_df = pd.DataFrame()
n_bases_in_seq = len(sequences[0])
for location in tqdm.tqdm(range(n_bases_in_seq)):
  bases_at_location = np.array([s[location] for s in sequences])
  if len(set(bases_at_location))==1: continue # If
  for base in ['A', 'T', 'G', 'C', '-']:
    feature_values = (bases_at_location==base)
    feature_values[bases_at_location== "N" 
                   ] = np.nan
    feature_values  = feature_values*1
    column_name = str(location) + '_' + base
    mutation_df[column_name] = feature_values

country = "USA"
#['China', 'Kazakhstan', 'India', 'Sri Lanka', 'Taiwan', 'Hong Kong', 'Viet Nam', 'Thailand', 'Nepal', 'Israel', 'South Korea', 'Iran', 'Pakistan', 'Turkey', 'Australia', 'USA']
countries = [(s.description).split('|')[-1] for s in sequences]
print("There are %i sequences from %s." %
     (Counter(countries)[country], country))

countries_to_regions_dict = {
    'Australia': 'Oceania',
    'China': 'East Asia',
    'Hong Kong': 'East Asia' ,
    'India': 'South Asia' ,
    'Nepal': 'South Asia' ,
    'South Korea': 'East Asia' ,
    'Sri Lanka': 'South Asia' ,
    'Taiwan': 'East Asia' ,
    'Thailand': 'South Asia' ,
    'USA': 'North America' ,
    'Viet Nam': 'South Asia',
    'Israel': 'Middle East',
    'Turkey': 'Middle East',
    'Iran': 'Middle East',
    'Pakistan': 'South Asia',

}

regions = [countries_to_regions_dict[c] if c in 
           countries_to_regions_dict else 'NA' for c in countries]
mutation_df['label'] = regions

region = "Middle East"
print("There are %i sequences from %s." %
      (Counter(regions)[region], region))

X = mutation_df.drop('label', 1)
Y = mutation_df.label

Z = X.iloc[[-1]]
print(Z)

lm = linear_model.LogisticRegression(
    multi_class="multinomial", max_iter=1000,
    fit_intercept=False, tol=0.001, solver='saga', random_state=42)

# Split into training/testing set. Use a training size of .8
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y)

# Train/fit model.
lm.fit(X_train, Y_train)

pred = lm.predict(Z)
print (pred)
