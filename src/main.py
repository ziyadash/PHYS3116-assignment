# TODO DEVELOP A DETAILED METHODOLOGY FOR PROCESSING THE DATA.. NEED TO DO LOTS OF RESEARCH AND LEARN SOME THEORY
# - write a brief outline of each dataset --> how its relevant to identifying globular clusters
# - design an analysis method for the datasets, including a set of criteria for deciding whether a globular cluster is likely to have been accreted
# - come up with a draft structure for our code --> modules for (1) parsing raw data, (2) doing analysis on this data, and (3) displaying results

import matplotlib as plt
import numpy as np
import pandas as pd


# import data
data_filepaths = {
    'HarrisPartI': '../data/HarrisPartI.csv',
    'HarrisPartIII': '../data/HarrisPartIII.csv',
    'Krause21': '../data/Krause21.csv',
    'vandenBerg': '../data/vandenBerg_table2.csv'
}

HarrisParti_df = pd.read_csv(data_filepaths['HarrisPartI'])
HarrisPartiii_df = pd.read_csv(data_filepaths['HarrisPartIII'])
Krause21_df = pd.read_csv(data_filepaths['Krause21'])
Vandenberg_df = pd.read_csv(data_filepaths['vandenBerg'])

# print(HarrisParti_df.head(3))
# print(HarrisPartiii_df.head(3))
# print(Krause21_df.head(3))
print(Vandenberg_df.head(3))