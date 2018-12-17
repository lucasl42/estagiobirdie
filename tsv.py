import pandas as pd
import numpy as np


data = pd.read_table('data_estag_ds.tsv',sep='\t')
#print(data.loc[16, 'TITLE'])
data['CLASS'] = np.where(data['TITLE'].str.contains("smartphone", case=False, regex=False), 'smartphone', 'não-smartphone')

#data.head(20)
data.to_csv('data_classified.tsv',sep='\t')