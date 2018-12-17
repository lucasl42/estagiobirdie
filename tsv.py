import pandas as pd
import numpy as np

#primeira classificação para o conjunto de teste
data = pd.read_table('data_estag_ds.tsv',sep='\t')
data['CLASS'] = np.where(data['TITLE'].str.contains("smartphone", case=False, regex=False), 'smartphone', 'não-smartphone')

data.to_csv('data_classified.tsv',sep='\t')
