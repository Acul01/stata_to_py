import pandas as pd
import numpy as np
from  pydynpd import regression
from custom_functions import descriptive_statistics, xtbalance

# Load data set
df = pd.read_csv('paper_data.csv')

# Simple data manipulation equivalent to stata code
df['lnfin'] = np.log(df['finance'])
df['lndr'] = df['patent']
df['geo'] = df['geo'] / 1000
df['patent'] = df['patent'].fillna(0)

# Set time period
df = df[(df['year'] >= 1996) & (df['year'] <= 2018)]

# Set id variable
id_var = 'country'

# Balance data using custom xtbalance python equilvalent
df_balanced = xtbalance(df, id_var)

# Print descriptive statistics
print(f'{descriptive_statistics(df_balanced)}\n\n')

# create newvar as sum of con_total per year
df['newvar'] = df_balanced.groupby('year')['conf_total'].transform('sum')


# pydynpd - gmm model as equivalent for xtbalance
command_str_m2 ='lnfinance L1.lnfinance lntotal lngdpp | gmm(lnfinance, 3:10) iv(lntotal lngdpp)| nolevel'
m2 = regression.abond(command_str_m2, df, [id_var, 'year'])
