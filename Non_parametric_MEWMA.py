import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.stats import entropy
import os



root_data = 'DATA_GENERATED/fix/2/alpha_1_50'



columns_ = ['bias', 'betha_1', 'betha_2', 'betha_3', 'betha_4', 'betha_5', 'R2',
            'locate', 'class', 'alpha', 'void', 'SSE', 'R2_adj', 'stat_error',
            'p_value_error', 'DW' , 'var_errors', 'SST', 'MSR', 'MSE', 'F', 'p_value_f']


Total_Df = pd.DataFrame()
Total_Df = pd.DataFrame(columns=columns_)
for index, name in enumerate(os.listdir(root_data)):
    df = pd.read_csv(f'{root_data}/{name}')
    Total_Df = pd.concat([Total_Df, df], axis=0)
    del df




B_fazI = Total_Df[Total_Df['class'] == 0.0 ].values
B_fazII = Total_Df[Total_Df['class'] == 0.0 ].values



kde_fazI = KernelDensity(kernel='gaussian', bandwidth=0.3).fit(B_fazI)
kde_fazII = KernelDensity(kernel='gaussian', bandwidth=0.3).fit(B_fazII)

sample_points = kde_fazI.sample(100)
p_fazI = np.exp(kde_fazI.score_samples(sample_points))
q_fazII = np.exp(kde_fazII.score_samples(sample_points))

kl_div = entropy(p_fazI, q_fazII)
print("KL Divergence:", kl_div)
