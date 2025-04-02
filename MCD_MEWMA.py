import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.covariance import MinCovDet
from scipy.stats import chi2



Total_Df = pd.DataFrame()

root_data = 'DATA_GENERATED/fix/2/alpha_1_50'

columns_ = ['bias', 'betha_1', 'betha_2', 'betha_3', 'betha_4', 'betha_5', 'R2',
            'locate', 'class', 'alpha', 'void', 'SSE', 'R2_adj', 'stat_error',
            'p_value_error', 'DW' , 'var_errors', 'SST', 'MSR', 'MSE', 'F', 'p_value_f']



Total_Df = pd.DataFrame(columns=columns_)
for index, name in enumerate(os.listdir(root_data)):
    df = pd.read_csv(f'{root_data}/{name}')
    Total_Df = pd.concat([Total_Df, df], axis=0)
    del df


df_health = Total_Df[Total_Df['class'] == 0.0 ][['bias', 'betha_1', 'betha_2', 'betha_3', 'betha_4', 'betha_5']]
df_fault  = Total_Df[Total_Df['class'] == 1.0 ][['bias', 'betha_1', 'betha_2', 'betha_3', 'betha_4', 'betha_5']]
# اجرای MCD روی داده‌های ضرایب
mcd = MinCovDet().fit(df_health)

# استخراج میانگین مقاوم و کوواریانس مقاوم
mu_robust = mcd.location_  # میانگین مقاوم
cov_robust = mcd.covariance_  # ماتریس کوواریانس مقاوم

print("Robust Mean (MCD):", mu_robust)
print("Robust Covariance Matrix (MCD):\n", cov_robust)

# pd.DataFrame(cov_robust).to_csv('covariance_robust.csv' , index = False )

XTX_INVERSE_SIGMA_2_inverse = np.linalg.inv(cov_robust)


root2 = 'DATA_GENERATED/fix/3/alpha_300_350'
Total_Df2 = pd.DataFrame(columns=columns_)
for index, name in enumerate(os.listdir(root2)):
    df2 = pd.read_csv(f'{root2}/{name}')
    Total_Df2 = pd.concat([Total_Df2, df2], axis=0)
    del df2

df_fault2  = Total_Df2[Total_Df2['class'] == 0.0 ][['bias', 'betha_1', 'betha_2', 'betha_3', 'betha_4', 'betha_5']]


# MEWMA parameters
lambda_ = 0.2  # Exponential weight
Z_t = np.zeros(6)  # Initialize Z_t
T2_mewma = []  # Store T^2 statistics

for t in range(len(df_fault2)):
    X_t = df_fault2.iloc[t].values  # Current profile coefficients
    Z_t = lambda_ * (X_t - mu_robust) + (1 - lambda_) * Z_t  # Compute Z_t
    T2_t = Z_t @ XTX_INVERSE_SIGMA_2_inverse @ Z_t.T  # Compute T^2 statistic
    T2_mewma.append(T2_t)


alpha = 0.05
UCL = (lambda_ / (2 - lambda_)) * chi2.ppf(1 - alpha, df=6)
print("UCL : " , UCL )
plt.figure(figsize=(10, 5))
plt.plot(T2_mewma, marker='o', linestyle='-', label='MEWMA T² Statistic')
plt.hlines(xmin=0 , xmax=len(T2_mewma) , y = UCL, color='r', linestyle='-', label=f'UCL = {UCL:.2f}')
plt.xlabel('Profile Index')
plt.ylabel('T² Statistic')
plt.title('Robust MEWMA Control Chart Phase 2 On Data Fault With Alpha : [0.300-0.332] and Lambda = 0.2 ')
plt.legend()
plt.grid()
plt.show()