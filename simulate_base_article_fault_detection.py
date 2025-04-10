import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy.stats import beta
from scipy.stats import chi2

root_data = 'DATA_GENERATED/fix/2/alpha_1_50'

columns_ = ['bias', 'betha_1', 'betha_2', 'betha_3', 'betha_4', 'betha_5', 'R2',
            'locate', 'class', 'alpha', 'void', 'SSE', 'R2_adj', 'stat_error',
            'p_value_error', 'DW', 'var_errors', 'SST', 'MSR', 'MSE', 'F', 'p_value_f']

Total_Df = pd.DataFrame(columns=columns_)
for index, name in enumerate(os.listdir(root_data)):
    df = pd.read_csv(f'{root_data}/{name}')
    Total_Df = pd.concat([Total_Df, df], axis=0)
    del df

columns2 = ['bias', 'betha_1', 'betha_2', 'betha_3', 'betha_4', 'betha_5']

df_health = Total_Df[Total_Df['class'] == 0.0][columns2]
df_fault = Total_Df[Total_Df['class'] == 1.0][columns2]

print("df_health shape : ", df_health.shape)

Total_v = []

for index in range(len(df_health) - 1):
    V = df_health.iloc[index + 1] - df_health.iloc[index]
    Total_v.append(V)

Total_v = np.array(Total_v)
print("Shape :: ", Total_v.shape)

S_matrix = Total_v.T @ Total_v
print("Smatrix Befoe : \n", S_matrix)
S_matrix /= (2 * len(Total_v))
print("Smatrix after : \n", S_matrix)
print("S matrix : \n", S_matrix.shape)

s_reverse = np.linalg.inv(S_matrix)

B_mu = [df_health['bias'].mean(), df_health['betha_1'].mean(), df_health['betha_2'].mean(), df_fault['betha_3'].mean(), df_fault['betha_4'].mean(),
        df_fault['betha_5'].mean()]

B_mu = np.array(B_mu)

print("B_mu shape : " , B_mu )

Total_T2_H_HEALTH = []
for index in range(len(df_health)) :
    diff_profile_from_mu = df_health.iloc[index , : ] - B_mu
    T2H = diff_profile_from_mu @ s_reverse @ diff_profile_from_mu.T
    Total_T2_H_HEALTH.append(T2H)



q = len(df_health)  # Number of profiles
k = 6  # Number of model coefficients
alpha = 0.05  # Significance level

# Calculate the Beta distribution parameters
a = k / 2
f = (2 * (q-1) ** 2 ) / (3*q-4)
b = (f - k - 1) / 2


# Calculate the critical value from the Beta distribution
beta_critical = beta.ppf(1 - alpha, a, b)

# Calculate UCL_B using Equation (6)
UCL_B = ((q - 1)**2 / q) * beta_critical
UCL_C = chi2.ppf(1 - alpha, k)


plt.figure(figsize=(10,10))
plt.scatter(range(len(Total_T2_H_HEALTH)) , Total_T2_H_HEALTH  , label="T2")
plt.hlines(xmin=0 , xmax= len(Total_T2_H_HEALTH) , y =UCL_B , label=f'UCL_b : {UCL_B}' , colors= 'r' )
plt.hlines(xmin=0 , xmax= len(Total_T2_H_HEALTH) , y =UCL_C , label=f'UCL_c : {UCL_C}' , colors ='purple')
plt.legend()
plt.grid()
plt.title("T2")
plt.show()









