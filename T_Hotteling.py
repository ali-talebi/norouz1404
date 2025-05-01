import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy.stats import beta
from scipy.stats import chi2
import matplotlib

matplotlib.use('TkAgg')  # Use TkAgg backend
# Modified_DATA_GENERATOR/fix2/alpha_1_50
root_data = 'DATA_GENERATED/fix/alpha_1_50'

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

B_mu = [df_health['bias'].mean(), df_health['betha_1'].mean(), df_health['betha_2'].mean(), df_health['betha_3'].mean(),
        df_health['betha_4'].mean(),
        df_health['betha_5'].mean()]

q = len(df_health)  # Number of profiles
k = 6  # Number of model coefficients
alpha = 0.05  # Significance level

df_SB = np.cov(df_health.values, rowvar=False)
df_SB_inverse = np.linalg.inv(df_SB)
a = k / 2
b = (q - k - 1) / 2

UCL_B = (((q - 1) ** 2) / q) * beta.ppf(1 - alpha, a, b)

remove_list = []

Total_T2_B_P_HEALTH = []
for index in range(len(df_health)):
    diff_profile_from_mu = df_health.iloc[index, :] - B_mu
    T2B = diff_profile_from_mu @ df_SB_inverse @ diff_profile_from_mu.T
    if T2B > UCL_B:
        remove_list.append(index)
    Total_T2_B_P_HEALTH.append(T2B)

new_df_health = []
for index in range(len(df_health)):
    if index in remove_list:
        continue
    row_select = df_health.iloc[index, :]
    row_select2 = np.array(row_select).tolist()
    new_df_health.append(row_select2)

new_df_health = np.array(new_df_health)
df_SB_new = np.cov(new_df_health, rowvar=False)
df_SB_inverse_new = np.linalg.inv(df_SB)

q = len(new_df_health)
a = k / 2
b = (q - k - 1) / 2

UCL_B_new = (((q - 1) ** 2) / q) * beta.ppf(1 - alpha, a, b)


Total_T2_B_P_HEALTH_new = []
for index in range(len(new_df_health)):
    diff_profile_from_mu = new_df_health[index] - B_mu
    T2B_new = diff_profile_from_mu @ df_SB_inverse_new @ diff_profile_from_mu.T
    Total_T2_B_P_HEALTH_new.append(T2B_new)

Total_T2_B_P_FALSE = []
for index in range(len(df_fault)):
    diff_profile_from_mu = df_fault.iloc[index] - B_mu
    T2B = diff_profile_from_mu @ df_SB_inverse_new @ diff_profile_from_mu.T
    Total_T2_B_P_FALSE.append(T2B)

_c = 0
for i in Total_T2_B_P_FALSE:
    if i >= UCL_B:
        _c += 1

plt.figure(figsize=(10, 10))
plt.scatter(range(len(Total_T2_B_P_HEALTH)), Total_T2_B_P_HEALTH, label="T2")
plt.hlines(xmin=0, xmax=len(Total_T2_B_P_HEALTH), y=UCL_B, label=f'UCL_B : {UCL_B}', colors='r')
plt.legend()
plt.grid()
plt.title("Phase 1 - T2 For Data Health , Sigma : variance - covariance ")
plt.show()

plt.figure(figsize=(10, 10))
plt.scatter(range(len(Total_T2_B_P_HEALTH_new)), Total_T2_B_P_HEALTH_new, label="T2")
plt.hlines(xmin=0, xmax=len(Total_T2_B_P_HEALTH_new), y=UCL_B_new, label=f'UCL_B : {UCL_B_new}', colors='r')
plt.legend()
plt.grid()
plt.title("Phase 1 - T2 For Data Health , Sigma : variance - covariance ")
plt.xlabel("Edited UCL")
plt.show()

plt.figure(figsize=(10, 10))
plt.scatter(range(len(Total_T2_B_P_FALSE)), Total_T2_B_P_FALSE, label="T2")
plt.hlines(xmin=0, xmax=len(Total_T2_B_P_FALSE), y=UCL_B_new, label=f'UCL_B : {UCL_B_new}', colors='r')
plt.legend()
plt.grid()
plt.title("Phase 2 - T2 For Data Fault Base Sigma : variance - covariance ")
acc = _c / len(Total_T2_B_P_FALSE)
plt.ylabel(f"Accuracy : {acc} ")
plt.xlabel("alpha : [0.001-0.050] , void :[0.1-0.4] ")
plt.show()
