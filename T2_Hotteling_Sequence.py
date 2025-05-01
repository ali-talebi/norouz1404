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

a = k / 2
f = (2 * (q - 1) ** 2) / (3 * q - 4)
b = (f - k - 1) / 2

UCL_H = (((q - 1) ** 2) / q) * beta.ppf(1 - alpha, a, b)

Total_v = []

for index in range(len(df_health) - 1):
    V = df_health.iloc[index + 1] - df_health.iloc[index]
    Total_v.append(V)

Total_v = np.array(Total_v)
S_matrix = Total_v.T @ Total_v
S_matrix /= (2 * (q - 1))
s_reverse = np.linalg.inv(S_matrix)

remove_list = []

Total_T2_H_HEALTH = []
for index in range(len(df_health)):
    diff_profile_from_mu = df_health.iloc[index, :] - B_mu
    T2H = diff_profile_from_mu @ s_reverse @ diff_profile_from_mu.T
    if T2H > UCL_H:
        remove_list.append(T2H)
    Total_T2_H_HEALTH.append(T2H)

new_df_health = []
for index in range(len(df_health)):
    if index in remove_list:
        continue
    row_select = df_health.iloc[index, :]
    row_select2 = np.array(row_select).tolist()
    new_df_health.append(row_select2)

new_df_health = np.array(new_df_health)

Total_v_new = []
for index in range(len(new_df_health) - 1):
    V = new_df_health[index + 1] - new_df_health[index]
    Total_v_new.append(V)

q = len(Total_v_new)
Total_v_new = np.array(Total_v_new)
S_matrix_new = Total_v_new.T @ Total_v_new
S_matrix_new /= (2 * (q - 1))
s_reverse_new = np.linalg.inv(S_matrix_new)


Total_T2_H_HEALTH_new = []
for index in range(len(Total_v_new)):
    diff_profile_from_mu = Total_v_new[index] - B_mu
    T2H_new = diff_profile_from_mu @ s_reverse_new @ diff_profile_from_mu.T
    Total_T2_H_HEALTH_new.append(T2H_new)

Total_T2_H_FAULT = []
for index in range(len(df_fault)):
    diff_profile_from_mu = df_fault.iloc[index, :] - B_mu
    T2H = diff_profile_from_mu @ s_reverse_new @ diff_profile_from_mu.T
    Total_T2_H_FAULT.append(T2H)


q = len(Total_T2_H_HEALTH_new)  # Number of profiles
k = 6  # Number of model coefficients
alpha = 0.05  # Significance level

a = k / 2
f = (2 * (q - 1) ** 2) / (3 * q - 4)
b = (f - k - 1) / 2

UCL_H_EDIT = (((q - 1) ** 2) / q) * beta.ppf(1 - alpha, a, b)


_c = 0
for i in Total_T2_H_FAULT:
    if i >= UCL_H:
        _c += 1

plt.figure(figsize=(10, 10))
plt.scatter(range(len(Total_T2_H_HEALTH)), Total_T2_H_HEALTH, label="T2")
plt.hlines(xmin=0, xmax=len(Total_T2_H_HEALTH), y=UCL_H, label=f'UCL_B : {UCL_H}', colors='r')
plt.legend()
plt.grid()
plt.title("Phase 1 - T2 For Data Health - Sigma With V(i+1) - V(i)")
plt.show()

plt.figure(figsize=(10, 10))
plt.scatter(range(len(Total_T2_H_HEALTH_new)), Total_T2_H_HEALTH_new, label="T2")
plt.hlines(xmin=0, xmax=len(Total_T2_H_HEALTH_new), y=UCL_H_EDIT, label=f'UCL_B : {UCL_H_EDIT}', colors='r')
plt.legend()
plt.grid()
plt.title("Phase 1 - T2 For Data Health")
plt.show()

plt.figure(figsize=(10, 10))
plt.scatter(range(len(Total_T2_H_FAULT)), Total_T2_H_FAULT, label="T2")
plt.hlines(xmin=0, xmax=len(Total_T2_H_FAULT), y=UCL_H_EDIT, label=f'UCL_B : {UCL_H_EDIT}', colors='r')
plt.legend()
plt.grid()
plt.title("Phase 2 - Base Data Fault - Sigma  Calculate = V(i+1) - V(i)")
acc = _c / len(Total_T2_H_FAULT)
plt.ylabel(f"Accuracy : {acc} ")
plt.xlabel("alpha : [0.001-0.050] , void :[0.1-0.4] ")
plt.show()
