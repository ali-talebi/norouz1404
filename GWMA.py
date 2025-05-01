import pandas as pd
import numpy as np
import os
from scipy.stats import norm, chi2
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend
import matplotlib.pyplot as plt

Total_Df = pd.DataFrame()
root_data = 'Modified_DATA_GENERATOR/fix/alpha_1_50'

columns_ = ['bias', 'betha_1', 'betha_2', 'betha_3', 'betha_4', 'betha_5', 'R2',
            'locate', 'class', 'alpha', 'void', 'SSE', 'R2_adj', 'stat_error',
            'p_value_error', 'DW', 'var_errors', 'SST', 'MSR', 'MSE', 'F', 'p_value_f']

Total_Df = pd.DataFrame(columns=columns_)
for index, name in enumerate(os.listdir(root_data)):
    df = pd.read_csv(f'{root_data}/{name}')
    Total_Df = pd.concat([Total_Df, df], axis=0)
    del df

df_health = Total_Df[Total_Df['class'] == 0.0]
df_fault = Total_Df[Total_Df['class'] == 1.0]

df_health['sigma_2'] = df_health['SSE'] / (48 - 6 - 1)
df_fault['sigma_2']  = df_fault['SSE'] / (48 - 6 - 1)

sigma_overall = ((df_health['sigma_2'] ** 0.5).sum()) / len(df_health)

# df_health = Total_Df[Total_Df['class'] == 0.0 ][['bias', 'betha_1', 'betha_2', 'betha_3', 'betha_4', 'betha_5']]
# df_fault  = Total_Df[Total_Df['class'] == 1.0 ][['bias', 'betha_1', 'betha_2', 'betha_3', 'betha_4', 'betha_5']]


target_mu = [df_health['bias'].mean().ravel().tolist()[0],
             df_health['betha_1'].mean().ravel().tolist()[0],
             df_health['betha_2'].mean().ravel().tolist()[0],
             df_health['betha_3'].mean().ravel().tolist()[0],
             df_health['betha_4'].mean().ravel().tolist()[0],
             df_health['betha_5'].mean().ravel().tolist()[0]]

V_Betha = pd.DataFrame(columns=['v_bias', 'v_betha_1', 'v_betha_2', 'v_betha_3', 'v_betha_4', 'v_betha_5'])

df_check = df_health

V_Betha['v_bias'] = (df_check['bias'] - target_mu[0]) / sigma_overall
V_Betha['v_betha_1'] = (df_check['betha_1'] - target_mu[1]) / sigma_overall
V_Betha['v_betha_2'] = (df_check['betha_2'] - target_mu[2]) / sigma_overall
V_Betha['v_betha_3'] = (df_check['betha_3'] - target_mu[3]) / sigma_overall
V_Betha['v_betha_4'] = (df_check['betha_4'] - target_mu[4]) / sigma_overall
V_Betha['v_betha_5'] = (df_check['betha_5'] - target_mu[5]) / sigma_overall

V_sigma = []
for i in range(len(df_check)):
    x_2_j = (df_check.iloc[i, 22] * (48 - 6)) / sigma_overall
    # print("x_2_j : " , x_2_j )
    v_j_sigma = norm.ppf(chi2.cdf(x_2_j, df=48 - 6))
    # print("v_j_sigma : " , v_j_sigma )
    V_sigma.append(v_j_sigma)

V_sigma = np.array(V_sigma).reshape(len(V_sigma), 1)
V_Total = np.concatenate((V_Betha.values, V_sigma), axis=1)

W_list = []
T2_list = []

p = 5  # تعداد ضرایب (غیر از SSE)
sigma2_0 = 1  # واریانس در حالت in-control
W0 = np.zeros(p + 2)

print("W0 : ", W0)

q = 0.9
alpha = 0.7
X_T_X_inv = pd.read_csv('XTX_inverse.csv').values
# print("X_T_X_inv shape: \n", X_T_X_inv)
historical_zero = np.array([0, 0, 0, 0, 0, 0]).reshape(1, 6)
Sigma = np.concatenate([X_T_X_inv, historical_zero], axis=0)

vertical_zero = np.array([0, 0, 0, 0, 0, 0, 0]).reshape(7, 1)
Sigma = np.concatenate([Sigma, vertical_zero], axis=1)
Sigma[6, 6] = 1
#
pd.DataFrame(Sigma).to_csv('RT.csv' , index = False )
# print(Sigma)
# print("Sigma shape : " , Sigma.shape )

Q_i = 0
for profile in range(1 , len(V_Total) + 1 ) :
    Q_i += (q ** ((i - 1) * alpha) - q ** ((i) * alpha ) ) ** 2



print("Q i :  " , Q_i )

for profile in range(len(V_Total)):
    _s = 0
    for i in range(1, profile + 1):
        _s += (q ** ((i - 1) * alpha) - q ** (i * alpha)) * V_Total[profile - i + 1, :]
    Wj = _s + q ** (profile * alpha) * W0
    T2_j = Wj.T @ np.linalg.inv(Q_i * Sigma) @ Wj
    T2_list.append(T2_j)

print("T2_list : ", T2_list)
plt.scatter(range(len(T2_list)) , T2_list , label = "T2" )
plt.legend()
plt.grid()
plt.title("Fault Data - Alpha:[0.001-0.050],Void:[0.1-0.4]")
# plt.title("Data Fault")
plt.show()

