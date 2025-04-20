import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.covariance import MinCovDet



Total_Df = pd.DataFrame()

root_data = 'DATA_GENERATED/fix/alpha_1_50'

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


root2 = 'DATA_GENERATED/fix/alpha_1_50'
Total_Df2 = pd.DataFrame(columns=columns_)
for index, name in enumerate(os.listdir(root2)):
    df2 = pd.read_csv(f'{root2}/{name}')
    Total_Df2 = pd.concat([Total_Df2, df2], axis=0)
    del df2

df_fault2  = Total_Df2[Total_Df2['class'] == 1.0 ][['bias', 'betha_1', 'betha_2', 'betha_3', 'betha_4', 'betha_5']]


S_0_positive = 0
S_0_negetive = 0
data_phase_1 = df_health
check_data   = df_fault2
for i in range(1 , 5 ) :
    shifts = np.array([(i/20) * data_phase_1['bias'].std() , (0/10) * data_phase_1['betha_1'].std() , 0 * data_phase_1['betha_2'].std() , 0 * data_phase_1['betha_3'].std() , 0* data_phase_1['betha_4'].std() , (i/10)* data_phase_1['betha_5'].std() ])
    SHIFT_COVARIANCE_VARIANCE = shifts @ XTX_INVERSE_SIGMA_2_inverse
    D      = SHIFT_COVARIANCE_VARIANCE @ shifts.T
    print("D : " , D )
    a_T = SHIFT_COVARIANCE_VARIANCE / (D**0.5)
    a_T = np.array(a_T).reshape(1,6)
    # print("a_T : " , a_T.shape )
    # print(a_T)
    s_s_positive = []
    s_s_negetive = []
    for j in range(len(check_data)) :
        zj_mines_mu = check_data.iloc[j, :] - mcd.location_
        zj_mines_mu2 = np.array(zj_mines_mu)
        zj_mines_mu2 = zj_mines_mu2.reshape(6,1)
        a_T_zj_mines_mu = a_T @ zj_mines_mu2
        # S = S_0 + a_T_zj_mines_mu - 0.5 * D
        s_p = S_0_positive + a_T_zj_mines_mu - 0.5 * D
        s_m = S_0_negetive - a_T_zj_mines_mu - 0.5 * D
        print("s_p" , s_p )
        s_positive = max(s_p[0][0]  , 0 )
        s_negetive = max(s_m[0][0]  , 0 )

        s_s_positive.append(s_positive)
        s_s_negetive.append(s_negetive)

        S_0_positive = s_positive
        S_0_negetive = s_negetive

        # print("S : " , S )


    # print("TOTAL S - Positive , " , s_s_positive )
    # print("TOTAL S - Positive SHAPE, ", type(s_s_positive))
    # print("TOTAL S - Negetive , " , s_s_negetive )
    plt.figure(figsize=(10,10))
    plt.scatter(range(len(s_s_positive)) , s_s_positive  , s = 20   , alpha = 0.5 , label = 'S-Positive')
    plt.hlines(xmin=0 , xmax=len(s_s_positive  ) ,  y = 9 , label='UCL = 9 ' , colors='r')
    plt.legend()
    plt.grid()
    # plt.title(f"MCUSUM Chart For Phase 1")
    plt.title(f"MCUSUM Chart For Phase 2 - Detecting Shift λ={i/10}σ In Bias , β5 ")
    plt.xlabel("DATA Fault With Alpha : [0.001-0.050] and void : [0.1-0.4] ")
    plt.show()
    print("*****************************************************************")