import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib import rcParams


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


overall_variance = Total_Df[(Total_Df['class'] == 0.0)]['var_errors'].sum()  / len(Total_Df[(Total_Df['class'] == 0.0)])
# Total_Df = Total_Df.sample()
root_new = 'DATA_GENERATED/fix/alpha_1_50'
Total_Df2 = pd.DataFrame(columns=columns_)
for index, name in enumerate(os.listdir(root_new)):
    df2 = pd.read_csv(f'{root_new}/{name}')
    Total_Df2 = pd.concat([Total_Df, df2], axis=0)
    del df2


df_main  = Total_Df[Total_Df['class'] == 0.0][['bias', 'betha_1', 'betha_2', 'betha_3', 'betha_4', 'betha_5']]

print("overall_variance : " , overall_variance )

df_fault = Total_Df2[Total_Df2['class'] == 1.0][['bias', 'betha_1', 'betha_2', 'betha_3', 'betha_4', 'betha_5']]

# print(df_main.describe())
# # df_main.describe().to_csv('DF_HEALTH.csv')
# df_fault.describe().to_csv('DF_FAULT[alpha_4000_4046]2.csv')
# import time
# time.sleep(600)
mu = np.array([df_main['bias'].mean(),df_main['betha_1'].mean(),df_main['betha_2'].mean(),df_main['betha_3'].mean(),df_main['betha_4'].mean(),df_main['betha_5'].mean()])

XTX_inverse = pd.read_csv('XTX_inverse.csv')
XTX_INVERSE_SIGMA_2 =  XTX_inverse * overall_variance
XTX_INVERSE_SIGMA_2_inverse = np.linalg.inv(XTX_INVERSE_SIGMA_2)

S_0_positive = 0
S_0_negetive = 0
for i in range(1 , 10 ) :

    shifts = np.array([(i/100) * df_main['bias'].std() , (i/100) * df_main['betha_1'].std() , 0 * df_main['betha_2'].std() , 0 * df_main['betha_3'].std() , 0* df_main['betha_4'].std() , (i/100)* df_main['betha_5'].std() ])
    SHIFT_COVARIANCE_VARIANCE = shifts @ XTX_INVERSE_SIGMA_2_inverse
    D      = (SHIFT_COVARIANCE_VARIANCE @ shifts.T) ** 0.5
    print("D : " , D )
    a_T = SHIFT_COVARIANCE_VARIANCE / D
    a_T = np.array(a_T).reshape(1,6)
    # print("a_T : " , a_T.shape )
    # print(a_T)
    s_s_positive = []
    s_s_negetive = []
    for j in range(len(df_fault)) :
        zj_mines_mu = df_fault.iloc[j, :] - mu
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
    plt.hlines(xmin=0 , xmax=len(s_s_positive  ) ,  y = 12.59 , label='UCL = 12.59 ' , colors='r')
    plt.legend()
    plt.grid()
    plt.title(f"MCUSUM Chart For Phase 2 - Detecting Shift λ={i/10}σ In Bias ")
    plt.xlabel("DATA Fault With alpha:[0.001-0.050]")
    plt.show()
    print("*****************************************************************")


