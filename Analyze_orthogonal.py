import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

Total_Df = pd.DataFrame()

root_data = 'DATA_GENERATED/fix_orthogonal'

columns_ = ['bias', 'coef_1', 'coef_2', 'coef_3', 'coef_4', 'coef_5', 'SSE', 'DW', 'R2', 'R2_adj', 'p_value_f',
           'p_value_error', 'class', 'alpha', 'void']

Total_Df = pd.DataFrame(columns=columns_)
for index, name in enumerate(os.listdir(root_data)):
    df = pd.read_csv(f'{root_data}/{name}')
    Total_Df = pd.concat([Total_Df, df], axis=0)
    del df

Total_Df.to_csv("main_orthogonal.csv" , index = False )
df_main  = Total_Df[(Total_Df['class'] == 0.0) & (Total_Df['DW'] < 2.5 ) & (Total_Df['DW'] > 1.5 )]
df_fault = Total_Df[(Total_Df['class'] != 0.0) & (Total_Df['DW'] < 2.5 ) & (Total_Df['DW'] > 1.5 )]

coefs = ['bias', 'coef_1', 'coef_2', 'coef_3', 'coef_4', 'coef_5']

for c in coefs :
    fig = plt.figure(figsize=(10, 10))
    # ax0 = fig.add_subplot(111)
    sns.kdeplot(df_fault[c] , label = f'{c} in Orthogonal Regression For Class Fault' , shade = True )
    plt.legend()
    plt.title("Alpha : [0.001 - 0.099]  - Void : [0.1-0.4]")
    plt.grid()
    plt.show()
#
#
# print(df_main.shape)

# print(df_main[:3])
# print("df_main shape " , df_main.shape )
# columns2 = ['bias', 'betha_1', 'betha_2', 'betha_3', 'betha_4', 'betha_5' ]
# df2 = df_main[columns2]
#
# bias_mean = df2['bias'].mean()
# betha_1_mean = df2['betha_1'].mean()
# betha_2_mean = df2['betha_2'].mean()
# betha_3_mean = df2['betha_3'].mean()
# betha_4_mean = df2['betha_4'].mean()
# betha_5_mean = df2['betha_5'].mean()
#
# df3 = []
#
# for index in range(df2.shape[0] - 1 ) :
#     r = df2.iloc[index + 1 , : ] - df2.iloc[index , : ]
#     df3.append(r)
#
#
# df4 = pd.DataFrame(df3 , columns=columns2 )
#
# n_df3 = np.array(df3)
# r = n_df3.T @ n_df3
# r2 = r / ( 2 * (len(df4)))
# print(r2)
#
#
# t2s = []
#
# for i in range(len(df2)) :
#     one = df2.iloc[i , 0 ] - bias_mean
#     two = df2.iloc[i, 1] - betha_1_mean
#     three = df2.iloc[i, 2] - betha_2_mean
#     four = df2.iloc[i, 3] - betha_3_mean
#     five = df2.iloc[i, 4] - betha_4_mean
#     six = df2.iloc[i, 5] - betha_5_mean
#     base = np.array([one,two,three,four,five,six])
#     t2 = base @  r2 @ base.T
#     t2s.append(t2)
#
#
# plt.scatter(range(len(t2s)) , t2s , label = 'T2 for Each Profile' )
# plt.xlabel('Profile Number')
# plt.title("T2 - Phase 1 ")
# plt.legend()
# plt.grid()
# plt.show()


# df_fault2 = df_fault[columns2]
# t3s = []
# for i in range(len(df_fault2)) :
#     one = df_fault2.iloc[i , 0 ] - bias_mean
#     two = df_fault2.iloc[i, 1] - betha_1_mean
#     three = df_fault2.iloc[i, 2] - betha_2_mean
#     four = df_fault2.iloc[i, 3] - betha_3_mean
#     five = df_fault2.iloc[i, 4] - betha_4_mean
#     six = df_fault2.iloc[i, 5] - betha_5_mean
#     base = np.array([one,two,three,four,five,six])
#     t2 = base @  r2 @ base.T
#     t3s.append(t2)
#
#
# plt.scatter(range(len(t3s)) , t3s , label = 'T2 for Each Profile' )
# plt.xlabel('Profile Number')
# plt.title("T2 - Phase 2 ")
# plt.legend()
# plt.grid()
# plt.show()
#
#




