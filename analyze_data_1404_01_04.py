import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

print(Total_Df['class'].value_counts())
print("Total_Df columns : " , Total_Df.columns )
df_main  = Total_Df[(Total_Df['class'] == 0.0)][['bias', 'betha_1', 'betha_2', 'betha_3', 'betha_4', 'betha_5','var_errors']]
df_fault = Total_Df[(Total_Df['class'] == 1.0)][['bias', 'betha_1', 'betha_2', 'betha_3', 'betha_4', 'betha_5','var_errors']]

print("df_main : ", df_main.shape  )

coefs = ['bias', 'betha_1', 'betha_2', 'betha_3', 'betha_4', 'betha_5']

# for c in coefs :
#     fig = plt.figure(figsize=(10, 10))
#     # ax0 = fig.add_subplot(111)
#     sns.kdeplot(df_fault[c] , label = f'{c}' , shade = True , color = 'purple' )
#     plt.legend()
#     plt.title("Coefs Distribution for Fault Data , Alpha : [10.0 - 10.050] - Void : [0.1-0.4]")
#     plt.grid()
#     plt.show()
#
#
# print(df_main.shape)



# df_main['var_i'] = df_main['SSE'] / ( 48 - 6 )
# sigma_2 = df_main['var_errors'].sum() / len(df_main['var_errors'])
#
# V_TOTAL = []
# for index  in range(len(df_main) - 1 ) :
#     r = df_main.iloc[index + 1 , : ] - df_main.iloc[index, : ]
#     V_TOTAL.append(r)
#
# V_TOTAL = np.array(V_TOTAL)
# print("V_TOTAL shape " , V_TOTAL.shape )



df_main['SD']=df_main['var_errors'] ** 0.5
mean_sd = df_main['SD'].mean()
print("mean_sd : " , mean_sd )
c4  = ( 4 * 47 ) / (4 * 48 - 3)
print("C4 : " , c4 )
UCL = mean_sd + 3 * ( mean_sd / c4 ) * (1 - c4 ** 2 ) ** 0.5
LCL = mean_sd - 3 * ( mean_sd / c4 ) * (1 - c4 ** 2 ) ** 0.5
print("UCL : " , UCL )
print("LCL : " , LCL )

plt.scatter(range(len(df_main['SD'])) , df_main['SD'] , label = 'S(i)' )
plt.ylabel('S(i)')
plt.xlabel('Profile Number ')
plt.title("Shewhart individuals control chart Before Phase 1")
plt.hlines(xmin= 0 , xmax=len(df_main['SD']) , lw=10 ,  y = UCL  , colors = 'r'  )
plt.hlines(xmin= 0 , xmax=len(df_main['SD']) , lw=10 ,  y = LCL , ls = '--' , colors = 'orange'  )

plt.legend()
plt.grid()
plt.show()

# print(sigma_2)
# df_sig_var = pd.read_csv('SIG-VAR.csv')
# df_sig_var2 = df_sig_var * sigma_2
# df_sig_var2.to_csv('df_sig_var2.csv' , index = False )
# print(df_main[:3])
# print("df_main shape " , df_main.shape )
columns2 = ['bias', 'betha_1', 'betha_2', 'betha_3', 'betha_4', 'betha_5' ]
df2 = df_main[columns2]
#
bias_mean = df2['bias'].mean()
betha_1_mean = df2['betha_1'].mean()
betha_2_mean = df2['betha_2'].mean()
betha_3_mean = df2['betha_3'].mean()
betha_4_mean = df2['betha_4'].mean()
betha_5_mean = df2['betha_5'].mean()
#
df3 = []

for index in range(df2.shape[0] - 1 ) :
    r = df2.iloc[index + 1 , : ] - df2.iloc[index , : ]
    df3.append(r)


df4 = pd.DataFrame(df3 , columns=columns2 )

n_df3 = np.array(df3)
r = n_df3.T @ n_df3
r2 = r / ( 2 * (len(df4)))
print(r2)
print("r2 : \n" , r2 )
#
#
t2s = []



for i in range(len(df2)) :
    one = df2.iloc[i , 0 ] - bias_mean
    two = df2.iloc[i, 1] - betha_1_mean
    three = df2.iloc[i, 2] - betha_2_mean
    four = df2.iloc[i, 3] - betha_3_mean
    five = df2.iloc[i, 4] - betha_4_mean
    six = df2.iloc[i, 5] - betha_5_mean
    base = np.array([one,two,three,four,five,six])
    t2 = base @  r2 @ base.T
    t2s.append(t2)


plt.scatter(range(len(t2s)) , t2s , label = 'T2 for Each Profile' )
plt.xlabel('Profile Number')
plt.title("T2 - Phase 1 ON Data Health")
plt.legend()
plt.grid()
plt.show()
#
#
root_data2 = 'DATA_GENERATED/fix/alpha_10000_11000'

Total_Df_33 = pd.DataFrame(columns=columns_)
for index, name in enumerate(os.listdir(root_data2)):
    df = pd.read_csv(f'{root_data2}/{name}')
    Total_Df_33 = pd.concat([Total_Df_33, df], axis=0)
    del df

t3s = []

Total_Df_33 = Total_Df_33[Total_Df_33['class'] == 1.0 ][columns2]
new_df = Total_Df_33

for i in range(len(new_df)) :
    one   = new_df.iloc[i , 0 ] - bias_mean
    two   = new_df.iloc[i, 1] - betha_1_mean
    three = new_df.iloc[i, 2] - betha_2_mean
    four  = new_df.iloc[i, 3] - betha_3_mean
    five  = new_df.iloc[i, 4] - betha_4_mean
    six   = new_df.iloc[i, 5] - betha_5_mean
    base  = np.array([one,two,three,four,five,six])
    t2    = base @  r2 @ base.T
    t3s.append(t2)


plt.scatter(range(len(t3s)) , t3s , label = 'T2 Hotteling' )
plt.xlabel('Profile Number')
plt.title("T2 - Phase 2 For Detecting Shift in Coef on Data with Alpha : 10.0 - 10.050  ")
plt.hlines(xmin= 0 , xmax=len(t3s) , lw=5 , label='UCL = 6.61' ,  y = 6.61  , colors = 'r'  )
plt.legend()
plt.grid()
plt.show()






