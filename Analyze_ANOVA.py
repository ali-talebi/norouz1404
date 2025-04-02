import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

Total_Df = pd.DataFrame()
#DATA_GENERATED/fix/4/alpha_3000_3050
root_data = 'DATA_GENERATED/fix/alpha_10000_11000'

columns_ = ['bias', 'betha_1', 'betha_2', 'betha_3', 'betha_4', 'betha_5', 'R2',
            'locate', 'class', 'alpha', 'void', 'SSE', 'R2_adj', 'stat_error',
            'p_value_error', 'DW' , 'var_errors' , 'SST', 'MSR', 'MSE', 'F', 'p_value_f']

Total_Df = pd.DataFrame(columns=columns_)
for index, name in enumerate(os.listdir(root_data)):
    df = pd.read_csv(f'{root_data}/{name}')
    Total_Df = pd.concat([Total_Df, df], axis=0)
    del df

print("Total_Df : " , Total_Df )
print("Total_Df : " , Total_Df['class'].value_counts())

# root2 = 'DATA_GENERATED/fix/2/alpha_1_50'
# Total_Df2 = pd.DataFrame()
# Total_Df2 = pd.DataFrame(columns=columns_)
# for index, name in enumerate(os.listdir(root2)):
#     df = pd.read_csv(f'{root2}/{name}')
#     Total_Df2 = pd.concat([Total_Df2, df], axis=0)
#     del df
#
# print("Total_Df2 : " , Total_Df2 )
# print("Total_Df2 : " , Total_Df2['class'].value_counts())
#
# fault  = Total_Df[Total_Df['class'] == 1.0 ]
# health = Total_Df2[Total_Df2['class'] == 0.0 ]
#
# combine_df = pd.concat([health,fault] ,axis = 0 )
combine_df = Total_Df
coefs = ['bias', 'betha_1', 'betha_2', 'betha_3', 'betha_4', 'betha_5']
sns.set(style="whitegrid")
for c in coefs :
    fig = plt.figure(figsize=(10, 10))
    # ax0 = fig.add_subplot(111)
    #sns.kdeplot(df_fault[c] , label = f'{c}' , shade = True )
    sns.boxplot(x='class', y=c, data=combine_df ,  palette='Set3' , hue='class' )
    plt.legend()
    plt.title("Class 0 : Health , Class 1 : Fault , Alpha : [10 - 10.050 ] - Void : [0.1-0.4]")
    plt.grid()
    # plt.tight_layout()
    plt.show()
