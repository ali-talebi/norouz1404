import pandas as pd
import os
root_data = "DATA_GENERATED/fix/alpha_4000_5000"
columns_ = ['bias', 'betha_1', 'betha_2', 'betha_3', 'betha_4', 'betha_5', 'R2',
            'locate', 'class', 'alpha', 'void', 'SSE', 'R2_adj', 'stat_error',
            'p_value_error', 'DW' , 'var_errors', 'SST', 'MSR', 'MSE', 'F', 'p_value_f']


Total_Df = pd.DataFrame(columns=columns_)
for index, name in enumerate(os.listdir(root_data)):
    df = pd.read_csv(f'{root_data}/{name}')
    Total_Df = pd.concat([Total_Df, df], axis=0)
    del df

df_main  = Total_Df[(Total_Df['class'] == 1.0)][['bias', 'betha_1', 'betha_2', 'betha_3', 'betha_4', 'betha_5','alpha','void']]
df_main.describe().to_csv('DATA_GENERATED/describes/alpha_4000_5000.csv')
print("df_main.describe()\n" , df_main.describe())