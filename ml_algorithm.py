import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

matplotlib.use('TkAgg')  # Use TkAgg backend

root_data = 'Modified_DATA_GENERATOR/fix2/alpha_1_50'
# root_data = 'Modified_DATA_GENERATOR/fix2/alpha_1_1.050'

columns_ = ['bias', 'betha_1', 'betha_2', 'betha_3', 'betha_4', 'betha_5', 'R2',
            'locate', 'class', 'alpha', 'void', 'SSE', 'R2_adj', 'stat_error',
            'p_value_error', 'DW', 'var_errors', 'SST', 'MSR', 'MSE', 'F', 'p_value_f']

Total_Df = pd.DataFrame(columns=columns_)
for index, name in enumerate(os.listdir(root_data)):
    df = pd.read_csv(f'{root_data}/{name}')
    Total_Df = pd.concat([Total_Df, df], axis=0)
    del df

columns2 = ['bias', 'betha_1', 'betha_2', 'betha_3', 'betha_4', 'betha_5', 'class']
main_df = Total_Df[columns2]
print(main_df.head())

plt.figure(figsize=(10, 10))
df_health = main_df[main_df['class'] == 0.0]
df_fault = main_df[main_df['class'] == 1.0]

y_label = main_df['class']
data = main_df.drop(['class'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(data, y_label, test_size=0.3, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

r = RandomForestClassifier()
r.fit(x_train, y_train)
y_pre = r.predict(x_test)
acc = accuracy_score(y_pre, y_test)

cm = confusion_matrix(y_test, y_pre)
sns.heatmap(cm,
            annot=True,
            fmt='g', )
plt.ylabel('Actual', fontsize=13)
plt.gca().xaxis.set_label_position('top')
plt.xlabel('alpha : 0.0001 - 0.0005 , void : 0.05 - 0.25 ', )
plt.gca().xaxis.tick_top()
plt.title(f'Accuracy : {acc}')
plt.gca().figure.subplots_adjust(bottom=0.2)
plt.gca().figure.text(0.5, 0.05, 'Prediction', ha='center', fontsize=13)
plt.show()
