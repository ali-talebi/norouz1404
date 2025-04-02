import pandas as pd
import numpy  as np
import os
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import yeojohnson

root_data = 'DATA_GENERATED/fix/alpha_10000_11000'

columns_ = ['bias', 'betha_1', 'betha_2', 'betha_3', 'betha_4', 'betha_5', 'R2',
            'locate', 'class', 'alpha', 'void', 'SSE', 'R2_adj', 'stat_error',
            'p_value_error', 'DW' , 'var_errors', 'SST', 'MSR', 'MSE', 'F', 'p_value_f']


Total_Df = pd.DataFrame()
Total_Df = pd.DataFrame(columns=columns_)
for index, name in enumerate(os.listdir(root_data)):
    df = pd.read_csv(f'{root_data}/{name}')
    Total_Df = pd.concat([Total_Df, df], axis=0)
    del df
Total_Df = Total_Df[Total_Df['class'] == 1.0 ]
Total_Df = Total_Df[['bias', 'betha_1', 'betha_2', 'betha_3', 'betha_4', 'betha_5']]

lof = LocalOutlierFactor(n_neighbors=20)
outlier_labels = lof.fit_predict(Total_Df)

# ---- حذف داده‌های پرت (برچسب -1 نشان‌دهنده داده پرت است) ----
Total_Df = Total_Df[outlier_labels != -1]


# for i in range(Total_Df.shape[1]):
#     coef = Total_Df.values[:, i]  # هر ستون یک ضریب
#     stat, p_value = stats.shapiro(coef)  # آزمون Shapiro-Wilk
#     print(f"ضریب {i+1} - Shapiro-Wilk p-value: {p_value}")
#
#     # ---- رسم نمودار QQ-Plot برای بررسی نرمال بودن ----
#     plt.figure(figsize=(5, 5))
#     stats.probplot(coef, dist="norm", plot=plt )
#     plt.title(f"QQ-Plot برای ضریب {i} - Shapiro-Wilk p-value: {p_value:.6f}")
#     plt.xlabel(f" In Data Fault - Alpha : [10-10.050] ")
#     plt.grid()
#     plt.show()

transformed_df = Total_Df.copy()  # کپی گرفتن از دیتافریم اصلی
lambda_values = {}

for column in Total_Df.columns:
    transformed_df[column], lambda_values[column] = yeojohnson(Total_Df[column])


for i in range(transformed_df.shape[1]):
    coef = transformed_df.values[:, i]  # هر ستون یک ضریب
    stat, p_value = stats.shapiro(coef)  # آزمون Shapiro-Wilk
    print(f"ضریب {i+1} - Shapiro-Wilk p-value: {p_value}")

    # ---- رسم نمودار QQ-Plot برای بررسی نرمال بودن ----
    plt.figure(figsize=(5, 5))
    stats.probplot(coef, dist="norm", plot=plt )
    plt.title(f"QQ-Plot برای ضریب {i} - Shapiro-Wilk p-value: {p_value:.6f}")
    plt.xlabel(f" After Yeo-Johnson Apply On Data Fault With Alpha : [10-10.050] ")
    plt.grid()
    plt.show()
