import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import pandas as pd
import os
import seaborn as sns

# داده‌ها


root_data = 'DATA_GENERATED/fix/2/alpha_1_50'

columns_ = ['bias', 'betha_1', 'betha_2', 'betha_3', 'betha_4', 'betha_5', 'R2',
            'locate', 'class', 'alpha', 'void', 'SSE', 'R2_adj', 'stat_error',
            'p_value_error', 'DW' , 'var_errors', 'SST', 'MSR', 'MSE', 'F', 'p_value_f']



Total_Df = pd.DataFrame(columns=columns_)
for index, name in enumerate(os.listdir(root_data)):
    df = pd.read_csv(f'{root_data}/{name}')
    Total_Df = pd.concat([Total_Df, df], axis=0)
    del df






corr = Total_Df[['bias', 'betha_1', 'betha_2', 'betha_3', 'betha_4', 'betha_5']].corr()
plt.figure(figsize=(8, 6))  # تنظیم اندازه نمودار
sns.heatmap(corr, annot=True, cmap="viridis", fmt=".2f", linewidths=0.5)
# تنظیمات نمودار
plt.title("Correlation Matrix of Coefficients")
plt.show()


data = Total_Df[Total_Df['class'] == 0.0 ]['betha_5'].values.reshape(-1, 1)
sns.set_palette("Set3")
# تعیین تعداد مولفه‌ها (k)
n_components_range = range(1, 6)
models = [GaussianMixture(n_components=n, random_state=0).fit(data) for n in n_components_range]
aics = [model.aic(data) for model in models]
bics = [model.bic(data) for model in models]

# رسم AIC و BIC
plt.plot(n_components_range, aics, label="AIC")
plt.plot(n_components_range, bics, label="BIC")
plt.xlabel("Number of Components (k)")
plt.ylabel("Criterion Value")
plt.legend()
plt.show()

# برازش GMM با k=2
gmm = GaussianMixture(n_components=4, random_state=0)
gmm.fit(data)

# استخراج پارامترها
means = gmm.means_.flatten()
variances = gmm.covariances_.flatten()
weights = gmm.weights_

print("Means:", means)
print("Variances:", variances)
print("Weights:", weights)

# تخمین PDF
x = np.linspace(-5, 5, 1000).reshape(-1, 1)
pdf = np.exp(gmm.score_samples(x))


# رسم نتایج
plt.hist(data, bins=30, density=True, alpha=0.8, color='green', label="Histogram")
plt.plot(x, pdf, label="GMM PDF", color='red')
plt.title("betha_5")
plt.grid()
plt.legend()
plt.show()
best_k_aic = n_components_range[np.argmin(aics)]
best_k_bic = n_components_range[np.argmin(bics)]

print(f"Best k based on AIC: {best_k_aic}")
print(f"Best k based on BIC: {best_k_bic}")