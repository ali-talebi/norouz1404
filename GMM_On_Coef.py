import pandas as pd
import numpy as np
import os
from sklearn.mixture import GaussianMixture
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

root = 'DATA_GENERATED/fix/2/alpha_1_50'

Total_df = pd.DataFrame()

for name in os.listdir(root) :
    name2  = f'{root}/{name}'
    df = pd.read_csv(name2)
    Total_df = pd.concat([Total_df,df] , axis= 0 )
    del df

# print(Total_df.shape)
# print(Total_df.columns)
df_health = Total_df[Total_df['class'] == 0.0 ]
df_fault  = Total_df[Total_df['class'] == 1.0 ]
columns2 = ['bias', 'betha_1', 'betha_2', 'betha_3', 'betha_4', 'betha_5']

counter = {0:3,1:5,2:4,3:2,4:2,5:4}

coefficients_IC = df_health[columns2].values
gmm_models = []
for i in range(coefficients_IC.shape[1]):  # برای هر ضریب
    n_component = counter[i]
    gmm = GaussianMixture(n_components=n_component)  # تعداد مؤلفه‌ها بر اساس توضیحات شما
    gmm.fit(coefficients_IC[:, i].reshape(-1, 1))
    gmm_models.append(gmm)
    # print("gmm means : " , gmm.means_ )



beta_new = np.array(df_health[columns2].values[15 , : ])

for index , value  in enumerate(beta_new):
    # محاسبه فاصله Mahalanobis برای مشاهده جدید
    mahalanobis_distance_squared = 0
    means     = gmm_models[index].means_.flatten()  # میانگین‌ها (μ_k)
    variances = gmm_models[index].covariances_.flatten()  # واریانس‌ها (σ_k^2)
    weights   = gmm_models[index].weights_  # وزن‌ها (w_k)

    print("Means : " , means)
    print("Variance : " , variances )
    for k in range(len(means)):
        diff = value - means[k]
        variance = variances[k]
        weight = weights[k]
        mahalanobis_distance_squared += weight * (diff ** 2 / variance)

    print("Mahalanobis Distance Squared (D^2):", mahalanobis_distance_squared)

    # تعیین حد کنترل (Control Limit)
    alpha = 0.05  # سطح معناداری
    p = len(means)  # تعداد مؤلفه‌ها (درجه آزادی)
    control_limit = chi2.ppf(1 - alpha, df=p)  # حد کنترل (L)

    print("Control Limit (L):", control_limit)

    # تصمیم‌گیری
    if mahalanobis_distance_squared > control_limit:
        print("The new observation is OUT-OF-CONTROL!")
    else:
        print("The new observation is IN-CONTROL.")






# print(gmm_models)
# # محاسبه میانگین و کواریانس ضرایب IC
# mu = np.array([gmm.means_[0] for gmm in gmm_models]).ravel()
# cov = [gmm.covariances_[0].tolist()[0][0] for gmm in gmm_models]
# Sigma = np.diag(cov)
# print("Mu : " , mu )
# print("Mu shape: " , mu.shape )
# print("cov : " , cov )
# print("Sigma : \n" , Sigma )
# print("Sigma Shape: \n" , Sigma.shape )
#
# #
# #
#
# print("beta_new : " , beta_new )
# print("beta_new shape : " , beta_new.shape )
# try :
#     distance = mahalanobis(beta_new, mu, np.linalg.inv(Sigma))
#
#     print("DISTANCE : " , distance )
# except Exception as e :
#     print("Er:" , e )

