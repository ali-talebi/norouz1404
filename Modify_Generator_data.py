import time

import pandas as pd
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import os
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
# from fitter import Fitter, get_common_distributions, get_distributions
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
# import tensorflow as tf
# from tensorflow import keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, BatchNormalization
# from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import cm
from scipy.stats import shapiro
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import f

# pip freeze > requirements.txt

root_path = 'Total_Data_Simulation'
# store_path = 'DATA_GENERATED/fix/alpha_10000_11000'
store_path = 'Modified_DATA_GENERATOR/fix2/alpha_150_200'

total_file_link = []
for i in os.listdir(root_path):
    total_file_link.append(root_path + f'/{i}')

pd.read_csv(total_file_link[2])

total_alpha_content = [i / 1000 for i in range(150, 200)]
total_flat_content  = [i / 10   for i in range(1, 5)]
total_location = list(range(2, 15))

points = []
start_x = 50
start_y = [0, -5, -10.0]
for i in range(15):
    start_x += 15

    for j in range(3):
        points.append([start_x, start_y[j], 0])

points.append([330, -7.5, 0])
points.append([330, -12.5, 0])
points.append([330, -17.5, 0])
total_points = np.array(points)
# plt.scatter(total_points[:, 0], total_points[:, 1], label="target")
# plt.legend()
# plt.title("Blade Shape - Picture 1.7 page 7 document volume 2 ")
# plt.xlabel(" X (in) ")
# plt.ylabel(" Y (in) ")
# plt.grid()
# plt.show()

p = pd.DataFrame(total_points, columns=['X', 'Y', 'Z'])
p.tail(10)
main_data_frame = pd.DataFrame()



total_simulation = {
    'bias_correlation': [],
    'betha_0_correlation': [],
    'betha_1_correlation': [],
    'betha_2_correlation': [],
    'betha_3_correlation': [],
    'betha_4_correlation': [],
    'betha_5_correlation': [],
    'alpha': [],
    'flap': [],
    'accuracy_nav': [],

    # 'accuracy_random_forest' : [] ,
    # 'n_estimators_RF' : [] ,
    # 'max_features_RF' : [] ,
    # 'max_depth_RF'    : [] ,
    # 'criterion_RF'    : [] ,

    'accuracy_extra_classifier': [],
    'accuracy_svc_': [],
    'C_svc': [],
    'kernel_svc': [],
    'gamma_svc': [],
    # 'ann'           : []

}

MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_HEALTH = {
    'mean_bias': [],
    'std_bias': [],
    'mean_betha_0': [],
    'std_betha_0': [],
    'mean_betha_1': [],
    'std_betha_1': [],
    'mean_betha_2': [],
    'std_betha_2': [],
    'mean_betha_3': [],
    'std_betha_3': [],
    'mean_betha_4': [],
    'std_betha_4': [],
    'mean_betha_5': [],
    'std_betha_5': [],
    'mean_error': [],
    'std_error': []
}

MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_HEALTH_T = {
    'mean_bias': [],
    'std_bias': [],
    'mean_betha_0': [],
    'std_betha_0': [],
    'mean_betha_1': [],
    'std_betha_1': [],
    'mean_betha_2': [],
    'std_betha_2': [],
    'mean_betha_3': [],
    'std_betha_3': [],
    'mean_betha_4': [],
    'std_betha_4': [],
    'mean_betha_5': [],
    'std_betha_5': [],
    'mean_error': [],
    'std_error': []
}

MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_FAULT = {
    'mean_bias': [],
    'std_bias': [],
    'mean_betha_0': [],
    'std_betha_0': [],
    'mean_betha_1': [],
    'std_betha_1': [],
    'mean_betha_2': [],
    'std_betha_2': [],
    'mean_betha_3': [],
    'std_betha_3': [],
    'mean_betha_4': [],
    'std_betha_4': [],
    'mean_betha_5': [],
    'std_betha_5': [],
    'mean_error': [],
    'std_error': []
}

MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_FAULT_T = {
    'mean_bias': [],
    'std_bias': [],
    'mean_betha_0': [],
    'std_betha_0': [],
    'mean_betha_1': [],
    'std_betha_1': [],
    'mean_betha_2': [],
    'std_betha_2': [],
    'mean_betha_3': [],
    'std_betha_3': [],
    'mean_betha_4': [],
    'std_betha_4': [],
    'mean_betha_5': [],
    'std_betha_5': [],
    'mean_error': [],
    'std_error': []
}

total_noew_for_generate = []
total_noew_for_fault1 = []
total_noew_for_fault2 = []
total_noew_for_fault3 = []

for_each_generate_health = []
for_each_generate_health_t = []
for_each_generate_fault = []
for_each_generate_fault_t = []

for iter_alpha in total_alpha_content:
    iter_each_generate_health = []
    iter_each_generate_health_t = []
    iter_each_generate_fault = []
    iter_each_generate_fault_t = []
    for iter_flat in total_flat_content:
        print(f"iter_alpha : {iter_alpha} | iter_flat : {iter_flat} ")
        params_health = []
        params_fault = []
        params_fault_t = []
        params_health_t = []
        for iter_link in total_file_link:
            df_table_change = pd.read_csv(iter_link)
            for locate in total_location:
                related_x = -1
                related_y = -1
                related_z = -1
                for iteration in range(5):
                    alpha = 0
                    term = 0
                    Flat_Add = 0
                    noise_activate = ((np.random.randn() + 1) / 100)
                    # print( f'Alpha : {iter_alpha} ' ,  f'Table Change : {iter_link}' , ' -- ' , f'Locate : {locate} ' , ' -- ' , f"Iter : {iteration}" , ' -- ' , '')
                    select_location_combin_flap_term = 0
                    select_locatiom_flap = 0

                    total_health_iter = []
                    total_health_iter_t = []
                    total_fault_iter = []
                    total_fault_t_iter = []
                    new_data_simulated_ = []
                    new_data_simulated_fault = []
                    new_data_simulated_fault_t = []
                    new_data_simulated_health_t = []

                    for i in range(df_table_change.shape[0]):
                        x_mean = df_table_change.iloc[i, 1]
                        x_std = df_table_change.iloc[i, 2]
                        y_mean = df_table_change.iloc[i, 3]
                        y_std = df_table_change.iloc[i, 4]
                        z_mean = df_table_change.iloc[i, 5]
                        z_std = df_table_change.iloc[i, 6]

                        if i > locate:
                            alpha = iter_alpha
                            term = -1 * (alpha * abs(i - locate) + noise_activate)
                            Flat_Add = iter_flat

                        for element in total_points[i * 3: (i + 1) * 3, :]:
                            x_sample = element[0]
                            y_sample = element[1]
                            z_sample = element[2]
                            # rng = np.random.default_rng()
                            # x_added  = rng.normal(x_mean, x_std, size=1)
                            # x_added  = x_added.tolist()[0]
                            # x_added = np.random.normal(x_mean, x_std, 1).tolist()[0]
                            # x_sample += related_x * x_added
                            #
                            # # y_added  = rng.normal(y_mean, y_std, size=1)
                            # # y_added  = y_added.tolist()[0]
                            # y_added = np.random.normal(y_mean, y_std, 1).tolist()[0]
                            #
                            # y_sample += related_y * y_added

                            # z_added  = rng.normal(z_mean, z_std, size=1)
                            z_added = np.random.normal(z_mean, z_std, 1)
                            z_base = z_sample

                            z_added_term = z_added.tolist()[0] + term
                            z_added = z_added.tolist()[0]
                            z_sample_new_term = z_base + related_z * z_added_term
                            z_sample_new = z_base + related_z * z_added
                            z_flat = z_sample_new_term
                            z_flap_health = z_sample_new

                            if select_location_combin_flap_term == 0:
                                if np.random.randint(0, 2):
                                    z_flat = z_sample_new_term + Flat_Add

                            if select_locatiom_flap == 0:
                                if np.random.randint(0, 2):
                                    z_flap_health = z_sample_new + Flat_Add

                            new_data_simulated_.append([x_sample, y_sample, z_sample_new])
                            new_data_simulated_fault.append([x_sample, y_sample, z_sample_new_term])
                            new_data_simulated_fault_t.append([x_sample, y_sample, z_flat])
                            new_data_simulated_health_t.append([x_sample, y_sample, z_flap_health])

                            total_health_iter.append([x_sample, y_sample, z_sample_new])
                            total_fault_iter.append([x_sample, y_sample, z_sample_new_term])
                            total_fault_t_iter.append([x_sample, y_sample, z_flat])
                            total_health_iter_t.append([x_sample, y_sample, z_flap_health])

                    # fig = plt.figure(figsize=(25 , 13 ))
                    # ax0 = fig.add_subplot(1 , 6 , 1 , projection='3d' )
                    # ax1 = fig.add_subplot(1 , 6 , 2 , projection='3d' )
                    # ax2 = fig.add_subplot(1 , 6 , 3 , projection='3d' )
                    # ax3 = fig.add_subplot(1 , 6 , 4 , projection='3d' )
                    # ax4 = fig.add_subplot(1 , 6 , 5 , projection='3d' )
                    # ax5 = fig.add_subplot(1 , 6 , 6 , projection='3d' )
                    #
                    # ax6  = fig.add_subplot(3 , 6 , 1 )
                    # ax60 = fig.add_subplot(3 , 6 , 2 )
                    # ax7  = fig.add_subplot(3 , 6 , 3 )
                    # ax8  = fig.add_subplot(3 , 6 , 4 )
                    #
                    #
                    #
                    #
                    #
                    #
                    #
                    # total_health_iter = np.array(total_health_iter)
                    # total_health_iter = np.array(total_health_iter)
                    # total_fault_iter = np.array(total_fault_iter)
                    # total_fault_t_iter = np.array(total_fault_t_iter)
                    # total_health_iter_t = np.array(total_health_iter_t)
                    # print(f"iter_alpha : {iter_alpha} , flap: {iter_flat} , locate : {locate} ,  iteration : {iteration} ")
                    #
                    # ax0.scatter3D(total_health_iter[: , 0 ] , total_health_iter[: , 1 ] , total_health_iter[: , 2 ]  , label='health' , c = 'r'   )
                    # ax0.scatter3D(total_fault_iter[: , 0 ] , total_fault_iter[: , 1 ] , total_fault_iter[: , 2 ]  , label=f'fault_alpha : {iter_alpha}' ,  c = 'b' )
                    # ax1.scatter3D(total_fault_iter[: , 0 ] , total_fault_iter[: , 1 ] , total_fault_iter[: , 2 ]  ,       label=f'fault_alpha : {iter_alpha}' , c ='blue', s = 30     )
                    # ax1.scatter3D(total_fault_t_iter[: , 0 ] , total_fault_t_iter[: , 1 ] , total_fault_t_iter[: , 2 ]  , label=f'fault_alpha : {iter_alpha}+ void {iter_flat/10}' , c='red' , s = 30 )
                    # ax2.scatter3D(total_fault_t_iter[: , 0 ] , total_fault_t_iter[: , 1 ] , total_fault_t_iter[: , 2 ]  , label=f'fault_alpha : {iter_alpha}+ void {iter_flat/10}' , c='green'  , s = 30  )
                    #
                    # ax3.scatter3D(total_health_iter[: , 0 ] , total_health_iter[: , 1 ]  , total_health_iter[: , 2 ]  ,       label=f'fault_void ' , c ='blue'  )
                    # ax3.scatter3D(total_fault_iter[: , 0 ] , total_fault_iter[: , 1]  , total_fault_iter[: , 2 ]  ,       label=f'fault_alpha : {iter_alpha}' , c ='yellow' )
                    # ax3.set_xlabel("axis x ")
                    # ax3.set_ylabel("axis y ")
                    # ax3.set_zlabel("axis z ")
                    #
                    # ax4.scatter3D(total_health_iter[: , 0 ] , total_health_iter[: , 1 ]  , total_health_iter[: , 2 ]  ,       label='health' , c ='blue',  )
                    # ax4.scatter3D(total_fault_t_iter[: , 0 ] , total_health_iter[: , 1 ]  , total_fault_t_iter[: , 2 ]  , label=f'fault_alpha+ void ' , c='red' ,  )
                    # ax4.set_xlabel("axis x")
                    # ax4.set_ylabel("axis y")
                    # ax4.set_zlabel("axis z")
                    #
                    # ax5.scatter3D(total_health_iter[: , 0 ]  , total_health_iter[: , 1 ] , total_fault_iter[: , 2 ]  ,       label='health' , c ='blue' ,  )
                    # ax5.scatter3D(total_health_iter_t[: , 0 ] , total_health_iter_t[: , 1] , total_fault_t_iter[: , 2 ]  , label='fault_void' , c='green' ,  )
                    # ax5.set_xlabel("axis x ")
                    # ax5.set_ylabel("axis y ")
                    # ax5.set_zlabel("axis z ")
                    #
                    # ax6.scatter(total_health_iter[: , 0 ]  , total_health_iter[: , 2 ]  , label='health' , c='b' , s = 40 ,  )
                    # ax6.scatter(total_health_iter_t[: , 0 ]  , total_health_iter_t[: , 2 ]  , label='fault_void' , c='r' , s = 30 ,  )
                    # ax6.set_xlabel("axis  x ")
                    # ax6.set_ylabel("axis  z ")
                    # ax6.set_title(f"alpha : {iter_alpha} , void : {iter_flat/10} , location : {locate}")
                    #
                    # ax60.scatter(total_health_iter[: , 0 ]  , total_health_iter[: , 2 ]  , label='health' , c='b' , s = 40 ,  )
                    # ax60.scatter(total_fault_iter[: , 0 ]  , total_fault_iter[: , 2 ]  , label='fault_alpha' , c='r' , s = 30 ,  )
                    # ax60.set_xlabel("axis  x ")
                    # ax60.set_ylabel("axis  z ")
                    #
                    # ax7.scatter(total_health_iter[: , 0 ]  , total_health_iter[: , 2 ]  , label='health' , c='b' , s = 40 , alpha = 0.5 )
                    # ax7.scatter(total_fault_t_iter[: , 0 ]  , total_fault_t_iter[: , 2 ]  , label='fault_alpha + void ' , c='yellow' , s = 30  )
                    # ax7.set_xlabel("axis  x ")
                    # ax7.set_ylabel("axis  z ")
                    #
                    # ax8.scatter(total_fault_iter[: , 0 ]  , total_fault_iter[: , 2 ]  , label=f'fault_alpha : {iter_alpha}' , c='b' , )
                    # ax8.scatter(total_fault_t_iter[: , 0 ]  , total_fault_t_iter[: , 2 ]  , label='Fault_alpha + void' , c='orange')
                    # ax8.set_xlabel("axis  x ")
                    # ax8.set_ylabel("axis  z ")
                    #
                    #
                    # ax0.legend()
                    # ax0.grid()
                    # ax0
                    # ax1.legend()
                    # ax1.grid()
                    # ax2.legend()
                    # ax2.grid()
                    # ax3.legend()
                    # ax3.grid()
                    #
                    # ax4.legend()
                    # ax4.grid()
                    # ax5.legend()
                    # ax5.grid()
                    #
                    #
                    # ax6.legend()
                    # ax6.grid()
                    # ax60.legend()
                    # ax60.grid()
                    #
                    # ax7.legend()
                    # ax7.grid()
                    # ax8.legend()
                    # ax8.grid()
                    # plt.show()

                    new_data_simulated_ = np.array(new_data_simulated_)
                    new_data_simulated_fault = np.array(new_data_simulated_fault)
                    new_data_simulated_fault_t = np.array(new_data_simulated_fault_t)
                    new_data_simulated_health_t = np.array(new_data_simulated_health_t)

                    total_noew_for_generate.append(new_data_simulated_)
                    total_noew_for_fault1.append(new_data_simulated_health_t)
                    total_noew_for_fault2.append(new_data_simulated_fault)
                    total_noew_for_fault3.append(new_data_simulated_fault_t)
                    # fig = plt.figure(figsize=(30, 10))
                    df_health = pd.DataFrame()
                    df_health['X_'] = new_data_simulated_[:, 0]
                    df_health['Y_'] = new_data_simulated_[:, 1]
                    df_health['Z_'] = new_data_simulated_[:, 2]
                    #
                    # matrix_design = pd.DataFrame(columns = ['one','x','y','x2','y2','xy'])
                    # matrix_design['one'] = [1 for i in range(48)]
                    # matrix_design['x']   = df_health['X_']
                    # matrix_design['y']   = df_health['Y_']
                    # matrix_design['x2']  = df_health['X_'] ** 2
                    # matrix_design['y2']  = df_health['Y_'] ** 2
                    # matrix_design['xy']  = df_health['X_'] * df_health['Y_']
                    # print("matrix_design" , matrix_design )
                    # XTX = (matrix_design.T @ matrix_design)
                    # # # print("***************************" , sig_var )
                    # XTX_inverse = np.linalg.inv(XTX)
                    # pd.DataFrame(XTX_inverse ,columns = ['one','x','y','x2','y2','xy'] ).to_csv('XTX_inverse.csv' , index = False )
                    # import time
                    # time.sleep(600)
                    # print("---->>>>>>>>>>>>>>>>>>>>XTX-1" , sig_var2 )
                    # print('corr : ', df_health[['X_', 'Y_']].corr())
                    #
                    # # آزمون Shapiro-Wilk برای X1
                    # stat_x, p_value_x = shapiro(df_health['X_'])
                    # print(f"Shapiro-Wilk Test for X1: Stat={stat_x}, P-Value={round(p_value_x, 5)}")
                    #
                    # # آزمون Shapiro-Wilk برای X2
                    # stat_y, p_value_y = shapiro(df_health['Y_'])
                    # print(f"Shapiro-Wilk Test for Y_: Stat={stat_y}, P-Value={round(p_value_y, 5)}")
                    #
                    # # آزمون Shapiro-Wilk برای Z
                    # stat_z, p_value_z = shapiro(df_health['Z_'])
                    # print(f"Shapiro-Wilk Test for Y_: Stat={stat_z}, P-Value={round(p_value_z, 5)}")
                    #
                    # import scipy.stats as stats
                    #
                    # # Q-Q Plot برای X1
                    # stats.probplot(df_health['X_'], dist="norm", plot=plt)
                    # plt.title(
                    #     f"Q-Q Plot for X_ ,Shapiro-Wilk Test for X_: Stat={stat_x}, P-Value={round(p_value_x, 5)}")
                    # plt.show()
                    #
                    # # Q-Q Plot برای X2
                    # stats.probplot(df_health['Y_'], dist="norm", plot=plt)
                    # plt.title(f"Q-Q Plot for Y_,Shapiro-Wilk Test for Y_: Stat={stat_y}, P-Value={round(p_value_y, 5)}")
                    # plt.show()
                    #
                    # # Q-Q Plot برای Z
                    # stats.probplot(df_health['Z_'], dist="norm", plot=plt)
                    # plt.title(f"Q-Q Plot for Z_,Shapiro-Wilk Test for Z_: Stat={stat_z}, P-Value={round(p_value_z, 5)}")
                    # plt.show()
                    #
                    # # فرض کنید داده‌ها در یک DataFrame به نام df قرار دارند
                    # sns.histplot(df_health['X_'], kde=True, color='blue', label='X_')
                    # sns.histplot(df_health['Y_'], kde=True, color='orange', label='Y_')
                    # sns.histplot(df_health['Z_'], kde=True, color='red', label='Z_')
                    # plt.legend()
                    # plt.title("Histogram of Independent Variables and Response Variable")
                    # plt.show()

                    # plt.hist(df_health['X_'], bins=50, color='blue', label='X_')
                    # plt.hist(df_health['Y_'], bins=50, color='orange', label='Y_')
                    # plt.hist(df_health['Z_'], bins=50, color='red', label='Z_')
                    # plt.legend()
                    # plt.title("Histogram of Independent Variables and Response Variable")
                    # plt.show()

                    poly = PolynomialFeatures(degree=2)
                    fig = plt.figure(figsize=(10, 10))
                    x_poly = poly.fit_transform(df_health[['X_', 'Y_']])
                    # model_stat = sm.OLS(df_health['Z_'], x_poly).fit()
                    # print(model_stat.summary())

                    scaler = StandardScaler()
                    df_stander = pd.DataFrame(scaler.fit_transform(x_poly),
                                              columns=['0', 'X', 'Y', 'x1^2', 'x1x2', 'x2^2', ])
                    df_stander['Z_'] = df_health['Z_']
                    x_train, x_test, z_train, z_test = train_test_split(
                        df_stander[['0', 'X', 'Y', 'x1^2', 'x1x2', 'x2^2']],
                        df_stander["Z_"])
                    model = LinearRegression()
                    model.fit(x_train, z_train)
                    z_predict = model.predict(x_test)
                    intercept_0 = model.intercept_
                    coef1 = model.coef_[0]
                    coef2 = model.coef_[1]
                    coef3 = model.coef_[2]
                    coef4 = model.coef_[3]
                    coef5 = model.coef_[4]
                    coef6 = model.coef_[5]

                    # x_range = np.linspace(df_health['X_'].min(), df_health['X_'].max(), 50)
                    # y_range = np.linspace(df_health['Y_'].min(), df_health['Y_'].max(), 50)
                    # x_grid, y_grid = np.meshgrid(x_range, y_range)
                    # z_grid = (intercept_0 +
                    #           coef1
                    #           + coef2 * x_grid
                    #           + coef3 * y_grid
                    #           + coef4 * x_grid**2
                    #           + coef5 * x_grid * y_grid
                    #           + coef6 * y_grid**2)

                    # fig = plt.figure(figsize=(10, 7))
                    # ax = fig.add_subplot(121, projection='3d')
                    # ax1 = fig.add_subplot(122, projection='3d')
                    # rrr = total_health_iter.copy()
                    # rrr2 = np.array(rrr)
                    # # ax.scatter(df_health['X_'], df_health['Y_'], df_health['Z_'] , color='red', label='Original Data')
                    # ax1.scatter(df_health['X_'], df_health['Y_'], rrr2[: , 2 ] , color='blue', label='Health Data')
                    # # Plot the regression plane
                    # ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.5 , color ="blue" , label='Health plan')
                    # #ax1.plot_surface(x_grid, y_grid, rrr2 , alpha=0.5, color='red' )

                    total_new_generate = []
                    total_error_health = 0
                    for_iter_X_health = []
                    for_iter_Y_health = []

                    error_health = []
                    for i in range(len(df_health['X_'])):
                        new_value = intercept_0 + coef1 * df_stander.iloc[i, 0] + coef2 * df_stander.iloc[
                            i, 1] + coef3 * df_stander.iloc[i, 2] + coef4 * df_stander.iloc[i, 3] + coef5 * \
                                    df_stander.iloc[i, 4]
                        + coef6 * df_stander.iloc[i, 5]
                        total_new_generate.append(new_value)
                        error_res = df_stander.iloc[i, -1] - new_value
                        total_error_health += error_res ** 2
                        error_health.append(error_res)
                        for_iter_X_health.append(df_health.loc[i, 'X_'])
                        for_iter_Y_health.append(df_health.loc[i, 'Y_'])

                    iter_each_generate_health.append([for_iter_X_health, for_iter_Y_health, total_new_generate])
                    for_each_generate_health.append([for_iter_X_health, total_new_generate])
                    print("r2_score in Simulation Health for Each Blade : ", r2_score(z_predict, z_test))

                    R2 = r2_score(z_predict, z_test)
                    R2_adj = 1 - (((1 - R2) * (48 - 1)) / (48 - 5 - 1))
                    stat, p_value = shapiro(error_health)
                    dw_statistic = durbin_watson(error_health)
                    var_errors = np.var(np.array(error_health))
                    # class health == 0
                    params_health.append(
                        [intercept_0, coef1, coef2, coef3, coef4, coef5, coef6, R2, locate, 0, iter_alpha, iter_flat,
                         total_error_health, R2_adj, stat, p_value, dw_statistic,var_errors])  ## added total_error_health
                    plt.figure(figsize=(15, 10))

                    total_change = [1, 1]
                    related_x *= total_change[np.random.randint(0, 2)]
                    related_y *= total_change[np.random.randint(0, 2)]
                    related_z *= total_change[np.random.randint(0, 2)]

                    # ---- setup for class health t ----

                    df_health_t = pd.DataFrame()
                    df_health_t['X_'] = new_data_simulated_health_t[:, 0]
                    df_health_t['Y_'] = new_data_simulated_health_t[:, 1]
                    df_health_t['Z_'] = new_data_simulated_health_t[:, 2]

                    poly_health_t = PolynomialFeatures(degree=2)
                    x_poly_health_t = poly_health_t.fit_transform(df_health_t[['X_', 'Y_']])
                    # scaler_health_t = StandardScaler()
                    df_stander_health_t = pd.DataFrame(scaler.transform(x_poly_health_t),
                                                       columns=['0', 'X', 'Y', 'x1^2', 'x1x2', 'x2^2', ])
                    df_stander_health_t['Z_'] = df_health_t['Z_']
                    x_train, x_test, z_train, z_test = train_test_split(
                        df_stander_health_t[['0', 'X', 'Y', 'x1^2', 'x1x2', 'x2^2']],
                        df_stander["Z_"])
                    model_health_t = LinearRegression()
                    model_health_t.fit(x_train, z_train)
                    z_predict_health_t = model_health_t.predict(x_test)
                    intercept_0 = model_health_t.intercept_
                    coef1 = model_health_t.coef_[0]
                    coef2 = model_health_t.coef_[1]
                    coef3 = model_health_t.coef_[2]
                    coef4 = model_health_t.coef_[3]
                    coef5 = model_health_t.coef_[4]
                    coef6 = model_health_t.coef_[5]

                    # x_range = np.linspace(df_health_t['X_'].min(), df_health_t['X_'].max(), 50)
                    # y_range = np.linspace(df_health_t['Y_'].min(), df_health_t['Y_'].max(), 50)
                    # x_grid, y_grid = np.meshgrid(x_range, y_range)
                    # z_grid_void = (intercept_0 +
                    #           coef1
                    #           + coef2 * x_grid
                    #           + coef3 * y_grid
                    #           + coef4 * x_grid**2
                    #           + coef5 * x_grid * y_grid
                    #           + coef6 * y_grid**2)

                    # rrr3 = total_health_iter_t.copy()
                    # rrr3 = np.array(total_health_iter_t)

                    # ax1.scatter(df_health['X_'], df_health['Y_'], rrr3[: , 2 ] , color='r', label='Void')
                    # ax.plot_surface(x_grid, y_grid, z_grid_void , alpha=0.5 , color ="r" , label='void')

                    for_iter_X_health_t = []
                    for_iter_Y_health_t = []
                    total_new_generate_health_t = []
                    total_error_health_t = 0

                    error_health_t = []
                    for i in range(len(df_health['X_'])):
                        new_value = intercept_0 + coef1 * df_stander_health_t.iloc[i, 0] + coef2 * \
                                    df_stander_health_t.iloc[i, 1] + coef3 * df_stander_health_t.iloc[i, 2] + coef4 * \
                                    df_stander_health_t.iloc[i, 3] + coef5 * df_stander_health_t.iloc[i, 4]
                        + coef6 * df_stander_health_t.iloc[i, 5]
                        total_new_generate_health_t.append(new_value)
                        error_res_t = df_stander_health_t.iloc[i, -1] - new_value
                        total_error_health_t += error_res_t ** 2
                        error_health_t.append(error_res_t)
                        for_iter_X_health_t.append(df_health_t.loc[i, 'X_'])
                        for_iter_Y_health_t.append(df_health_t.loc[i, 'Y_'])

                    iter_each_generate_health_t.append(
                        [for_iter_X_health_t, for_iter_Y_health_t, total_new_generate_health_t])
                    for_each_generate_health_t.append([for_iter_X_health_t, total_new_generate_health_t])

                    print("r2_score in Simulation Failt  void  for Each Blade : ", r2_score(z_predict_health_t, z_test))
                    R2 = r2_score(z_predict_health_t, z_test)
                    R2_adj = 1 - (((1 - R2) * (48 - 1)) / (48 - 5 - 1))
                    stat, p_value = shapiro(error_health_t)
                    dw_statistic = durbin_watson(error_health_t)
                    var_errors = np.var(np.array(error_health_t))
                    # class health == 0
                    params_health_t.append(
                        [intercept_0, coef1, coef2, coef3, coef4, coef5, coef6, R2, locate, 1, iter_alpha, iter_flat,
                         total_error_health_t, R2_adj, stat, p_value, dw_statistic,var_errors])  ## add total_error_health_t
                    total_change = [1, 1]
                    related_x *= total_change[np.random.randint(0, 2)]
                    related_y *= total_change[np.random.randint(0, 2)]
                    related_z *= total_change[np.random.randint(0, 2)]

                    # ---- setup for class Fault ----
                    df_fault = pd.DataFrame()
                    df_fault['X_'] = new_data_simulated_fault[:, 0]
                    df_fault['Y_'] = new_data_simulated_fault[:, 1]
                    df_fault['Z_'] = new_data_simulated_fault[:, 2]

                    poly_fault = PolynomialFeatures(degree=2)
                    x_poly_fault = poly_fault.fit_transform(df_fault[['X_', 'Y_']])
                    # scaler_fault = StandardScaler()
                    df_stander_fault = pd.DataFrame(scaler.transform(x_poly_fault),
                                                    columns=['0', 'X', 'Y', 'x1^2', 'x1x2', 'x2^2', ])
                    df_stander_fault['Z_'] = df_fault['Z_']
                    x_train_fault, x_test_fault, z_train_fault, z_test_fault = train_test_split(
                        df_stander_fault[['0', 'X', 'Y', 'x1^2', 'x1x2', 'x2^2']], df_stander_fault["Z_"])
                    model_fault = LinearRegression()
                    model_fault.fit(x_train_fault, z_train_fault)
                    z_predict_fault = model_fault.predict(x_test_fault)
                    intercept_0 = model_fault.intercept_
                    coef1 = model_fault.coef_[0]
                    coef2 = model_fault.coef_[1]
                    coef3 = model_fault.coef_[2]
                    coef4 = model_fault.coef_[3]
                    coef5 = model_fault.coef_[4]
                    coef6 = model_fault.coef_[5]

                    # x_range = np.linspace(df_fault['X_'].min(), df_fault['X_'].max(), 50)
                    # y_range = np.linspace(df_fault['Y_'].min(), df_fault['Y_'].max(), 50)
                    # x_grid, y_grid = np.meshgrid(x_range, y_range)
                    # z_grid_void = (intercept_0 +
                    #           coef1
                    #           + coef2 * x_grid
                    #           + coef3 * y_grid
                    #           + coef4 * x_grid**2
                    #           + coef5 * x_grid * y_grid
                    #           + coef6 * y_grid**2)

                    # rrr55 = total_fault_iter.copy()
                    # rrr55 = np.array(rrr55)
                    # # ax.scatter(df_health['X_'], df_health['Y_'], df_health['Z_'] , color='red', label='Original Data')
                    # ax1.scatter(df_health['X_'], df_health['Y_'], rrr55[: , 2 ] , color='orange', label='alpha')
                    # ax.plot_surface(x_grid, y_grid, z_grid_void , alpha=0.5 , color ="orange" , label='alpha')

                    total_new_generate_fault = []
                    total_error_fault = 0
                    for_iter_X_fault = []
                    for_iter_Y_fault = []

                    errors_fault = []
                    for i in range(len(df_fault['X_'])):
                        new_value = intercept_0 + coef1 * df_stander_fault.iloc[i, 0] + coef2 * df_stander_fault.iloc[
                            i, 1] + coef3 * df_stander_fault.iloc[i, 2] + coef4 * df_stander_fault.iloc[i, 3] + coef5 * \
                                    df_stander_fault.iloc[i, 4] + coef6 * df_stander_fault.iloc[i, 5]
                        total_new_generate_fault.append(new_value)
                        error_Fault_res = df_stander_fault.iloc[i, -1] - new_value
                        total_error_fault += error_Fault_res ** 2
                        errors_fault.append(error_Fault_res)
                        for_iter_X_fault.append(df_fault.loc[i, 'X_'])
                        for_iter_Y_fault.append(df_fault.loc[i, 'Y_'])

                    iter_each_generate_fault.append([for_iter_X_fault, for_iter_Y_fault, total_new_generate_fault])
                    for_each_generate_fault.append([for_iter_X_fault, total_new_generate_fault])
                    print("Error in Simulation Fault for Each Blade : ", total_error_fault,
                          " --- r2_score Fault S ---- ", r2_score(z_predict_fault, z_test_fault))
                    R2 = r2_score(z_predict_fault, z_test_fault)
                    R2_adj = 1 - (((1 - R2) * (48 - 1)) / (48 - 5 - 1))
                    stat, p_value = shapiro(errors_fault)
                    dw_statistic = durbin_watson(errors_fault)
                    var_errors = np.var(np.array(errors_fault))
                    params_fault.append(
                        [intercept_0, coef1, coef2, coef3, coef4, coef5, coef6, R2, locate, 1, iter_alpha, iter_flat,
                         total_error_fault, R2_adj, stat, p_value, dw_statistic,var_errors])  ## added total_error_fault
                    total_change = [1, 1]
                    related_x *= total_change[np.random.randint(0, 2)]
                    related_y *= total_change[np.random.randint(0, 2)]
                    related_z *= total_change[np.random.randint(0, 2)]

                    # ---- setup for class Fault T ----

                    df_fault_t = pd.DataFrame()
                    df_fault_t['X_'] = new_data_simulated_fault_t[:, 0]
                    df_fault_t['Y_'] = new_data_simulated_fault_t[:, 1]
                    df_fault_t['Z_'] = new_data_simulated_fault_t[:, 2]

                    poly_fault_t = PolynomialFeatures(degree=2)
                    x_poly_fault_t = poly_fault_t.fit_transform(df_fault_t[['X_', 'Y_']])
                    scaler_fault_t = StandardScaler()
                    df_stander_fault_t = pd.DataFrame(scaler.transform(x_poly_fault_t),
                                                      columns=['0', 'X', 'Y', 'x1^2', 'x1x2', 'x2^2', ])
                    df_stander_fault_t['Z_'] = df_fault_t['Z_']
                    x_train_fault_t, x_test_fault_t, z_train_fault_t, z_test_fault_t = train_test_split(
                        df_stander_fault_t[['0', 'X', 'Y', 'x1^2', 'x1x2', 'x2^2']], df_stander_fault["Z_"])
                    model_fault_t = LinearRegression()
                    model_fault_t.fit(x_train_fault_t, z_train_fault_t)
                    z_predict_fault_t = model_fault_t.predict(x_test_fault_t)
                    intercept_0 = model_fault_t.intercept_
                    coef1 = model_fault_t.coef_[0]
                    coef2 = model_fault_t.coef_[1]
                    coef3 = model_fault_t.coef_[2]
                    coef4 = model_fault_t.coef_[3]
                    coef5 = model_fault_t.coef_[4]
                    coef6 = model_fault_t.coef_[5]

                    # x_range = np.linspace(df_fault_t['X_'].min(), df_fault_t['X_'].max(), 50)
                    # y_range = np.linspace(df_fault_t['Y_'].min(), df_fault_t['Y_'].max(), 50)
                    # x_grid, y_grid = np.meshgrid(x_range, y_range)
                    # z_grid_void = (intercept_0 +
                    #           coef1
                    #           + coef2 * x_grid
                    #           + coef3 * y_grid
                    #           + coef4 * x_grid**2
                    #           + coef5 * x_grid * y_grid
                    #           + coef6 * y_grid**2)

                    # rrr66 = total_fault_t_iter.copy()
                    # rrr66 = np.array(rrr66)
                    # # ax.scatter(df_health['X_'], df_health['Y_'], df_health['Z_'] , color='red', label='Original Data')
                    # ax1.scatter(df_health['X_'], df_health['Y_'], rrr66[: , 2 ] , color='purple', label='alpha + void')
                    # ax.plot_surface(x_grid, y_grid, z_grid_void , alpha=0.5 , color ="purple" , label='alpha + void ')

                    # ax.legend()
                    # ax.grid()
                    # ax.set_xlabel("X Axis")
                    # ax.set_ylabel("Y Axis")
                    # ax.set_zlabel("Z Axis")

                    # custom_z_ticks = [0, 50000 , 100000 , 200000  , 300000 , 350000 ]
                    # custom_z_labels = ['0', '2', '4', '6', '8', '10']
                    # ax.set_zticks(custom_z_ticks)
                    # ax.set_zticklabels(custom_z_labels)

                    # ax1.legend()
                    # ax1.grid()

                    total_new_generate_fault_t = []
                    total_error_fault_t = 0
                    for_iter_X_fault_t = []
                    for_iter_Y_fault_t = []
                    errors_fault_t = []
                    for i in range(len(df_fault['X_'])):
                        new_value = intercept_0 + coef1 * df_stander_fault_t.iloc[i, 0] + coef2 * \
                                    df_stander_fault_t.iloc[i, 1] + coef3 * df_stander_fault_t.iloc[i, 2] + coef4 * \
                                    df_stander_fault_t.iloc[i, 3] + coef5 * df_stander_fault_t.iloc[i, 4] + coef6 * \
                                    df_stander_fault_t.iloc[i, 5]
                        total_new_generate_fault_t.append(new_value)
                        error_Fault_t = df_stander_fault_t.iloc[i, -1] - new_value
                        total_error_fault_t += error_Fault_t ** 2
                        errors_fault_t.append(error_Fault_t)
                        for_iter_X_fault_t.append(df_fault_t.loc[i, 'X_'])
                        for_iter_Y_fault_t.append(df_fault_t.loc[i, 'Y_'])
                    iter_each_generate_fault_t.append(
                        [for_iter_X_fault_t, for_iter_Y_fault_t, total_new_generate_fault_t])
                    for_each_generate_fault_t.append([for_iter_X_fault_t, total_new_generate_fault_t])
                    print("Error in Simulation Fault T  for Each Blade : ", total_error_fault,
                          " --- r2_score Fault T : ", r2_score(z_predict_fault_t, z_test_fault_t))

                    R2 = r2_score(z_predict_fault_t, z_test_fault_t)
                    R2_adj = 1 - (((1 - R2) * (48 - 1)) / (48 - 5 - 1))
                    stat, p_value = shapiro(errors_fault_t)
                    dw_statistic = durbin_watson(errors_fault_t)
                    var_errors = np.var(np.array(errors_fault_t))
                    params_fault_t.append(
                        [intercept_0, coef1, coef2, coef3, coef4, coef5, coef6, R2, locate, 1, iter_alpha, iter_flat,
                         total_error_fault_t, R2_adj, stat, p_value, dw_statistic,var_errors])  ## added total_error_fault_t
                    total_change = [1, 1]
                    related_x *= total_change[np.random.randint(0, 2)]
                    related_y *= total_change[np.random.randint(0, 2)]
                    related_z *= total_change[np.random.randint(0, 2)]

                    print(" ----- for each Blade Simulated ----- ")

        # ----
        params_health = np.array(params_health)
        df_params = pd.DataFrame(
            {'bias': params_health[:, 0], 'betha_0': params_health[:, 1], 'betha_1': params_health[:, 2],
             'betha_2': params_health[:, 3], 'betha_3': params_health[:, 4], 'betha_4': params_health[:, 5],
             'betha_5': params_health[:, 6], 'R2': params_health[:, 7],
             'locate': params_health[:, 8].astype(int), 'class': params_health[:, 9],
             'alpha': params_health[:, 10], 'void': params_health[:, 11], 'SSE': params_health[:, 12],
             'R2_adj': params_health[:, 13], 'stat_error': params_health[:, 14], 'p_value_error': params_health[:, 15],
             'DW': params_health[:, 16] , 'var_errors' :params_health[:, 17] })

        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_HEALTH['mean_bias'].append(df_params['bias'].mean())
        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_HEALTH['std_bias'].append(df_params['bias'].std())

        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_HEALTH['mean_betha_0'].append(df_params['betha_0'].mean())
        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_HEALTH['std_betha_0'].append(df_params['betha_0'].std())

        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_HEALTH['mean_betha_1'].append(df_params['betha_1'].mean())
        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_HEALTH['std_betha_1'].append(df_params['betha_1'].std())

        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_HEALTH['mean_betha_2'].append(df_params['betha_2'].mean())
        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_HEALTH['std_betha_2'].append(df_params['betha_2'].std())

        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_HEALTH['mean_betha_3'].append(df_params['betha_3'].mean())
        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_HEALTH['std_betha_3'].append(df_params['betha_3'].std())

        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_HEALTH['mean_betha_4'].append(df_params['betha_4'].mean())
        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_HEALTH['std_betha_4'].append(df_params['betha_4'].std())

        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_HEALTH['mean_betha_5'].append(df_params['betha_5'].mean())
        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_HEALTH['std_betha_5'].append(df_params['betha_5'].std())
        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_HEALTH['mean_error'].append(df_params['SSE'].mean())
        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_HEALTH['std_error'].append(df_params['SSE'].std())

        # -------------------------------------------------------------------------------------------

        params_health_t = np.array(params_health_t)
        df_params_health_t = pd.DataFrame(
            {'bias': params_health_t[:, 0], 'betha_0': params_health_t[:, 1], 'betha_1': params_health_t[:, 2],
             'betha_2': params_health_t[:, 3], 'betha_3': params_health_t[:, 4], 'betha_4': params_health_t[:, 5],
             'betha_5': params_health[:, 6], 'R2': params_health_t[:, 7],
             'locate': params_health_t[:, 8].astype(int), 'class': params_health_t[:, 9],
             'alpha': params_health_t[:, 10], 'void': params_health_t[:, 11], 'SSE': params_health_t[:, 12],
             'R2_adj': params_health_t[:, 13], 'stat_error': params_health_t[:, 14],
             'p_value_error': params_health_t[:, 15], 'DW': params_health_t[:, 16],'var_errors' :params_health_t[:, 17]})

        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_HEALTH_T['mean_bias'].append(df_params_health_t['bias'].mean())
        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_HEALTH_T['std_bias'].append(df_params_health_t['bias'].std())

        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_HEALTH_T['mean_betha_0'].append(df_params_health_t['betha_0'].mean())
        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_HEALTH_T['std_betha_0'].append(df_params_health_t['betha_0'].std())

        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_HEALTH_T['mean_betha_1'].append(df_params_health_t['betha_1'].mean())
        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_HEALTH_T['std_betha_1'].append(df_params_health_t['betha_1'].std())

        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_HEALTH_T['mean_betha_2'].append(df_params_health_t['betha_2'].mean())
        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_HEALTH_T['std_betha_2'].append(df_params_health_t['betha_2'].std())

        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_HEALTH_T['mean_betha_3'].append(df_params_health_t['betha_3'].mean())
        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_HEALTH_T['std_betha_3'].append(df_params_health_t['betha_3'].std())

        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_HEALTH_T['mean_betha_4'].append(df_params_health_t['betha_4'].mean())
        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_HEALTH_T['std_betha_4'].append(df_params_health_t['betha_4'].std())

        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_HEALTH_T['mean_betha_5'].append(df_params_health_t['betha_5'].mean())
        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_HEALTH_T['std_betha_5'].append(df_params_health_t['betha_5'].std())
        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_HEALTH_T['mean_error'].append(df_params_health_t['SSE'].mean())
        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_HEALTH_T['std_error'].append(df_params_health_t['SSE'].std())

        # --------------------------------------------------------------------------------------------
        params_fault = np.array(params_fault)
        df_params_fault = pd.DataFrame(
            {'bias': params_fault[:, 0], 'betha_0': params_fault[:, 1], 'betha_1': params_fault[:, 2],
             'betha_2': params_fault[:, 3], 'betha_3': params_fault[:, 4], 'betha_4': params_fault[:, 5],
             'betha_5': params_fault[:, 6], 'R2': params_fault[:, 7],
             'locate': params_fault[:, 8].astype(int),
             'class': params_fault[:, 9],
             'alpha': params_fault[:, 10], 'void': params_fault[:, 11], 'SSE': params_fault[:, 12],
             'R2_adj': params_fault[:, 13], 'stat_error': params_fault[:, 14], 'p_value_error': params_fault[:, 15],
             'DW': params_fault[:, 16],'var_errors' :params_fault[:, 17]})

        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_FAULT['mean_bias'].append(df_params_fault['bias'].mean())
        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_FAULT['std_bias'].append(df_params_fault['bias'].std())

        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_FAULT['mean_betha_0'].append(df_params_fault['betha_0'].mean())
        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_FAULT['std_betha_0'].append(df_params_fault['betha_0'].std())

        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_FAULT['mean_betha_1'].append(df_params_fault['betha_1'].mean())
        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_FAULT['std_betha_1'].append(df_params_fault['betha_1'].std())

        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_FAULT['mean_betha_2'].append(df_params_fault['betha_2'].mean())
        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_FAULT['std_betha_2'].append(df_params_fault['betha_2'].std())

        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_FAULT['mean_betha_3'].append(df_params_fault['betha_3'].mean())
        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_FAULT['std_betha_3'].append(df_params_fault['betha_3'].std())

        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_FAULT['mean_betha_4'].append(df_params_fault['betha_4'].mean())
        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_FAULT['std_betha_4'].append(df_params_fault['betha_4'].std())

        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_FAULT['mean_betha_5'].append(df_params_fault['betha_5'].mean())
        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_FAULT['std_betha_5'].append(df_params_fault['betha_5'].std())

        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_FAULT['mean_error'].append(df_params_fault['SSE'].mean())
        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_FAULT['std_error'].append(df_params_fault['SSE'].std())

        params_fault_t = np.array(params_fault_t)
        df_params_fault_t = pd.DataFrame(
            {'bias': params_fault_t[:, 0], 'betha_0': params_fault_t[:, 1], 'betha_1': params_fault_t[:, 2],
             'betha_2': params_fault_t[:, 3], 'betha_3': params_fault_t[:, 4], 'betha_4': params_fault_t[:, 5],
             'betha_5': params_fault_t[:, 6], 'R2': params_fault_t[:, 7],
             'locate': params_fault_t[:, 8].astype(int),
             'class': params_fault_t[:, 9],
             'alpha': params_fault_t[:, 10], 'void': params_fault_t[:, 11], 'SSE': params_fault_t[:, 12],
             'R2_adj': params_fault_t[:, 13], 'stat_error': params_fault_t[:, 14],
             'p_value_error': params_fault_t[:, 15], 'DW': params_fault_t[:, 16],'var_errors' :params_fault_t[:, 17]})

        # check_columns = ['bias' , 'betha_0' , 'betha_1' , 'betha_2' , 'betha_3' , 'betha_4' , 'betha_5' ]
        # for element in check_columns :
        #   fig = plt.figure(figsize = (10 , 10 ))
        #   ax0 = fig.add_subplot(111 )
        #   sns.kdeplot( df_params[element]  , label=f'mean_health - {element} coef'  , shade=True )
        #   sns.kdeplot(df_params_fault[element] , label=f'mean_fault_alpha - {element} coef '  , shade=True  )
        #   sns.kdeplot(df_params_fault_t[element] , label=f'mean_fault-alpha + void- {element} coef ' ,  shade=True  )
        #   sns.kdeplot(df_params_health_t[element] , label=f'mean_fault-void- {element} coef ' ,  shade=True  )
        #   plt.legend()
        #   plt.title(f"For This  alpha : {iter_alpha} and void : {iter_flat / 10 }  ")
        #   plt.grid()
        #   plt.show()

        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_FAULT_T['mean_bias'].append(df_params_fault_t['bias'].mean())
        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_FAULT_T['std_bias'].append(df_params_fault_t['bias'].std())

        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_FAULT_T['mean_betha_0'].append(df_params_fault_t['betha_0'].mean())
        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_FAULT_T['std_betha_0'].append(df_params_fault_t['betha_0'].std())

        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_FAULT_T['mean_betha_1'].append(df_params_fault_t['betha_1'].mean())
        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_FAULT_T['std_betha_1'].append(df_params_fault_t['betha_1'].std())

        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_FAULT_T['mean_betha_2'].append(df_params_fault_t['betha_2'].mean())
        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_FAULT_T['std_betha_2'].append(df_params_fault_t['betha_2'].std())

        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_FAULT_T['mean_betha_3'].append(df_params_fault_t['betha_3'].mean())
        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_FAULT_T['std_betha_3'].append(df_params_fault_t['betha_3'].std())

        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_FAULT_T['mean_betha_4'].append(df_params_fault_t['betha_4'].mean())
        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_FAULT_T['std_betha_4'].append(df_params_fault_t['betha_4'].std())

        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_FAULT_T['mean_betha_5'].append(df_params_fault_t['betha_5'].mean())
        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_FAULT_T['std_betha_5'].append(df_params_fault_t['betha_5'].std())

        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_FAULT_T['mean_error'].append(df_params_fault_t['SSE'].mean())
        MEAN_STD_FOR_EACH_ITER_ALPHA_FLAP_FAULT_T['std_error'].append(df_params_fault_t['SSE'].std())

        concat_2_df_health_fault = pd.concat([df_params, df_params_health_t, df_params_fault, df_params_fault_t],
                                             axis=0)

        df_Data = concat_2_df_health_fault[
            (concat_2_df_health_fault["p_value_error"] > 0.05) & (concat_2_df_health_fault["DW"] > 1.5) & (
                        concat_2_df_health_fault["DW"] < 2.5)]
        df_Data["SST"] = df_Data["SSE"] / (1 - df_Data["R2"])
        df_Data["MSR"] = (df_Data["SST"] - df_Data["SSE"]) / 5
        df_Data["MSE"] = df_Data["SSE"] / (48 - 5 - 1)
        df_Data["F"] = df_Data["MSR"] / df_Data["MSE"]
        df_Data['p_value_f'] = f.sf(df_Data["F"], 5, 48 - 5 - 1)
        df_Data.drop(['betha_0'], axis=1, inplace=True)
        df_Data.to_csv(f'{store_path}/alpha_{iter_alpha}void-{iter_flat}.csv', index=False)

        # --- learning
        # plt.figure(figsize= (10 , 10 ) )
        # ax = sns.heatmap(df_corr , annot=True )
        # print(df_corr)
        # plt.show()
        # total_simulation['alpha'].append(iter_alpha)
        # total_simulation['flap'].append(iter_flat)
        # total_simulation['bias_correlation'].append(df_corr['class']['bias'])
        # total_simulation['betha_0_correlation'].append(df_corr['class']['betha_0'])
        # total_simulation['betha_1_correlation'].append(df_corr['class']['betha_1'])
        # total_simulation['betha_2_correlation'].append(df_corr['class']['betha_2'])
        # total_simulation['betha_3_correlation'].append(df_corr['class']['betha_3'])
        # total_simulation['betha_4_correlation'].append(df_corr['class']['betha_4'])
        # total_simulation['betha_5_correlation'].append(df_corr['class']['betha_5'])

        # df2_health  =  concat_2_df_health_fault[concat_2_df_health_fault['class'] == 0 ]
        # df2_fault   =  concat_2_df_health_fault[concat_2_df_health_fault['class'] == 1 ]
        # df2_fault_t =  concat_2_df_health_fault[concat_2_df_health_fault['class'] == 2 ]

        # stander_u = StandardScaler()
        # stand_df_with_out = concat_2_df_health_fault.drop(['class' , 'alpha' , 'flap'] , axis = 1 )
        # stand_df_with_out = pd.DataFrame(stander_u.fit_transform(stand_df_with_out )  , columns = stand_df_with_out.columns )
        # x_u_train , x_u_test , y_u_train , y_u_test = train_test_split(stand_df_with_out , concat_2_df_health_fault['class'] , random_state=42 , test_size=0.3  )
        # obj_gaunb = GaussianNB()
        # obj_gaunb.fit(x_u_train , y_u_train )
        # pre_nb = obj_gaunb.predict(x_u_test)
        # accuray_nav = accuracy_score(pre_nb ,y_u_test )
        # print(classification_report(pre_nb ,y_u_test ) )
        # # plt.figure(figsize = (5 ,5 ))
        # # cm = confusion_matrix(pre_nb ,y_u_test , normalize='pred')
        # # disp = ConfusionMatrixDisplay(confusion_matrix=cm )
        # # disp = disp.plot(cmap=plt.cm.Blues,values_format='g')
        # # plt.title("Nave Bays Confusion Matrix ")
        # # plt.show()
        # print(f"accuracy Nave Bays : {accuray_nav}")
        # total_simulation['accuracy_nav'].append(accuray_nav)

        # ---------------------------------------------------------
        # rfc=RandomForestClassifier(random_state=42)
        # param_grid = {
        #     'n_estimators': [200, 300 ],
        #     'max_features': ['auto', 'sqrt', 'log2'],
        #     'max_depth' : [4,5,6,7,8],
        #     'criterion' :['gini', 'entropy']
        # }

        # CV_rfc = GridSearchCV(estimator=rfc, param_grid = param_grid,
        #                       cv = 3, n_jobs = -1, verbose = 2)
        # CV_rfc.fit(x_u_train, y_u_train)
        # best_params_for_random_forest = CV_rfc.best_params_
        # pre_random = CV_rfc.predict(x_u_test)
        # accuracy_random_forest = accuracy_score(pre_random ,y_u_test )
        # print(classification_report(pre_nb ,y_u_test ) )
        # # plt.figure(figsize = (5 , 5 ))
        # # cm = confusion_matrix(pre_random ,y_u_test , normalize='pred')
        # # disp = ConfusionMatrixDisplay(confusion_matrix=cm )
        # # disp = disp.plot(cmap=plt.cm.Blues,values_format='g')
        # # plt.title("Random Forest Confusion Matrix ")
        # # plt.show()
        # print(f"accuracy accuracy_random_forest  : {accuracy_random_forest}")
        # total_simulation['accuracy_random_forest'].append(accuracy_random_forest)
        # total_simulation['n_estimators_RF'].append(best_params_for_random_forest['n_estimators'])
        # total_simulation['max_features_RF'].append(best_params_for_random_forest['max_features'])
        # total_simulation['max_depth_RF'].append(best_params_for_random_forest['max_depth'])
        # total_simulation['criterion_RF'].append(best_params_for_random_forest['criterion'])

        # # ---------------------------------------------------------
        # extra_classifier = ExtraTreesClassifier()
        # extra_classifier.fit(x_u_train , y_u_train )
        # predict_extra = extra_classifier.predict(x_u_test)
        # total_simulation['accuracy_extra_classifier'].append(accuracy_score(predict_extra ,y_u_test ) )

        # # # --------------------------------------------------------
        # param_grid = {'C':[1,10,100],'gamma':[1,0.1,0.001], 'kernel':['linear','rbf']}
        # grid = GridSearchCV(SVC(),param_grid,refit = True, verbose=2)
        # grid.fit(x_u_train, y_u_train)
        # best_params_for_svc = grid.best_params_
        # model_svc = SVC()
        # svc_predict = grid.predict(x_u_test)
        # accuracy_svc = accuracy_score(svc_predict ,y_u_test )
        # # plt.figure(figsize = (5 , 5 ))
        # # cm = confusion_matrix(svc_predict ,y_u_test, normalize='pred')
        # # disp = ConfusionMatrixDisplay(confusion_matrix=cm )
        # # disp = disp.plot(cmap=plt.cm.Blues,values_format='g')
        # # plt.title("SVM(SVC) Confusion Matrix ")
        # # plt.show()
        # total_simulation['accuracy_svc_'].append(accuracy_svc)
        # total_simulation['C_svc'].append(best_params_for_svc['C'])
        # total_simulation['kernel_svc'].append(best_params_for_svc['kernel'])
        # total_simulation['gamma_svc'].append(best_params_for_svc['gamma'])

        # # ----------------------------------------------------------

        # new_y_u_train_cat = to_categorical(y_u_train , 2 )
        # new_y_u_test_cat  = to_categorical(y_u_test , 2 )

        # new_y_u_train_cat = to_categorical(y_u_train , 4 )
        # new_y_u_test_cat  = to_categorical(y_u_test , 4 )

        # model_learning = Sequential([
        #     Dense(256 , activation='relu' ) ,
        #     BatchNormalization() ,
        #     Dropout(0.2) ,
        #     Dense(128 , activation='relu' ,  ) ,
        #     BatchNormalization() ,
        #     Dropout(0.2) ,
        #     Dense(56 , activation='relu' ,  ) ,
        #     BatchNormalization() ,
        #     Dropout(0.2) ,
        #     Dense(32 , activation='relu' ,  ) ,
        #     BatchNormalization() ,
        #     Dropout(0.2) ,
        #     Dense(10 , activation= 'relu' ) ,
        #     Dense(2 , activation='softmax' ) ,
        # ])

        # model_learning.compile('adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy' ] )
        # model_learning_info = model_learning.fit(x_u_train , new_y_u_train_cat , batch_size= 50  , epochs = 50 , validation_split = 0.2  )
        # plt.plot(range(50) , model_learning_info.history['acc'] , label='acc' )
        # plt.plot(range(50) , model_learning_info.history['val_acc'] , label='val_acc' )
        # plt.legend()
        # plt.grid()
        # plt.show()

        # total_simulation['ann'].append(model_learning.evaluate(x_u_test , new_y_u_test_cat)[1])

        print(" ----------------- //////// End This Iter alpha and Flap //////// ------------------------- ")

    # fig = plt.figure(figsize = (12 , 5 ))
    # ax1 = fig.add_subplot(1 , 4 , 1 , projection = '3d' )
    # ax2 = fig.add_subplot(1 , 4 , 2 , projection = '3d'  )
    # ax3 = fig.add_subplot(1 , 4 , 3 , projection = '3d'  )
    # ax4 = fig.add_subplot(1 , 4 , 4 , projection = '3d'  )

    # iter_each_generate_health   = np.array(iter_each_generate_health)
    # iter_each_generate_health_t = np.array(iter_each_generate_health_t)
    # iter_each_generate_fault    = np.array(iter_each_generate_fault)
    # iter_each_generate_fault_t  = np.array(iter_each_generate_fault_t)

    # ax1.scatter3D(iter_each_generate_health[ : , 0]   , iter_each_generate_health[ : , 1] , iter_each_generate_health[ : , 2 ] , label = 'health'  )
    # ax2.scatter3D(iter_each_generate_health_t[ : , 0] , iter_each_generate_health[: , 1] ,  iter_each_generate_health_t[ : , 2 ] , label = 'fault void'  )
    # ax3.scatter3D(iter_each_generate_fault[ : , 0]    , iter_each_generate_health[: , 1] , iter_each_generate_fault[ : , 2 ] , label = 'fault alpha '  )
    # ax4.scatter3D(iter_each_generate_fault_t[ : , 0]  , iter_each_generate_health[: , 1 ] , iter_each_generate_fault_t[ : , 2 ] , label = 'fault alpha + void '  )

    # ax1.grid()
    # ax1.set_title(" health data ")

    # ax2.grid()
    # ax2.set_title("fault void ")

    # ax3.grid()
    # ax3.set_title("fault alpha ")

    # ax4.grid()
    # ax4.set_title("alpha + void ")
    # plt.show()

    # fig1 = plt.figure(figsize = (12 , 8 ))

    # ax5 = fig1.add_subplot(4 , 4 , 1   )
    # ax6 = fig1.add_subplot(4 , 4 , 2   )
    # ax7 = fig1.add_subplot(4 , 4 , 3   )
    # ax8 = fig1.add_subplot(4 , 4 , 4   )

    # ax5.scatter(iter_each_generate_health[ : , 0] , iter_each_generate_health[ : , 2 ] , label = 'health' , c='r' )
    # ax6.scatter(iter_each_generate_health_t[ : , 0] , iter_each_generate_health_t[ : , 2 ] , label = 'fault void' , c='aqua' )
    # ax7.scatter(iter_each_generate_fault[ : , 0] , iter_each_generate_fault[ : , 2 ] , label = 'fault alpha' , c='b')
    # ax8.scatter(iter_each_generate_fault_t[ : , 0] , iter_each_generate_fault_t[ : , 2 ] , label = 'fault alpha + void' , c='g' )

    # ax5.set_title("health data ")
    # ax5.grid()

    # ax6.set_title("fault void ")
    # ax6.grid()

    # ax7.set_title(f"fault alpha alpha : {iter_alpha}")
    # ax7.grid()

    # ax8.set_title(f"fault alpha  + void : {iter_alpha} ")
    # ax8.grid()

    # plt.show()

    # fig2 = plt.figure(figsize = (12 , 8 ))
    # ax9  = fig2.add_subplot(4 , 4 , 1   )
    # ax10 = fig2.add_subplot(4 , 4 , 2   )
    # ax11 = fig2.add_subplot(4 , 4 , 3   )
    # ax12 = fig2.add_subplot(4 , 4 , 4   )

    # ax9.scatter(iter_each_generate_health[ : , 0] , iter_each_generate_health[ : , 1 ] , label = 'health' , c='r' )
    # ax10.scatter(iter_each_generate_health_t[ : , 0] , iter_each_generate_health_t[ : , 1 ] , label = 'fault void' , c='aqua' )
    # ax11.scatter(iter_each_generate_fault[ : , 0] , iter_each_generate_fault[ : , 1 ] , label = 'fault alpha' , c='b')
    # ax12.scatter(iter_each_generate_fault_t[ : , 0] , iter_each_generate_fault_t[ : , 1 ] , label = 'fault alpha + void' , c='g' )

    # ax9.set_title("health data ")
    # ax9.grid()

    # ax10.set_title("fault void ")
    # ax10.grid()

    # ax11.set_title(f"fault alpha alpha : {iter_alpha}")
    # ax11.grid()

    # ax12.set_title(f"fault alpha  + void : {iter_alpha} ")
    # ax12.grid()

    # plt.show()
