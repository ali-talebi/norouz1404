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
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import cm
from scipy.stats import shapiro
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import f

# pip freeze > requirements.txt

root_path = 'Total_Data_Simulation'
store_path = 'DATA_GENERATED'

total_file_link = []
for i in os.listdir(root_path):
    total_file_link.append(root_path + f'/{i}')

pd.read_csv(total_file_link[2])

total_alpha_content = [i / 1000 for i in range(1, 100)]
total_flat_content  = [i / 10   for i in range(1, 5)]
total_location = list(range(5, 15))

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
plt.scatter(total_points[:, 0], total_points[:, 1], label="target")
plt.legend()
plt.title("Blade Shape - Picture 1.7 page 7 document volume 2 ")
plt.xlabel(" X (in) ")
plt.ylabel(" Y (in) ")
plt.grid()
plt.show()

p = pd.DataFrame(total_points, columns=['X', 'Y', 'Z'])

main_data_frame = pd.DataFrame()

p.tail(10)

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
        main_list = []
        for iter_link in total_file_link:
            df_table_change = pd.read_csv(iter_link)
            for locate in total_location:
                related_x = -1
                related_y = -1
                related_z = -1
                for iteration in range(10):
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
                    # # ax3 = fig.add_subplot(1 , 6 , 4 )
                    # # ax4 = fig.add_subplot(1 , 6 , 5 )
                    # # ax5 = fig.add_subplot(1 , 6 , 6 )
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
                    #
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




                    X_B = pd.DataFrame()
                    X_B['x_p'] = 2 *( (df_health['X_'] - 65 ) / (330-65)) - 1
                    X_B['y_p'] = 2 *( (df_health['Y_'] + 17.5 ) / (-3+17.5)) - 1


                    X_B['one'] = 1
                    X_B['p1_x_p'] = X_B['x_p']
                    X_B['q1_y_p'] = X_B['y_p']
                    X_B['p2_x_p'] = (3 *( (X_B['x_p']) ** 2 ) -1 ) / 2

                    # X_B['q0_y_p'] = 1
                    X_B['p1_x_q1_y'] = X_B['x_p'] * X_B['y_p']
                    X_B['q2_y_p'] = (3 *( (X_B['y_p']) ** 2 ) -1 ) / 2

                    X_B.drop(['x_p','y_p'] , axis = 1 , inplace = True )
                    x_train , x_test , y_train , y_test = train_test_split(X_B,df_health['Z_'])
                    model = LinearRegression()
                    model.fit(x_train, y_train )
                    z_predict = model.predict(x_test)
                    intercept_0 = model.intercept_
                    coef1 = model.coef_[0]
                    coef2 = model.coef_[1]
                    coef3 = model.coef_[2]
                    coef4 = model.coef_[3]
                    coef5 = model.coef_[4]
                    coef6 = model.coef_[5]

                    # A_coef =  (( X_B.T @ X_B ) ** -1 ) @ X_B.T @ df_health['Z_']
                    SSE_HEALTH = 0
                    TOTAL_ERROR_HEALTH = []
                    z_predict_health   = []
                    result = pd.DataFrame(model.predict(X_B), columns=['result'])
                    # print("RESULT : " , result )
                    reseduals = df_health['Z_'] - result['result']
                    print('reseduals : ' , reseduals )
                    reseduals2 = [i for i in reseduals ]
                    R2 = r2_score(result, df_health['Z_'] )
                    R2_adj = 1 - (((1 - R2) * (48 - 1)) / (48 - 5 - 1))
                    stat, p_value = shapiro(reseduals2)
                    dw_statistic = durbin_watson(reseduals2)
                    print(f"DW : {dw_statistic} - R2 : {R2} , p_value : {p_value} ")
                    for i in reseduals2 :
                        SSE_HEALTH += i**2
                    SST_HEALTH = SSE_HEALTH / (1 - R2 )
                    MSR_HEALTH = ( SST_HEALTH - SSE_HEALTH ) / 5
                    MSE_HEALTH = SSE_HEALTH / (48 - 5 - 1)
                    F_HEALTH = MSR_HEALTH / MSE_HEALTH
                    p_value_f = f.sf(F_HEALTH, 5, 48 - 5 - 1)
                    main_list.append([intercept_0,coef1,coef2,coef3,coef4,coef5 , SSE_HEALTH ,dw_statistic,R2,R2_adj,p_value_f,p_value , 0 , iter_alpha , iter_flat ])
                    ###--------------------------------------------------###
                    SSE_HEALTH_T = 0
                    df_health_t = pd.DataFrame()
                    df_health_t['X_'] = new_data_simulated_health_t[:, 0]
                    df_health_t['Y_'] = new_data_simulated_health_t[:, 1]
                    df_health_t['Z_'] = new_data_simulated_health_t[:, 2]

                    X_B_t = pd.DataFrame()
                    X_B_t['x_p'] = 2 *( (df_health_t['X_'] - 65 ) / (330-65)) - 1
                    X_B_t['y_p'] = 2 *( (df_health_t['Y_'] + 17.5 ) / (-3+17.5)) - 1


                    X_B_t['one'] = 1
                    X_B_t['p1_x_p'] = X_B_t['x_p']
                    X_B_t['q1_y_p'] = X_B_t['y_p']
                    X_B_t['p2_x_p'] = (3 *( (X_B_t['x_p']) ** 2 ) -1 ) / 2

                    # X_B['q0_y_p'] = 1
                    X_B_t['p1_x_q1_y'] = X_B_t['x_p'] * X_B_t['y_p']
                    X_B_t['q2_y_p'] = (3 *( (X_B_t['y_p']) ** 2 ) -1 ) / 2

                    X_B_t.drop(['x_p','y_p'] , axis = 1 , inplace = True )
                    x_train_h , x_test_h , y_train_h , y_test_h = train_test_split(X_B_t,df_health_t['Z_'])
                    model = LinearRegression()
                    model.fit(x_train_h, y_train_h )
                    z_predict = model.predict(x_test_h)
                    intercept_0_t = model.intercept_
                    coef1_t = model.coef_[0]
                    coef2_t = model.coef_[1]
                    coef3_t = model.coef_[2]
                    coef4_t = model.coef_[3]
                    coef5_t = model.coef_[4]
                    coef6_t = model.coef_[5]

                    # A_coef =  (( X_B.T @ X_B ) ** -1 ) @ X_B.T @ df_health['Z_']
                    result_t = pd.DataFrame(model.predict(X_B_t), columns=['result'])
                    # print("RESULT : " , result )
                    reseduals = df_health_t['Z_'] - result['result']
                    print('reseduals : ' , reseduals )
                    reseduals2 = [i for i in reseduals ]
                    R2 = r2_score(result_t, df_health_t['Z_'] )
                    R2_adj = 1 - (((1 - R2) * (48 - 1)) / (48 - 5 - 1))
                    stat, p_value = shapiro(reseduals2)
                    dw_statistic = durbin_watson(reseduals2)
                    print(f"DW : {dw_statistic} - R2 : {R2} , p_value : {p_value} ")
                    for i in reseduals2 :
                        SSE_HEALTH_T += i**2

                    SST_HEALTH_T = SSE_HEALTH_T / (1 - R2 )
                    MSR_HEALTH_T = ( SST_HEALTH_T - SSE_HEALTH_T ) / 5
                    MSE_HEALTH_T = SSE_HEALTH_T / (48 - 5 - 1)
                    F_HEALTH_T   = MSR_HEALTH_T / MSE_HEALTH_T
                    p_value_f = f.sf(F_HEALTH_T, 5, 48 - 5 - 1)
                    main_list.append([intercept_0_t,coef1_t,coef2_t,coef3_t,coef4_t,coef5_t ,SSE_HEALTH_T,dw_statistic,R2,R2_adj , p_value_f ,p_value , 1 , iter_alpha , iter_flat ])
                    ###--------------------------------------------------###
                    SSE_FAULT = 0
                    df_fault = pd.DataFrame()
                    df_fault['X_'] = new_data_simulated_fault[:, 0]
                    df_fault['Y_'] = new_data_simulated_fault[:, 1]
                    df_fault['Z_'] = new_data_simulated_fault[:, 2]

                    X_B_t = pd.DataFrame()
                    X_B_t['x_p'] = 2 * ((df_fault['X_'] - 65) / (330 - 65)) - 1
                    X_B_t['y_p'] = 2 * ((df_fault['Y_'] + 17.5) / (-3 + 17.5)) - 1

                    X_B_t['one'] = 1
                    X_B_t['p1_x_p'] = X_B_t['x_p']
                    X_B_t['q1_y_p'] = X_B_t['y_p']
                    X_B_t['p2_x_p'] = (3 * ((X_B_t['x_p']) ** 2) - 1) / 2

                    # X_B['q0_y_p'] = 1
                    X_B_t['p1_x_q1_y'] = X_B_t['x_p'] * X_B_t['y_p']
                    X_B_t['q2_y_p'] = (3 * ((X_B_t['y_p']) ** 2) - 1) / 2

                    X_B_t.drop(['x_p', 'y_p'], axis=1, inplace=True)
                    x_train_h, x_test_h, y_train_h, y_test_h = train_test_split(X_B_t, df_fault['Z_'])
                    model = LinearRegression()
                    model.fit(x_train_h, y_train_h)
                    z_predict = model.predict(x_test_h)
                    intercept_0_t = model.intercept_
                    coef1_t = model.coef_[0]
                    coef2_t = model.coef_[1]
                    coef3_t = model.coef_[2]
                    coef4_t = model.coef_[3]
                    coef5_t = model.coef_[4]
                    coef6_t = model.coef_[5]

                    # A_coef =  (( X_B.T @ X_B ) ** -1 ) @ X_B.T @ df_health['Z_']
                    result_t = pd.DataFrame(model.predict(X_B_t), columns=['result'])
                    # print("RESULT : " , result )
                    reseduals = df_fault['Z_'] - result['result']
                    print('reseduals : ', reseduals)
                    reseduals2 = [i for i in reseduals]
                    R2 = r2_score(result_t, df_fault['Z_'])
                    R2_adj = 1 - (((1 - R2) * (48 - 1)) / (48 - 5 - 1))
                    stat, p_value = shapiro(reseduals2)
                    dw_statistic = durbin_watson(reseduals2)
                    print(f"DW : {dw_statistic} - R2 : {R2} , p_value : {p_value} ")
                    for i in reseduals2 :
                        SSE_FAULT += i**2

                    SST_FAULT    = SSE_FAULT / (1 - R2 )
                    MSR_FAULT = ( SST_FAULT - SSE_FAULT ) / 5
                    MSE_FAULT = SSE_FAULT / (48 - 5 - 1)
                    F_FAULT_T   = MSR_FAULT / MSE_FAULT
                    p_value_f = f.sf(F_FAULT_T, 5, 48 - 5 - 1)

                    main_list.append(
                        [intercept_0_t, coef1_t, coef2_t, coef3_t, coef4_t, coef5_t,SSE_FAULT, dw_statistic, R2, R2_adj , p_value_f , p_value,
                         2, iter_alpha , iter_flat])
                    ###--------------------------------------------------###
                    SSE_FAULT_T = 0
                    df_fault_t = pd.DataFrame()
                    df_fault_t['X_'] = new_data_simulated_fault_t[:, 0]
                    df_fault_t['Y_'] = new_data_simulated_fault_t[:, 1]
                    df_fault_t['Z_'] = new_data_simulated_fault_t[:, 2]

                    X_B_t = pd.DataFrame()
                    X_B_t['x_p'] = 2 * ((df_fault_t['X_'] - 65) / (330 - 65)) - 1
                    X_B_t['y_p'] = 2 * ((df_fault_t['Y_'] + 17.5) / (-3 + 17.5)) - 1

                    X_B_t['one'] = 1
                    X_B_t['p1_x_p'] = X_B_t['x_p']
                    X_B_t['q1_y_p'] = X_B_t['y_p']
                    X_B_t['p2_x_p'] = (3 * ((X_B_t['x_p']) ** 2) - 1) / 2

                    # X_B['q0_y_p'] = 1
                    X_B_t['p1_x_q1_y'] = X_B_t['x_p'] * X_B_t['y_p']
                    X_B_t['q2_y_p'] = (3 * ((X_B_t['y_p']) ** 2) - 1) / 2

                    X_B_t.drop(['x_p', 'y_p'], axis=1, inplace=True)
                    x_train_h, x_test_h, y_train_h, y_test_h = train_test_split(X_B_t, df_fault_t['Z_'])
                    model = LinearRegression()
                    model.fit(x_train_h, y_train_h)
                    z_predict = model.predict(x_test_h)
                    intercept_0_t = model.intercept_
                    coef1_t = model.coef_[0]
                    coef2_t = model.coef_[1]
                    coef3_t = model.coef_[2]
                    coef4_t = model.coef_[3]
                    coef5_t = model.coef_[4]
                    coef6_t = model.coef_[5]

                    # A_coef =  (( X_B.T @ X_B ) ** -1 ) @ X_B.T @ df_health['Z_']
                    result_t = pd.DataFrame(model.predict(X_B_t), columns=['result'])
                    # print("RESULT : " , result )
                    reseduals = df_fault_t['Z_'] - result['result']
                    print('reseduals : ', reseduals)
                    reseduals2 = [i for i in reseduals]
                    R2 = r2_score(result_t, df_fault_t['Z_'])
                    R2_adj = 1 - (((1 - R2) * (48 - 1)) / (48 - 5 - 1))
                    stat, p_value = shapiro(reseduals2)
                    dw_statistic = durbin_watson(reseduals2)
                    print(f"DW : {dw_statistic} - R2 : {R2} , p_value : {p_value} ")
                    for i in reseduals2 :
                        SSE_FAULT_T += i**2

                    SST_FAULT_T    = SSE_FAULT_T / (1 - R2 )
                    MSR_FAULT_T = ( SST_FAULT_T - SSE_FAULT_T ) / 5
                    MSE_FAULT_T = SSE_FAULT_T / (48 - 5 - 1)
                    F_FAULT_T   = MSR_FAULT_T / MSE_FAULT_T
                    p_value_f = f.sf(F_FAULT_T, 5, 48 - 5 - 1)

                    main_list.append(
                        [intercept_0_t, coef1_t, coef2_t, coef3_t, coef4_t, coef5_t , SSE_FAULT_T , dw_statistic, R2, R2_adj,p_value_f, p_value,
                         3 , iter_alpha , iter_flat ])

        main_df = pd.DataFrame(main_list , columns = ['bias','coef_1','coef_2','coef_3','coef_4','coef_5', 'SSE' ,'DW','R2','R2_adj','p_value_f','p_value_error','class' , 'alpha' , 'void'])
        main_df2 = main_df[(main_df['p_value_error'] > 0.05 ) & (main_df['R2'] > 0.9 ) & (main_df['R2_adj'] > 0.9 ) & ((main_df['p_value_f'] < 0.05 ))]
        main_df2.to_csv(f'{store_path}/fix_orthogonal/alpha_{iter_alpha}void-{iter_flat}.csv', index=False)












