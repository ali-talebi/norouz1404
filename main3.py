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

root_path = 'Total_Data_Simulation'
total_file_link = []
for i in os.listdir(root_path):
    total_file_link.append(root_path + f'/{i}')

total_alpha_content = [i / 1000 for i in range(1, 10)]
total_flat_content = [i for i in range(1, 5)]
total_location = list(range(5, 15))

points = []
start_x = 50
start_y = [-2.5, -7.5, -12.5 ]
for i in range(15):
    start_x += 15

    for j in range(3):
        points.append([start_x, start_y[j], 0])

points.append([330, -7.5, 0])
points.append([330, -12.5, 0])
points.append([330, -17.5, 0])
points = np.array(points)
df = pd.DataFrame(points , columns=['X','Y','Z'])
fig = plt.figure(figsize=(10 , 5 ))
ax0 = fig.add_subplot(1 , 6 , 1 , projection='3d' )
ax0.scatter3D(points[: , 0 ] , points[: , 1 ] , points[: , 2 ] )

ax0.set_xlabel("axis x ")
ax0.set_ylabel("axis y ")
ax0.set_zlabel("axis z ")
plt.show()
