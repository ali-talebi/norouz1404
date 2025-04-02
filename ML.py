import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from library import  *

Total_Df = pd.DataFrame()

root_data = 'DATA_GENERATED/fix_orthogonal'

columns_ = ['bias','coef_1','coef_2','coef_3','coef_4','coef_5', 'SSE' ,'DW','R2','R2_adj','p_value_f','p_value_error','class' , 'alpha' , 'void']

Total_Df = pd.DataFrame(columns=columns_)
for index, name in enumerate(os.listdir(root_data)):
    df = pd.read_csv(f'{root_data}/{name}')
    Total_Df = pd.concat([Total_Df, df], axis=0)
    del df

columns2 = ['bias','coef_1','coef_2','coef_3','coef_4','coef_5' , 'class']
Total_Df['class'] = Total_Df['class'].apply(lambda x : 0 if x == 0 else 1 )
df2 = Total_Df[columns2].reset_index().drop(['index'] , axis = 1 )
print(df2)
print("Classes : " , df2['class'].value_counts())
y_label = df2[['class']]
df2.drop(['class'] , axis = 1 , inplace=True )
x_trains , x_test , y_train , y_test = train_test_split(df2,y_label , test_size=0.5 , random_state= 42 )

random_classifier = RandomForestClassifier()
random_classifier.fit(x_trains,y_train)
y_pre = random_classifier.predict(x_test)
print("accuracy Random Forest",accuracy_score(y_pre,y_test))



print("*------------------*-----------------------*-------------*-------------------*")

rfc = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [200, 300],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [4, 5, 6],
    'criterion': ['gini', 'entropy']
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid,
                      cv=3, n_jobs=-1, verbose=2)
CV_rfc.fit(x_trains,y_train)
best_params_for_random_forest = CV_rfc.best_params_
pre_random = CV_rfc.predict(x_test)
accuracy_random_forest = accuracy_score(pre_random, y_test)
print("accuracy_random_forest : " , accuracy_random_forest )

confusion_matrix = metrics.confusion_matrix(y_test, pre_random)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])
cm_display.plot()
plt.title(f"WITH DATA ORTHOGONAL , Random Forest Classifier-Accuracy = {accuracy_random_forest} ")
plt.show()

print("*------------------*-----------------------*-------------*-------------------*")
print("Y trains shape : " , y_train.shape )
print("Y train : " , y_train )

print("*------------------*-----------------------*-------------*-------------------*")
extra_classifier = ExtraTreesClassifier()
extra_classifier.fit(x_trains,y_train)
predict_extra = extra_classifier.predict(x_test)
print("accuracy_Extra" , accuracy_score(predict_extra, y_test) )
acc_e = accuracy_score(predict_extra, y_test)


confusion_matrix = metrics.confusion_matrix(y_test, predict_extra)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])
cm_display.plot()
plt.title(f"WITH DATA ORTHOGONAL , Extra Classifier - Accuracy = {acc_e}")
plt.show()


print("*------------------*-----------------------*-------------*-------------------*")
obj_gaunb = GaussianNB()
obj_gaunb.fit(x_trains,y_train)
pre_nb = obj_gaunb.predict(x_test)
accuray_nav = accuracy_score(pre_nb, y_test)
print(f"accuracy Nave Bays : {accuray_nav}")

confusion_matrix = metrics.confusion_matrix(y_test, pre_nb)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])
cm_display.plot()
plt.title(f"WITH DATA ORTHOGONAL , Naive Bayes - Accuracy={accuray_nav}")
plt.show()


print("*------------------*-----------------------*-------------*-------------------*")

param_grid = {'C': [1, 10, 100, 1000], 'gamma': [1, 0.1, 0.001, 0.0001], 'kernel': ['linear', 'rbf']}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid.fit(x_trains, y_train)
best_params_for_svc = grid.best_params_
model_svc = SVC()
svc_predict = grid.predict(x_test)
accuracy_svc = accuracy_score(svc_predict, y_test)
print("accuracy_svc : " , accuracy_svc )


confusion_matrix = metrics.confusion_matrix(y_test, svc_predict)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])
cm_display.plot()
plt.title(f"WITH DATA ORTHOGONAL , Support Vector Machine - Accuracy = {accuracy_svc} ")
plt.show()
print("*------------------*-----------------------*-------------*-------------------*")

y_train2 = to_categorical(y_train , 2 )
y_test2 = to_categorical(y_test , 2 )


model_learning = Sequential([
    Dense(256, activation='relu', input_shape=x_trains.shape[1:]),
    BatchNormalization(),
    Dropout(0.2),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(56, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(10, activation='relu'),
    Dense(2 , activation='softmax'),
])

model_learning.compile('adam', loss='binary_crossentropy', metrics=['acc'])
model_learning_info = model_learning.fit(x_trains, y_train2, epochs=50,
                                         validation_data=[x_test, y_test2])
MLP_Pre = model_learning.predict(x_test)
mlp_pre = np.array([ np.argmax(i) for i in MLP_Pre ])

accuracy_mlp = model_learning.evaluate(x_test , y_test2)[1]
plt.plot(range(50), model_learning_info.history['acc'], label='acc')
plt.plot(range(50), model_learning_info.history['val_acc'], label='val_acc')
plt.legend()
plt.grid()
plt.show()


confusion_matrix = metrics.confusion_matrix(y_test, mlp_pre)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])
cm_display.plot()
plt.title(f"WITH DATA ORTHOGONAL  , MLP - Accuracy = {accuracy_mlp} ")
plt.show()