import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from pandas.plotting import parallel_coordinates
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.graph_objs as go #visualization
import plotly.offline as py #visualization

from sklearn.svm import SVC
from yellowbrick.classifier import DiscriminationThreshold
import plotly.tools as tls
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.metrics import roc_auc_score,roc_curve

warnings.filterwarnings('ignore')

pd.set_option("display.max_rows", 21, "display.max_columns", 21)
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# Read Excel file
# 상운 dataset = pd.read_csv(r'C:\Users\dltkd\OneDrive\바탕 화면\3학년 1학기\데이터 과학\termproject\WA_Fn-UseC_-Telco-Customer-Churn.csv')
# 수경
dataset = pd.read_csv('C:/WA_Fn-UseC_-Telco-Customer-Churn.csv')
# dataset = pd.read_csv(r'C:\Users\dltkd\OneDrive\바탕 화면\3학년 1학기\데이터 과학\termproject\WA_Fn-UseC_-Telco-Customer-Churn.csv')

# statistical summary of dataset
print("# dataset statistical")
print(dataset.describe(), "\n")
print("# dataset head")
print(dataset.head(5), "\n")
print("# dataset shape")
print(dataset.shape, "\n")
print("# dataset index")
print(dataset.index, "\n")
print("# dataset features")
print(dataset.columns, "\n")

#  ID column because it is meaningless
dataset = dataset.drop(columns='customerID')
# tenure <= 0 is wrong data
dataset.loc[dataset['tenure']<=0, 'tenure']=np.NaN

target_col = ["Churn"]
# categorical columns
cat_cols   = dataset.nunique()[dataset.nunique() < 6].keys().tolist()
cat_cols   = [x for x in cat_cols if x not in target_col]

# Binary columns with 2 values
bin_cols   = dataset.nunique()[dataset.nunique() == 2].keys().tolist()
# Columns more than 2 values
multi_cols = [i for i in cat_cols if i not in bin_cols]

# Label encoding Binary columns
le = LabelEncoder()
for i in bin_cols :
    dataset[i] = le.fit_transform(dataset[i])
for i in multi_cols :
    dataset[i] = le.fit_transform(dataset[i])

# Scaling Numerical columns
cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']
std = StandardScaler()
scaled = std.fit_transform(dataset)
dataset = pd.DataFrame(scaled,columns=cols)

# correlation
correlation = dataset.corr()
# tick labels
matrix_cols = correlation.columns.tolist()
# convert to array
corr_array  = np.array(correlation)

# Plotting
trace = go.Heatmap(z = corr_array,
                   x = matrix_cols,
                   y = matrix_cols,
                   colorscale = "Viridis",
                   colorbar   = dict(title = "Pearson Correlation coefficient",
                                     titleside = "right"
                                    ) ,
                  )

layout = go.Layout(dict(title = "Correlation Matrix for variables",
                        autosize = False,
                        height  = 720,
                        width   = 800,
                        margin  = dict(r = 0 ,l = 210,
                                       t = 25,b = 210,
                                      ),
                        yaxis   = dict(tickfont = dict(size = 9)),
                        xaxis   = dict(tickfont = dict(size = 9))
                       )
                  )

data = [trace]
fig = go.Figure(data=data,layout=layout)
py.iplot(fig)

# drop the columns which correlation is under 0.15
index=[]
for i in range(20):
    if abs(corr_array[19][i]) < 0.15:
        index.append(matrix_cols[i])

print("# columns which have less correlation(threshold = 0.15)")
print(index)
print("")

for i in range (len(index)):
    dataset = dataset.drop(columns=index[i])

print("# Handle missing values")
# print is nan
print("# number of null values ")
print(dataset.isnull().sum(), "\n")

print("# 1. drop the rows which have null value")
dataset_drop = pd.DataFrame(dataset.dropna(how='any'), columns=dataset.columns)
print(dataset_drop.isnull().sum())

print("# 2. fill missing values with k-means clustering")
# test dataset 만들기
test_x = dataset[dataset['tenure'].isna()]
test_y = dataset[dataset['tenure'].isna()]
test_x=test_x.drop(columns='tenure')
test_x=test_x.drop(columns='TotalCharges')

# train dataset 만들기
train_x = dataset.dropna(how='any')
train_y=train_x[['tenure','TotalCharges']]
temp = train_x

train_x = train_x.drop(columns='tenure')
train_x = train_x.drop(columns='TotalCharges')

distortions = []

K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(train_x)
    kmeanModel.fit(train_x)
    distortions.append(sum(np.min(cdist(train_x, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / train_x.shape[0])
# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

# KMeans clustering
model = KMeans(n_clusters=3)
model.fit(train_x,train_y)
temp2 =  model.predict(train_x)
train_x=temp
train_x['KMeans'] =temp2
test_y['KMeans'] =  model.predict(test_x)

# KMeans 값의 평균값 넣어주기
test_y.loc[(test_y['KMeans']==0),'TotalCharges']=train_x.loc[(train_x['KMeans']==0),'TotalCharges'].mean()
test_y.loc[(test_y['KMeans']==1),'TotalCharges']=train_x.loc[(train_x['KMeans']==1),'TotalCharges'].mean()
test_y.loc[(test_y['KMeans']==2),'TotalCharges']=train_x.loc[(train_x['KMeans']==2),'TotalCharges'].mean()

test_y.loc[(test_y['KMeans']==0),'tenure']=train_x.loc[(train_x['KMeans']==0),'tenure'].mean()
test_y.loc[(test_y['KMeans']==1),'tenure']=train_x.loc[(train_x['KMeans']==1),'tenure'].mean()
test_y.loc[(test_y['KMeans']==2),'tenure']=train_x.loc[(train_x['KMeans']==2),'tenure'].mean()

dataset_KMeans = pd.concat([test_y, train_x])

print(dataset_KMeans)

parallel_coordinates(dataset_KMeans,'KMeans',color=('r','g','b'),alpha=0.5)
plt.show()

print("# 3. fill missing values with regression")
# test dataset 만들기
test_x = dataset[dataset['tenure'].isna()]
test_y = dataset[dataset['tenure'].isna()]
test_x = test_x.drop(columns='tenure')
test_x = test_x.drop(columns='TotalCharges')

# train dataset 만들기
train_x = dataset.dropna(how='any')
train_y = train_x['tenure']
temp = train_x

train_x = train_x.drop(columns='tenure')
train_x = train_x.drop(columns='TotalCharges')

# compute regression for tenure
tenure_reg = LinearRegression()
tenure_reg.fit(train_x, train_y)
predict_tenure = tenure_reg.predict(test_x)
print(predict_tenure)
print(tenure_reg.score(train_x, train_y))

train_y = temp['TotalCharges']
# compute regression for TotalCharges
charge_reg = LinearRegression()
charge_reg.fit(train_x,train_y)
predict_charge = charge_reg.predict(test_x)
print(predict_charge)

test_x['tenure'] = predict_tenure
test_x['TotalCharges']=predict_charge

train_x['TotalCharges'] = temp['TotalCharges']
train_x['tenure'] = temp['tenure']

dataset_linearRegression =  pd.concat([train_x, test_x])

print(dataset_linearRegression)


def telecom_churn_prediction(algorithm, training_x, testing_x,
                             training_y, testing_y, cols, cf, threshold_plot):
    # model
    algorithm.fit(training_x, training_y)
    predictions = algorithm.predict(testing_x)
    probabilities = algorithm.predict_proba(testing_x)
    # coeffs
    if cf == "coefficients":
        coefficients = pd.DataFrame(algorithm.coef_.ravel())
    elif cf == "features":
        coefficients = pd.DataFrame(algorithm.feature_importances_)

    column_df = pd.DataFrame(cols)
    coef_sumry = (pd.merge(coefficients, column_df, left_index=True,
                           right_index=True, how="left"))
    coef_sumry.columns = ["coefficients", "features"]
    coef_sumry = coef_sumry.sort_values(by="coefficients", ascending=False)

    print(algorithm)
    print("\n Classification report : \n", classification_report(testing_y, predictions))
    print("Accuracy   Score : ", accuracy_score(testing_y, predictions))
    # confusion matrix
    conf_matrix = confusion_matrix(testing_y, predictions)
    # roc_auc_score
    model_roc_auc = roc_auc_score(testing_y, predictions)
    print("Area under curve : ", model_roc_auc, "\n")
    fpr, tpr, thresholds = roc_curve(testing_y, probabilities[:, 1])

    # plot confusion matrix
    trace1 = go.Heatmap(z=conf_matrix,
                        x=["Not churn", "Churn"],
                        y=["Not churn", "Churn"],
                        showscale=False, colorscale="Picnic",
                        name="matrix")

    # plot roc curve
    trace2 = go.Scatter(x=fpr, y=tpr,
                        name="Roc : " + str(model_roc_auc),
                        line=dict(color=('rgb(22, 96, 167)'), width=2))
    trace3 = go.Scatter(x=[0, 1], y=[0, 1],
                        line=dict(color=('rgb(205, 12, 24)'), width=2,
                                  dash='dot'))

    # plot coeffs
    trace4 = go.Bar(x=coef_sumry["features"], y=coef_sumry["coefficients"],
                    name="coefficients",
                    marker=dict(color=coef_sumry["coefficients"],
                                colorscale="Picnic",
                                line=dict(width=.6, color="black")))

    # subplots
    fig = tls.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],
                            subplot_titles=('Confusion Matrix',
                                            'Receiver operating characteristic',
                                            'Feature Importances'))

    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 2)
    fig.append_trace(trace3, 1, 2)
    fig.append_trace(trace4, 2, 1)

    fig['layout'].update(showlegend=False, title="Model performance",
                         autosize=False, height=900, width=800,
                         plot_bgcolor='rgba(240,240,240, 0.95)',
                         paper_bgcolor='rgba(240,240,240, 0.95)',
                         margin=dict(b=195))
    fig["layout"]["xaxis2"].update(dict(title="false positive rate"))
    fig["layout"]["yaxis2"].update(dict(title="true positive rate"))
    fig["layout"]["xaxis3"].update(dict(showgrid=True, tickfont=dict(size=10),
                                        tickangle=90))
    py.iplot(fig)

    if threshold_plot == True:
        visualizer = DiscriminationThreshold(algorithm)
        visualizer.fit(training_x, training_y)
        visualizer.poof()

print(" Predict dataset")
# split dataset to train, test with drop
Y_drop = dataset_drop['Churn']
X_drop = dataset_drop.drop(columns='Churn')

X_drop_train, X_drop_test, Y_drop_train, Y_drop_test = train_test_split(X_drop, Y_drop, test_size=0.1, shuffle=True, stratify=Y_drop, random_state=34)

# split dataset to train, test with k-means
Y_KMeans = dataset_KMeans['Churn']
X_KMeans = dataset_KMeans.drop(columns='Churn')

X_KMeans_train, X_KMeans_test, Y_KMeans_train, Y_KMeans_test = train_test_split(X_KMeans, Y_KMeans, test_size=0.1, shuffle=True, stratify=Y_KMeans, random_state=34)

# split dataset to train, test with regression
Y_regression = dataset_linearRegression['Churn']
X_regression = dataset_linearRegression.drop(columns='Churn')

X_reg_train, X_reg_test, Y_reg_train, Y_reg_test = train_test_split(X_regression, Y_regression, test_size=0.1, shuffle=True, stratify=Y_regression, random_state=34)

print(" 1. Random forest")
# 소라가 코딩 - dataset_drop이랑 dataset_clustering 으로 두번 진행
print(" 2. SVM - support vector machine")
# 수경이가 코딩 - dataset_drop이랑 dataset_clustering, dataset_regression으로 3번 진행
print(" 2-1. use dataset that missing value is dropped")
svc_lin  = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
               decision_function_shape='ovr', degree=3, gamma=1.0, kernel='linear',
               max_iter=-1, probability=True, random_state=None, shrinking=True,
               tol=0.001, verbose=False)

lab_enc = LabelEncoder()
Y_drop_train_encod = lab_enc.fit_transform(Y_drop_train)
Y_drop_test_encod = lab_enc.fit_transform(Y_drop_test)

cols = [i for i in dataset.columns if i not in target_col]
telecom_churn_prediction(svc_lin,X_drop_train,X_drop_test,Y_drop_train_encod,Y_drop_test_encod,
                         cols,"coefficients",threshold_plot = False)
print(" 2-2. use dataset that missing value is filled with k-means")
svc_lin  = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
               decision_function_shape='ovr', degree=3, gamma=1.0, kernel='linear',
               max_iter=-1, probability=True, random_state=None, shrinking=True,
               tol=0.001, verbose=False)

lab_enc = LabelEncoder()
Y_KMeans_train_encod = lab_enc.fit_transform(Y_KMeans_train)
Y_KMeans_test_encod = lab_enc.fit_transform(Y_KMeans_test)

cols = [i for i in dataset.columns if i not in target_col]
telecom_churn_prediction(svc_lin,X_KMeans_train,X_KMeans_test,Y_KMeans_train_encod,Y_KMeans_test_encod,
                         cols,"coefficients",threshold_plot = False)

print(" 2-3. use dataset that missing value is filled with regression")
svc_lin  = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
               decision_function_shape='ovr', degree=3, gamma=1.0, kernel='linear',
               max_iter=-1, probability=True, random_state=None, shrinking=True,
               tol=0.001, verbose=False)

lab_enc = LabelEncoder()
Y_reg_train_encod = lab_enc.fit_transform(Y_reg_train)
Y_reg_test_encod = lab_enc.fit_transform(Y_reg_test)

cols = [i for i in dataset.columns if i not in target_col]
telecom_churn_prediction(svc_lin,X_reg_train,X_reg_test,Y_reg_train_encod,Y_reg_test_encod,
                         cols,"coefficients",threshold_plot = False)