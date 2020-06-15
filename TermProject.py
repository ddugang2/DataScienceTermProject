import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

pd.set_option("display.max_rows", 18, "display.max_columns", 21)
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# Read Excel file
# 상운 dataset = pd.read_csv(r'C:\Users\dltkd\OneDrive\바탕 화면\3학년 1학기\데이터 과학\termproject\WA_Fn-UseC_-Telco-Customer-Churn.csv')
# 수경 dataset = pd.read_csv('C:/WA_Fn-UseC_-Telco-Customer-Churn.csv')
# 소라 dataset = pd.read_csv('C:\Users\ehrqo\Desktop\WA_Fn-UseC_-Telco-Customer-Churn.csv')
dataset = pd.read_csv(r'C:\Users\dltkd\OneDrive\바탕 화면\3학년 1학기\데이터 과학\termproject\WA_Fn-UseC_-Telco-Customer-Churn.csv')

# statistical summary of dataset
print("# dataset statistical")
print(dataset.describe(),"\n")
print("# dataset head")
print(dataset.head(5), "\n")
print("# dataset shape")
print(dataset.shape, "\n")
print("# dataset index")
print(dataset.index, "\n")
print("# dataset features")
print(dataset.columns, "\n")

dataset = dataset.drop(columns='customerID')
dataset = dataset.drop(columns='PaperlessBilling')
dataset = dataset.drop(columns='PaymentMethod')

# tenure == 0 is wrong data
dataset['tenure'].replace(0, np.NAN, inplace=True)

# print is nan
print("# the number of null values")
print(dataset.isna().sum(), "\n")

# replace missing & wrong data with mean
dataset['tenure'].replace(np.NAN, dataset['tenure'].mean(), inplace=True)
dataset['TotalCharges'].replace(np.NAN, dataset['TotalCharges'].mean(), inplace=True)

print("# the number of null values after fill missing and wrong values ")
print(dataset.isnull().sum(), "\n")

print("# histograms")
plt.subplot(121)
plt.title("gender")
plt.hist(dataset['gender'], bins=3)
plt.subplot(122)
plt.title("SeniorCitizen")
plt.hist(dataset['SeniorCitizen'], bins=3)
plt.xlabel("0 = junior,1 = senior")
plt.xticks([0,1])
plt.show()
plt.subplot(121)
plt.title("Partner")
plt.hist(dataset['Partner'], bins=3)
plt.subplot(122)
plt.title("Dependents")
plt.hist(dataset['Dependents'], bins=3)
plt.show()
plt.subplot(121)
plt.title("tenure")
plt.hist(dataset['tenure'], bins=10)
plt.xticks([0,10,20,30,40,50,60,dataset['tenure'].max()])
plt.subplot(122)
plt.title("PhoneService")
plt.hist(dataset['PhoneService'], bins=3)
plt.show()
plt.subplot(121)
plt.title("MultipleLines")
plt.hist(dataset['MultipleLines'], bins=5)
plt.subplot(122)
plt.title("InternetService")
plt.hist(dataset['InternetService'], bins=5)
plt.show()
plt.subplot(121)
plt.title("OnlineSecurity")
plt.hist(dataset['OnlineSecurity'], bins=5)
plt.subplot(122)
plt.title("OnlineBackup")
plt.hist(dataset['OnlineBackup'], bins=5)
plt.show()
plt.subplot(121)
plt.title("DeviceProtection")
plt.hist(dataset['DeviceProtection'], bins=5)
plt.subplot(122)
plt.title("TechSupport")
plt.hist(dataset['TechSupport'], bins=5)
plt.show()
plt.subplot(121)
plt.title("StreamingTV")
plt.hist(dataset['StreamingTV'], bins=5)
plt.subplot(122)
plt.title("StreamingMovies")
plt.hist(dataset['StreamingMovies'], bins=5)
plt.show()
plt.subplot(121)
plt.title("Contract")
plt.hist(dataset['Contract'], bins=5)
plt.subplot(122)
plt.title("MonthlyCharges")
plt.hist(dataset['PhoneService'], bins=3)
plt.show()
plt.title("TotalCharges")
plt.hist(dataset['TotalCharges'], bins=10)
plt.xticks([0,1000,2000,3000,4000,5000,6000,7000,8000,dataset['TotalCharges'].max()])
plt.show()
plt.title("Churn")
plt.hist(dataset['Churn'], bins=3)
plt.show()

categoricalValue = np.array(dataset)

label_encoder = LabelEncoder()

for i in range(0, 18):
    categoricalValue[:, i] = label_encoder.fit_transform(categoricalValue[:, i])

col = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'MonthlyCharges', 'TotalCharges', 'Churn']
dataset = pd.DataFrame(categoricalValue, columns=col)

Y = dataset['Churn']
X = dataset.drop(columns='Churn')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

standard_X = StandardScaler()

X_train = standard_X.fit_transform(X_train)
X_test = standard_X.fit_transform(X_test)

print(X_train)
