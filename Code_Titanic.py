import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

file_path = "D:\\Books\\project\\0003-Titanic\\train.csv"

Dataset = pd.read_csv(file_path)


#_______________________[Deta Analysis]_______________________

Dataset['Title'] = Dataset['Name'].apply(lambda x : x.split(',')[1].split('.')[0].strip())
title_sex_mean = Dataset.groupby('Title')['Age'].mean()  # ✅ این خط را اضافه کنید
Dataset['Age'] = Dataset.apply(
    lambda row: title_sex_mean[row['Title']] if pd.isna(row['Age']) else row['Age'],
    axis=1
)

Dataset['Family_number']=Dataset['SibSp'] + Dataset['Parch']

Dataset = Dataset.drop(['PassengerId','Name','Cabin','Ticket','SibSp','Parch','Title'],axis=1)

Dataset['Sex'] = Dataset['Sex'].replace({'female':0,'male':1})

Dataset['Embarked']=Dataset['Embarked'].fillna(Dataset['Embarked'].mode()[0])
Dataset['Embarked'] = Dataset['Embarked'].replace({'S':1,'C':2,'Q':3})

Dataset = Dataset.astype(float)

#_______________________[Train and Create model]_______________________

x = Dataset.drop('Survived',axis=1)
y = Dataset['Survived']

x_train , x_test , y_train , y_test = train_test_split(x, y, random_state=42, test_size=0.2)

scaler = StandardScaler()  
x_train = scaler.fit_transform(x_train)  
x_test = scaler.transform(x_test) 


model =SVC(random_state=42,C=1,gamma='scale',kernel='rbf').fit(x_train,y_train)

print(model.score(x_test,y_test))

# _______________________[Test]_______________________


file_path = "D:\\Books\\project\\0003-Titanic\\test.csv"

Dataset_test = pd.read_csv(file_path)

passenger_ids = Dataset_test['PassengerId'].copy()

Dataset_test['Title'] = Dataset_test['Name'].apply(lambda x : x.split(',')[1].split('.')[0].strip())

Dataset_test['Age'] = Dataset_test.apply(
    lambda row: title_sex_mean[row['Title']] if pd.isna(row['Age']) else row['Age'],
    axis=1
)

Dataset_test['Family_number']=Dataset_test['SibSp'] + Dataset_test['Parch']

Dataset_test = Dataset_test.drop(['PassengerId','Name','Cabin','Ticket','SibSp','Parch','Title'],axis=1)

Dataset_test['Sex'] = Dataset_test['Sex'].replace({'female':0,'male':1})

Dataset_test['Fare']=Dataset_test['Fare'].fillna(Dataset_test['Fare'].mode()[0])

Dataset_test['Embarked']=Dataset_test['Embarked'].fillna(Dataset_test['Embarked'].mode()[0])
Dataset_test['Embarked'] = Dataset_test['Embarked'].replace({'S':1,'C':2,'Q':3})

Dataset_test = Dataset_test.astype(float)

Dataset_test_scaled = scaler.transform(Dataset_test)

predictions = model.predict(Dataset_test_scaled)


output = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': predictions
})

output_path = "D:\\Books\\project\\0003-Titanic\\FFF.csv"
output.to_csv(output_path, index=False)
