# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn 


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Vijayashree B
RegisterNumber: 212223040238 
*/
```
      import pandas as pd
      data=pd.read_csv("/content/Employee.csv")
      data.head()
      data.info()
      
      from sklearn.preprocessing import LabelEncoder
      le=LabelEncoder()
      data["salary"]=le.fit_transform(data['salary'])
      data.head()
      
      x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
      y=data['left']
      x.head()
      y.head()
      
      from sklearn.model_selection import train_test_split
      x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
      from sklearn.tree import DecisionTreeClassifier
      dt=DecisionTreeClassifier(criterion='entropy')
      dt.fit(x_train,y_train)
      y_pred=dt.predict(x_test)
      from sklearn.metrics import accuracy_score
      acc=accuracy_score(y_test,y_pred)
      print(acc)
      dt.predict([[0.5,0.8,9,260,6,0,1,2]])


## Output:

![image](https://github.com/user-attachments/assets/b24a872c-ca39-4580-bb11-84eead82e4da)

![image](https://github.com/user-attachments/assets/aca38bad-dbb2-4e4b-bf46-785ddc2a17db)

![image](https://github.com/user-attachments/assets/8e20b589-95ab-4483-bc6a-2eb184dced94)




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
