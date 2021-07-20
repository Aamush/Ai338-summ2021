import pandas as pd
from sklearn import linear_model

#Read CSV Files
trainData = pd.read_csv('train.csv')
testData = pd.read_csv('test.csv')

#Get Labels and remove them from Train data
YTrain = trainData.Survived;
trainData.drop('Survived',inplace=True,axis=1)

#Convert Data to Features
trainData.drop('Name',inplace=True,axis=1)
trainData.drop('Cabin',inplace=True,axis=1)
trainData.drop('Ticket',inplace=True,axis=1)
testData.drop('Name',inplace=True,axis=1)
testData.drop('Cabin',inplace=True,axis=1)
testData.drop('Ticket',inplace=True,axis=1)
#Convert String based columns to integer classes
trainData["Sex"] = trainData["Sex"].replace(['female','male'],[0,1])
testData["Sex"] = testData["Sex"].replace(['female','male'],[0,1])
trainData["Embarked"] = trainData["Embarked"].replace(['S','Q','C'],[0,1,2])
testData["Embarked"] = testData["Embarked"].replace(['S','Q','C'],[0,1,2])

#Remove Nan Values from Train
trainData.fillna(value=0,inplace=True)
#Dummy values in Test for all NaN
testData.fillna(value=trainData['Age'].mean(),inplace=True)
testData.fillna(value=trainData['Fare'].mean(),inplace=True)

#Test
print(YTrain.shape)
print(trainData)
print(testData.shape)

#Train
mnb = linear_model.Lasso(alpha=1)
mnb.fit(trainData,YTrain)
#Predictions
predictions = mnb.predict(testData)
print(predictions.shape)

# Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(X_train, Y_train)
Y_pred = mnb.predict(X_test)
acc_mnb = round(mnb.score(X_train, Y_train) * 100, 2)
print("Multinomial Naive Bayes accuracy =",round(acc_mnb,2,), "%")
print(Y_pred.shape)
print(Y_pred)

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submission.csv', index=False)

# K-Fold Cross Validation
Mul = MultinomialNB()
scores = cross_val_score(Mul, X_train, Y_train, cv=10, scoring = "accuracy")
print("Scores:\n", pd.Series(scores))
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print("Logistic Regression accuracy =",round(acc_log,2,), "%")
print(Y_pred.shape)
print(Y_pred)

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('LR_submission.csv', index=False)

logreg = LogisticRegression()
scores = cross_val_score(logreg, X_train, Y_train, cv=10, scoring = "accuracy")
print("Scores:\n", pd.Series(scores))
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())

# kNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print("kNN accuracy =",round(acc_knn,2,), "%")
print(Y_pred.shape)
print(Y_pred)

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('KNN_submission.csv', index=False)

knn = KNeighborsClassifier(n_neighbors = 3)
scores = cross_val_score(knn, X_train, Y_train, cv=10, scoring = "accuracy")
print("Scores:\n", pd.Series(scores))
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())

# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
print("Linear SVC accuracy =", round(acc_linear_svc,2,), "%")
print(Y_pred.shape)
print(Y_pred)

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('SVC_submission.csv', index=False)

linear_svc = LinearSVC()
scores = cross_val_score(linear_svc, X_train, Y_train, cv=10, scoring = "accuracy")
print("Scores:\n", pd.Series(scores))
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())


