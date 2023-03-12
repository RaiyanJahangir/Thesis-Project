import joblib
model = joblib.load('model.pkl')
prediction = model.predict(X_test)
acccuracy = model.score(X_test, y_test)
print("prediction :",prediction)
print("Accuracy :",acccuracy)
    