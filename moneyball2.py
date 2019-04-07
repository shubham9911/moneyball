
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
Mb = pd.read_csv("baseball.csv")
Mb.head()

#RD=moneyball[3]-moneyball[4]



RS=Mb.iloc[:,3]
RA=Mb.iloc[:,4]

RD=np.array(RS.sub(RA))
RDf=pd.DataFrame(RD)
Mb=pd.concat([Mb,RDf],axis=1)


x1 = np.array(Mb.RS)
y = np.array(Mb.W)

# Deriving slope,intercept values

slope, intercept = np.polyfit(x1, y, 1)
abline_values = [slope * i + intercept for i in x1]

#Plotting the figure

plt.figure(figsize=(10,6))
plt.scatter(x1, y)
plt.plot(x1, abline_values, 'r')
plt.title("Slope = %s" % (slope))
plt.xlabel("Run Scored")
plt.ylabel("Wins")
plt.show()

x2 = np.array(Mb.RA)
y = np.array(Mb.W)

# Deriving slope,intercept values

slope, intercept = np.polyfit(x2, y, 1)
abline_values = [slope * i + intercept for i in x2]

#Plotting the figure

plt.figure(figsize=(10,6))
plt.scatter(x2, y)
plt.plot(x2, abline_values, 'r')
plt.title("Slope = %s" % (slope))
plt.xlabel("Run Allowed")
plt.ylabel("Wins")
plt.show()

x3 = np.array(Mb.SLG)
y = np.array(Mb.W)
# Deriving slope,intercept values

slope, intercept = np.polyfit(x3, y, 1)
abline_values = [slope * i + intercept for i in x3]
#Plotting the figure
plt.figure(figsize=(10,6))
plt.scatter(x3, y)
plt.plot(x3, abline_values, 'r')
plt.title("Slope = %s" % (slope))
plt.xlabel("OBP")
plt.ylabel("Wins")
plt.show()



x4 = np.array(Mb.OBP)
y = np.array(Mb.W)

# Deriving slope,intercept values

slope, intercept = np.polyfit(x4, y, 1)
abline_values = [slope * i + intercept for i in x4]

#Plotting the figure

plt.figure(figsize=(10,6))
plt.scatter(x4, y)
plt.plot(x4, abline_values, 'r')
plt.title("Slope = %s" % (slope))
plt.xlabel("Slag")
plt.ylabel("Wins")
plt.show()




import warnings
warnings.simplefilter('ignore',np.RankWarning)

x5 = np.array(Mb.OOBP)
y = np.array(Mb.W)

# Deriving slope,intercept values

slope, intercept = np.polyfit(x5, y, 1)
abline_values = [slope * i + intercept for i in x5]

#Plotting the figure
plt.figure(figsize=(10,6))
plt.scatter(x5, y)
plt.plot(x5, abline_values, 'r')
plt.title("Slope = %s" % (slope))
plt.xlabel("OOBP")
plt.ylabel("Wins")
plt.show()



x6 = np.array(Mb.OSLG)
y = np.array(Mb.W)

# Deriving slope,intercept values

slope, intercept = np.polyfit(x6, y, 1)
abline_values = [slope * i + intercept for i in x5]

#Plotting the figure

plt.figure(figsize=(10,6))
plt.scatter(x6, y)
plt.plot(x6, abline_values, 'r')
plt.title("Slope = %s" % (slope))
plt.xlabel("OSLG")
plt.ylabel("Wins")
plt.show()

plt.figure(figsize =(18,6))
plt.bar(Mb.Team , Mb.W , color ='blue')
plt.ylabel('Wins')
plt.xlabel('Teams')
plt.figure(figsize =(18,6))
plt.bar(Mb.Team , Mb.RS , color ='red')
plt.ylabel('Run scored')
plt.xlabel('Teams')
plt.figure(figsize =(18,6))
plt.bar(Mb.Team , Mb.RA , color ='green')
plt.ylabel('Run allowed')
plt.xlabel('Teams')


print(np.corrcoef(Mb.OBP,Mb.RS))


print(np.corrcoef(Mb.SLG,Mb.RS))


print(np.corrcoef(Mb.BA,Mb.RS))




from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

# Extracting our variables from the dataframe.

x = Mb[['OBP','SLG','BA']].values
y = Mb[['RS']].values

# Calling our model object.

RS_model = LinearRegression()

# Fitting the model.

RS_model.fit(x,y)

# Printing model intercept and coefficients.

print(RS_model.intercept_)


print(RS_model.coef_)

# Extracting our variables from the dataframe.

x = Mb[['OBP','SLG']].values
y = Mb[['RS']].values

# Calling our model object.

RS_model = LinearRegression()

# Fitting the model.

RS_model.fit(x,y)

# Printing model intercept and coefficients.

print(RS_model.intercept_)

print(RS_model.coef_)


Mbnew = Mb.dropna()

# Extracting our variables from the dataframe.

x = Mbnew[['OOBP','OSLG']].values
y = Mbnew[['RA']].values

# Calling our model object.

RA_model = LinearRegression()

# Fitting the model.

RA_model.fit(x,y)

# Printing model intercept and coefficients.

print(RA_model.intercept_)
print(RA_model.coef_)


# Extracting our variables from the dataframe.


x = Mb[[0]].values
y = Mb[['W']].values
# Calling our model object.


W_model = LinearRegression()


# Fitting the model.


W_model.fit(x,y)
# Printing model intercept and coefficients.


print(W_model.intercept_)
print(W_model.coef_)

#from sklearn.cross_validation import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

#prediction

RS_pred=RS_model.predict([[0.328,0.418]])


RA_pred=RA_model.predict([[0.317,0.415]])

# Prediction for wins.


W_pred=W_model.predict([[47]])







