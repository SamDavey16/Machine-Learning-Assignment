import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import math
from sklearn.model_selection import train_test_split

data_train = pd.read_csv('Task1 - dataset - pol_regression.csv')
data_test = pd.read_csv('Task1 - dataset - pol_regression.csv')

dt = data_train.sort_values(by='x', ascending=False)

dt2 = data_test.sort_values(by='x', ascending=False)

x_train = dt['x']
y_train = dt['y']

x_test = dt2['x']
y_test = dt2['y']

plt.clf()
plt.plot(x_train,y_train, 'bo')
plt.plot(x_test,y_test, 'g')
plt.legend(('training points', 'ground truth'))
#plt.hold(True)
plt.savefig('trainingdata.png')
plt.show()

X = np.column_stack((np.ones(x_train.shape), x_train))

A = np.array([[1, 0.5], [0.5, 1]])
a = np.array([[1], [0]])

# specify data points for x0 and x1 (from - 5 to 5, using 51 uniformly distributed points)
x0Array = np.linspace(-5, 5, 51)
x1Array = np.linspace(-5, 5, 51)

Earray = np.zeros((51,51))
for i in range(0,50):
    for j in range(0,50):
        
        x = np.array([[x0Array[i]], [x1Array[j]]])
        tmp = a - 5 * x
        
        Earray[i,j] = tmp.transpose().dot(A).dot(tmp)

fig = plt.figure()
ax = fig.gca(projection='3d')

x0Grid, x1Grid = np.meshgrid(x0Array, x1Array)

# Plot the surface.
surf = ax.plot_surface(x0Grid, x1Grid, Earray, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.xlabel('beta0')
plt.ylabel('beta1')
plt.savefig('errorfunction.png')
plt.show()

w_old = [np.random.randn(),np.random.randn()]
w_new = [np.random.randn(),np.random.randn()]
epsilon = 0.00001
alpha = 0.01
def grad(w):
    #temp = -2*(y_train-X.dot(w))
    #return temp.transpose().dot(X)
    #(𝐴𝐵)𝑇=𝐵𝑇𝐴𝑇
    return -2*X.transpose().dot(y_train-X.dot(w))

states = []
while abs(np.linalg.norm(np.array(w_new)-np.array(w_old)))>epsilon:
    w_old = w_new
    w_new = w_old - alpha*grad(w_old)
    states.append(w_new)
#    print('w_new={}'.format(w_new))
#print("Local minimum occurs at {}".format(w_new))

error = y_train -X.dot(w_new)
SSE = error.dot(error)
SSE

XX = X.transpose().dot(X)

#w = np.linalg.solve(XX, X.transpose().dot(y_train))
w = np.linalg.inv(XX).dot(X.transpose().dot(y_train))

w

Xtest = np.column_stack((np.ones(x_test.shape), x_test))
ytest_predicted = Xtest.dot(w)

plt.figure()
plt.plot(x_test,y_test, 'g')
plt.plot(x_test, ytest_predicted, 'r')
plt.plot(x_train,y_train, 'bo')
plt.legend(('training points', 'ground truth', 'prediction'), loc = 'lower right')
#plt.hold(True)
plt.savefig('regression_LSS.png')
plt.show()

def getPolynomialDataMatrix(x, degree):
    X = np.ones(x.shape)
    for i in range(1,degree + 1):
        X = np.column_stack((X, x ** i))
    return X

def pol_regression(x,y,degree):
    X = getPolynomialDataMatrix(x, degree)

    XX = X.transpose().dot(X)
    w = np.linalg.solve(XX, X.transpose().dot(y))
    w = np.linalg.inv(XX).dot(X.transpose().dot(y))

    return w

plt.figure()
plt.plot(x_train,y_train, 'g')
plt.plot(x_train,y_train, 'bo')
    
w1 = pol_regression(x_train,y_train,1)
Xtest1 = getPolynomialDataMatrix(x_test, 1)
ytest1 = Xtest1.dot(w1)
plt.plot(x_test, ytest1, 'r')

w2 = pol_regression(x_train,y_train,2)
Xtest2 = getPolynomialDataMatrix(x_test, 2)
ytest2 = Xtest2.dot(w2)
plt.plot(x_test, ytest2, 'g')

w3 = pol_regression(x_train,y_train,3)
Xtest3 = getPolynomialDataMatrix(x_test, 3)
ytest3 = Xtest3.dot(w3)
plt.plot(x_test, ytest3, 'm')

w4 = pol_regression(x_train,y_train,6)
Xtest4 = getPolynomialDataMatrix(x_test, 6)
ytest4 = Xtest4.dot(w4)
plt.plot(x_test, ytest4, 'c')

w5 = pol_regression(x_train,y_train,10)
Xtest5 = getPolynomialDataMatrix(x_test, 10)
ytest5 = Xtest5.dot(w5)
plt.plot(x_test, ytest5, 'c')

Xtest6 = getPolynomialDataMatrix(x_test,0)
ytest6 = Xtest6 + Xtest6 / Xtest6
plt.plot(x_test, ytest6, 'c')

plt.legend(('training points', 'ground truth', '$x$', '$x^2$', '$x^3$', '$x^4$', '$x^5$', '$x^6$'), loc = 'lower right')

plt.savefig('polynomial.png')

## errors on test dataset Bashir
error1 = y_test-ytest1
SSE1 = error1.dot(error1)

error2 = y_test-ytest2
SSE2 = error2.dot(error2)

error3 = y_test-ytest3
SSE3 = error3.dot(error3)

error4 = y_test-ytest4
SSE4 = error4.dot(error4)

SSE1, SSE2, SSE3, SSE4

SSEtrain = np.zeros((11,1))
SSEtest = np.zeros((11,1))
MSSEtrain = np.zeros((11,1))
MSSEtest = np.zeros((11,1))

for i in range(1,12):
    
    Xtrain = getPolynomialDataMatrix(x_train, i) 
    Xtest = getPolynomialDataMatrix(x_test, i)
    
    w = pol_regression(x_train, y_train, i)  
    
    MSSEtrain[i - 1] = np.mean((Xtrain.dot(w) - y_train)**2)
    MSSEtest[i - 1] = np.mean((Xtest.dot(w) - y_test)**2)
    
    errortrain = y_train - Xtrain.dot(w) 
    errortest = y_test - Xtest.dot(w)
    SSEtrain[i-1] = errortrain.dot(errortrain)
    SSEtest[i-1] = errortest.dot(errortest)
    
plt.figure();
plt.semilogy(range(1,12), SSEtrain)
plt.semilogy(range(1,12), SSEtest)
plt.legend(('SSE on training set', 'SSE on test set'))
plt.savefig('polynomial_evaluation.png')
plt.show()

data = pd.read_csv('Task1 - dataset - pol_regression.csv')
dt = data_train.sort_values(by='x', ascending=False)
x = dt['x']
y = dt['y']
def eval_pol_regression(parameter, x, y, degree):
    RMSE_train = []
    RMSE_test = []
    degrees = [1, 2, 3, 6, 10]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
    w1 = pol_regression(x_train,y_train,1)
    Xtest1 = getPolynomialDataMatrix(x_test, 1)
    xtrain1 = getPolynomialDataMatrix(x_train, 1)
    ytest1 = Xtest1.dot(w1)
    error = y_test - Xtest1.dot(w1)
    squared_array = error * error
    mse_test = squared_array.mean()
    rmse = math.sqrt(mse_test)
    RMSE_test.append(rmse)
    print(rmse)
    error = y_train - xtrain1.dot(w1)
    squared_array = error * error
    mse_train = squared_array.mean()
    rmse_t = math.sqrt(mse_train)
    RMSE_train.append(rmse_t)
    print(rmse_t)

    w2 = pol_regression(x_train,y_train,2)
    Xtest2 = getPolynomialDataMatrix(x_test, 2)
    xtrain2 = getPolynomialDataMatrix(x_train, 2)
    ytest2 = Xtest2.dot(w2)
    error = y_test - Xtest2.dot(w2)
    squared_array = error * error
    mse_test = squared_array.mean()
    rmse2 = math.sqrt(mse_test)
    RMSE_test.append(rmse2)
    print(rmse2)
    error = y_train - xtrain2.dot(w2)
    squared_array = error * error
    mse_train = squared_array.mean()
    rmse_t2 = math.sqrt(mse_train)
    RMSE_train.append(rmse_t2)
    print(rmse_t2)

    w3 = pol_regression(x_train,y_train,3)
    Xtest3 = getPolynomialDataMatrix(x_test, 3)
    xtrain3 = getPolynomialDataMatrix(x_train, 3)
    ytest3 = Xtest3.dot(w3)
    error = y_test - Xtest3.dot(w3)
    squared_array = error * error
    mse_test = squared_array.mean()
    rmse3 = math.sqrt(mse_test)
    RMSE_test.append(rmse3)
    print(rmse3)
    error = y_train - xtrain3.dot(w3)
    squared_array = error * error
    mse_train = squared_array.mean()
    rmse_t3 = math.sqrt(mse_train)
    RMSE_train.append(rmse_t3)
    print(rmse_t3)

    w4 = pol_regression(x_train,y_train,6)
    Xtest4 = getPolynomialDataMatrix(x_test, 6)
    xtrain4 = getPolynomialDataMatrix(x_train, 6)
    ytest4 = Xtest4.dot(w4)
    error = y_test - Xtest4.dot(w4)
    squared_array = error * error
    mse_test = squared_array.mean()
    rmse4 = math.sqrt(mse_test)
    RMSE_test.append(rmse4)
    print(rmse4)
    error = y_train - xtrain4.dot(w4)
    squared_array = error * error
    mse_train = squared_array.mean()
    rmse_t4 = math.sqrt(mse_train)
    RMSE_train.append(rmse_t4)
    print(rmse_t4)

    w5 = pol_regression(x_train,y_train,10)
    Xtest5 = getPolynomialDataMatrix(x_test, 10)
    xtrain5 = getPolynomialDataMatrix(x_train, 10)
    ytest5 = Xtest5.dot(w5)
    error = y_test - Xtest5.dot(w5)
    squared_array = error * error
    mse_test = squared_array.mean()
    rmse5 = math.sqrt(mse_test)
    RMSE_test.append(rmse4)
    print(rmse5)
    error = y_train - xtrain5.dot(w5)
    squared_array = error * error
    mse_train = squared_array.mean()
    rmse_t5 = math.sqrt(mse_train)
    RMSE_train.append(rmse_t4)
    print(rmse_t5)

    print(RMSE_train)
    print(RMSE_test)
    plt.plot(RMSE_train, degrees, label = "train")
    plt.plot(RMSE_test, degrees, label = "test")
    plt.show()

p=1
eval_pol_regression(p, x, y, 1)