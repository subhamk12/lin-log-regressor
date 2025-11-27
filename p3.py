import numpy as np
import pandas as pd

# we will implement newton's method to find the minima of
# L(θ) = {sigma i=1 to m} (y(i) log hθ(x(i)) + (1 − y(i))log(1 − hθ(x(i))))
# we will need to find out the gradient of l(theta) and hessian

#newton's update is x = x -H(x0)^(-1) . grad(f(x0))
X=pd.read_excel("logisticX.xlsx", header=None)
y=pd.read_excel("logisticY.xlsx", header=None)
X=np.array(X)
y=np.array(y)

print(X.shape)
print(y.shape)

#let's add another column of 1's to X so that we can
#write matrix formulas simply without worrying about the intercept

new_col=np.ones((X.shape[0],1))
X=np.hstack((new_col,X))
print(X.shape)

#let's determine the gradient of L(θ)
#it will be a column vector

#∇w​L(w)=​i=1∑n​(y^​i​−yi​)xi​
#we will calculate and store h(theta)x
theta=np.zeros(X[0].shape)

h_theta_x=np.zeros(theta.shape)
theta=theta.reshape(-1,1)
h_theta_x=h_theta_x.reshape(-1,1)

print(theta.shape)
print(h_theta_x.shape)

h_theta_x=np.matmul(X,theta)
print(h_theta_x.shape)

gradient_loss=np.matmul(X.T,(h_theta_x-y))
print(gradient_loss.shape)

#now we calculate the hessian which can be written as Xt.S.X
#where S=Diag(h_theta_x.(1-h_theta_x))
temp=np.zeros(h_theta_x.shape)
temp=1-h_theta_x
print(temp.shape)