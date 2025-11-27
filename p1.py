import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X=pd.read_csv("linearX.csv", header=None)
y=pd.read_csv("linearY.csv", header=None)

train_ratio=1
train_size=int(train_ratio*len(X))
X_train=X[:train_size]
y_train=y[:train_size]
X_test=X[train_size:]
y_test=y[train_size:]

new_col=np.ones(X_train.shape[0])
X_train=np.insert(X_train,0,new_col, axis=1)

def calc_J_theta(theta,X_train,y_train):
  sample_size=len(X_train)
  J_theta=np.dot(X_train,theta)
  J_theta=J_theta-y_train
  J_theta=np.sum(J_theta**2)
  J_theta=J_theta/(2*sample_size)
  return J_theta

def calc_gradient_J_theta(theta,X_train,y_train):
  sample_size=len(X_train)
  gradient_J_theta=np.matmul(X_train.T,np.matmul(X_train,theta)-y_train)
  gradient_J_theta=gradient_J_theta/sample_size
  return gradient_J_theta


def gradient_descent(learning_rate, X_train, y_train):
  sample_size=len(X_train)
  y_train=y_train.reshape(-1,1)
  stopping_criteria=0.01
  theta=np.zeros(X_train.shape[1]);
  theta=theta.reshape(-1,1)
  iters=100000
  while(iters>0):
    last_theta=theta
    gradient_J_theta=calc_gradient_J_theta(theta,X_train,y_train)
    change_J_theta=gradient_J_theta*learning_rate
    theta=theta-change_J_theta
    if(np.abs(calc_J_theta(last_theta,X_train,y_train)-calc_J_theta(theta,X_train,y_train))<stopping_criteria):
      break
    iters-=1
  return theta


# let's do the descent
theta=np.array([0,0])
theta=gradient_descent(0.1, X_train, y_train)
print(theta)

y_pred=np.dot(X_test,theta)
print(y_pred)

#let's calculate the error

error=np.sum((y_pred-y_test)**2)
print(error)

#let's do some plotting now

plt.scatter(X_train[:,1],y_train)
plt.show()


#let's see how our gradient descended on the test data
plt.scatter(X_test[:,1],y_test)
plt.scatter(X_test[:,1],y_pred)
plt.show()

# the slope is slightly off but overall really good

# now let's draw a 3d mesh and visualise how does the j(theta look like)

theta_1=np.linspace(-14,26,200)
theta_2=np.linspace(-8,48,200)
theta_1,theta_2=np.meshgrid(theta_1,theta_2)

#we are flattening the grid points and stacking them in a matrix and calculating the errors and then reshaping the error matrix
theta_grid=np.stack([theta_1.ravel(),theta_2.ravel()],axis=0)
J_theta=np.matmul(X_train,theta_grid)
J_theta=J_theta-y_train
J_theta=np.sum(J_theta**2,axis=0)
J_theta=J_theta/(2*len(X_train))
J_theta=J_theta.reshape(theta_1.shape)
print(J_theta.shape)


#the graph looks cool now, yeah let's plot it a bit bigger
fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(111,projection='3d')
ax.plot_surface(theta_1,theta_2,J_theta, cmap= 'magma')
plt.show()


#okay let's start with now showing the values of theta as we descend,
#so let's just store the path as a list while we do a gradient descent

def gradient_descent_store(learning_rate, X_train, y_train):
  sample_size=len(X_train)
  y_train=y_train.reshape(-1,1)
  stopping_criteria=0.01
  theta=np.zeros(X_train.shape[1]);
  theta=theta.reshape(-1,1)
  iters=100
  path=[]
  while(iters>0):
    last_theta=theta
    gradient_J_theta=calc_gradient_J_theta(theta,X_train,y_train)
    change_J_theta=gradient_J_theta*learning_rate
    theta=theta-change_J_theta
    path.append(theta)
    if(np.abs(calc_J_theta(last_theta,X_train,y_train)-calc_J_theta(theta,X_train,y_train))<stopping_criteria):
      break
    iters-=1
  return path

path=gradient_descent_store(0.1,X_train,y_train)
path=np.array(path)
print(path.shape)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

# draw the surface and calculte the paths
ax.plot_surface(theta_1, theta_2, J_theta, cmap="viridis", alpha=0.6)
path_theta_1 = [theta[0] for theta in path]
path_theta_2 = [theta[1] for theta in path]
path_J = [calc_J_theta(theta, X_train, y_train) for theta in path]


point, = ax.plot([path_theta_1[0]], [path_theta_2[0]], [path_J[0]], 'ro', markersize=8)


for x, y, z in zip(path_theta_1, path_theta_2, path_J):
    x = float(np.squeeze(x))
    y = float(np.squeeze(y))
    z = float(np.squeeze(z))

    point.set_data([x], [y])         
    point.set_3d_properties([z])     
    plt.draw()
    plt.pause(0.2)

plt.show()

#let's plot the contours now
# 3D surface
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)


# contours projected on the bottom (XY plane at offset)
contours=ax.contour(theta_1, theta_2, J_theta, zdir='z', offset=J_theta.min()-1,levels=30, cmap='magma')
ax.clabel(contours, inline=True, fontsize=8)
# labels
ax.set_xlabel("theta_1")
ax.set_ylabel("theta_2")
plt.show()


# 2D contour animation
fig, ax = plt.subplots(figsize=(10, 10))

# drawing the contour map of cost function
contours = ax.contour(theta_1, theta_2, J_theta, levels=30, cmap='magma')

ax.clabel(contours, inline=True, fontsize=8)

ax.set_xlabel("theta_1")
ax.set_ylabel("theta_2")


# Initialize the point
point, = ax.plot([path_theta_1[0]], [path_theta_2[0]], 'ro', markersize=8)

# animate path
for x, y in zip(path_theta_1, path_theta_2):
    x = float(np.squeeze(x))
    y = float(np.squeeze(y))

    point.set_data([x], [y])   
    plt.draw()
    plt.pause(0.2)

plt.show()
