import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassier

xBlue = np.array([0,2,1,4,7,3,6,3,5,7])
yBlue = np.array([1,2,5,4,6,7,3,2,6,2])

xRed = np.array([2,1,2,4,5,6,7,8,9,2])
yRed = np.array([1,2,4,2,2,6,7,3,4,5])

X = np.array([[0,1],[2,2],[1,5],[4,4],[7,6],[3,7],[6,3],[3,2],[5,6],[7,2],[2,1],[1,2],[2,4],[4,2],[5,2],[6,6],[7,7],[8,3],[9,4],[2,5]])
Y = np.array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]) #0: blue and 1: Red class

plt.plot(xBlue, yBlue, 'ro', color = 'blue')
plt.plot(xRed, yRed, 'ro', color = 'red')

plt.plot(2,5,'ro',color= 'green', markersize = 15)

plt.axis([-0.5,10,-0.5,10])

classifer = KNeighborsClassier(n_neighbors = 3)#this is the k value , k small is underfitting and k when large is overfitting
classifer.fit(X,Y)

pred = classifer.predict([2,5])
print(pred)

plt.show()
