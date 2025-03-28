#thomas method to solve tri-diagola system 

import numpy as np 
import time

A = np.array([[2,-1,0,0],
              [-1,2,-1,0],
              [0,-1,2,-1],
              [0,0,-1,2 ]],dtype= np.float64)

r = np.array([1,0,0,1],dtype= np.float64)
# x = np.linalg.solve(A,r)
# print(x)
# A = np.array([[ 1.4,-0.2,0 ],[-0.2,1.4,-0.2],[ 0,-0.2,1.4]],dtype= np.float64)

# r = np.array([0.7071,1,0.7071],dtype= np.float64)
# x = np.linalg.solve(A,r)
# print(x)
n = len(r)
# print(n)
b = np.zeros(n)
a = np.zeros(n)
c1 = np.zeros(n)
r1 = np.zeros(n)
c = np.zeros(n)
x = np.zeros(n)

for j in range(n):
    for i in range(n):
        if i == j:
            b[i] = A[i][j]
        elif i == j-1:
            c[i] = A[i][j]
        elif i == j+1:
            a[i] = A[i][j]
            
print("b = ",b)
print("a = ",a)
print("c = ",c)
ti = time.time()

c1[0] = c[0]/b[0]
r1[0] = r[0]/b[0]

for i in range(1,n):
    dnum = b[i] - a[i]*c1[i-1] 
    # a[i] = a[i]/(d[i]-b[i]*a[i-1])

    c1[i] = c[i]/dnum
    r1[i] = (r[i]-a[i]*r1[i-1])/dnum
# print(r)

x[n-1] = r1[n-1]
print(x)
for i in np.arange(n-2,-1,-1):
    x[i] = r1[i]-c1[i]*x[i+1]
    
print("solution space x= ", x)

tf = time.time()    
print(tf-ti)
# print(a)
# print(r)
# print("x",x)

 

