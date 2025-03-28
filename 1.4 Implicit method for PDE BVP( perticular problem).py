# import numpy as np 
# import math as mt 
# import sympy as smp 
# from sympy import symbols, Eq, solve 
# import time 
# import matplotlib.pyplot as plt
# m1 = symbols('m1')
# m2 = symbols('m2')
# m3 = symbols('m3')
# n1 = symbols('n1')
# n2 = symbols('n2')
# n3 = symbols('n3')


# x0 = 0
# x1 = 2
# t0 = 0
# t1 = 0.1
# h = 0.5
# k = 0.05  
# c2 = 1
# l = (c2*k)/h**2 
# print("lambda = ",l)
# n = int((x1-x0)/h)
# print("n=",n)
# x = np.zeros(n+1)
# x[0]= x0

# ti= time.time()
# for i in range(n+1):
#     x[i] = x[i]+i*h
# print("x = ",x)

# nt = int((t1-t0)/k)
# print("nt=",nt)
# t = np.zeros(nt+1)
# t[0]= t0
# for i in range(nt+1):
#     t[i] = t[i]+i*k
# print("t = ",t)

# u = np.zeros([nt+1,n+1 ])

# u0t = 0
# u1t = 0
# u[0][0] = u0t
# u[1][0] = u0t
# u[2][0] = u0t
# u[0][n] = u1t
# u[1][n] = u1t
# u[2][n] = u1t
# for j in range(1,n):
#     u[0][j] = mt.sin(mt.pi/2*x[j])
#     u[0][j] = round(u[0][j],4)
#     # for i in range(1,nt+1):
#         # u[i][j-1] = (1+2*l)*u[i][j]-l*u[i+1][j]-l*u[i-1][j] 
    
# eq1 = Eq((1+2*l)*m1-l*m2-l*u[1][0],u[0][1])  
# eq2 = Eq((1+2*l)*m2-l*m3-l*u[1][1],u[0][2])
# eq3 = Eq((1+2*l)*m3-l*u[1][4]+l*u[1][2],u[0][3])
# m = solve((eq1, eq2, eq3 ), (m1,m2,m3))
# u[1][1] = m[m1]
# u[1][2] = m[m2]
# u[1][3] = m[m3]

# eq4 = Eq((1+2*l)*n1-l*n2-l*u[2][0],u[1][1])  
# eq5 = Eq((1+2*l)*n2-l*n3-l*u[2][1],u[1][2])
# eq6 = Eq((1+2*l)*n3-l*u[1][4]+l*u[2][2],u[1][3])
# v = solve((eq4, eq5, eq6 ), (n1,n2,n3))

# u[2][1] = v[n1]
# u[2][2] = v[n2]
# u[2][3] = v[n3]

# tf = time.time()
# print(u)
# print(tf-ti)
# # plt.scatter(x, u, label= "stars", color= "red", 
# #             marker= "o", s=50)
# plt.plot(x,u,color='g', linewidth =1)
# plt.title("Curve plotted using  the given points")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()









import numpy as np
import math as mt
import sympy as smp
from sympy import symbols, Eq, solve
import time
import matplotlib.pyplot as plt

m1, m2, m3 = symbols('m1 m2 m3')
n1, n2, n3 = symbols('n1 n2 n3')

x0 = 0
x1 = 2
t0 = 0
t1 = 0.1
h = 0.5
k = 0.05
c2 = 1
l = (c2 * k) / h**2
print("lambda =", l)

n = int((x1 - x0) / h)
print("n =", n)
x = np.linspace(x0, x1, n + 1)
print("x =", x)

nt = int((t1 - t0) / k)
print("nt =", nt)
t = np.linspace(t0, t1, nt + 1)
print("t =", t)

u = np.zeros((nt + 1, n + 1))

# Initial condition
for j in range(1, n):
    u[0, j] = round(mt.sin(mt.pi / 2 * x[j]), 4)

# Boundary conditions
u[:, 0] = 0  # u(0, t) = 0 for all t
u[:, -1] = 0  # u(2, t) = 0 for all t

# Time-stepping loop
for i in range(1, nt + 1):
    eq1 = Eq((1 + 2 * l) * m1 - l * m2 - l * u[i - 1, 0], u[i - 1, 1])
    eq2 = Eq((1 + 2 * l) * m2 - l * m3 - l * u[i - 1, 1], u[i - 1, 2])
    eq3 = Eq((1 + 2 * l) * m3 - l * u[i - 1, 4] - l * u[i - 1, 2], u[i - 1, 3])
    m = solve((eq1, eq2, eq3), (m1, m2, m3))
    u[i, 1] = m[m1]
    u[i, 2] = m[m2]
    u[i, 3] = m[m3]

    if i < nt:
        eq4 = Eq((1 + 2 * l) * n1 - l * n2 - l * u[i, 0], u[i, 1])
        eq5 = Eq((1 + 2 * l) * n2 - l * n3 - l * u[i, 1], u[i, 2])
        eq6 = Eq((1 + 2 * l) * n3 - l * u[i, 4] - l * u[i, 2], u[i, 3])
        v = solve((eq4, eq5, eq6), (n1, n2, n3))
        u[i + 1, 1] = v[n1]
        u[i + 1, 2] = v[n2]
        u[i + 1, 3] = v[n3]

# tf = time.time()
print(u)
# print("Time taken:", tf - ti)

# Plotting the results
plt.figure(figsize=(10, 6))
for i in range(0, nt + 1, max(1, nt // 5)):  # Plot every 1/5th of nt
    plt.plot(x, u[i, :], label=f't = {t[i]:.2f}')
plt.title("Heat Equation Solution")
plt.xlabel("x")
plt.ylabel("u(x, t)")
plt.legend()
plt.grid(True)
plt.show()
