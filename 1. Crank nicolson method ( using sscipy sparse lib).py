from scipy.sparse import diags
import numpy as np
import math as mt
import time
import scipy
# # import matplotlib.pyplot as plt

# h = 0.1
# k = 0.005
# l = h**2/k 
# r = 2+l
# for z1 in range(1,8):
#     n = 2**z1*10
#     m = 2**z1*10    
#     p = np.zeros([n-1,n-1])
#     ai = np.zeros(n-1)
#     di = np.zeros(n-1)
#     ui = np.zeros(n-1)
#     vi = np.zeros(n-1)
#     v1= np.zeros(n-1)
#     c1 = np.zeros(n-1)
#     for i in range(0,n-1):
#         di[i] = r
#         ai[i] = -1
#         ui[i] = mt.sin(mt.pi*i*h)
#     # print(ui)
#     for z in range(m):
#         for i in range(n-1):
#             A = diags([ai,di,ai],[-1,0,1],shape = (n-1,n-1)).toarray()
#             ti = time.time()
#             vi = scipy.sparse.linalg.spsolve(A,ui) 
#             t = z*k
#             for i in range(1,n-1):
#                     ui[i] =abs( mt.e**((-(mt.pi**2)*t))*(mt.sin(mt.pi*i*h))-vi[i])
#                     # ui = np.round(ui,4)
#     print(f"\n n = m = {n}")
#     ui = max(ui)
#     print(f"max error(e{z1}):",ui)
#     tf = time.time()
#     T1 = tf-ti
#     print(f"time cost (t{z1}):",T1)  




from scipy.sparse import diags
import numpy as np
import math as mt
import time
import scipy
import matplotlib.pyplot as plt

h = 0.1
k = 0.005
l = h**2/k 
r = 2+l

# Arrays to store max errors and times
max_errors = []
times = []

for z1 in range(1,6):
    n = 2**z1 * 10
    m = 2**z1 * 10

    di = np.zeros(n-1)
    ai = np.zeros(n-1)
    ui = np.zeros(n-1)
    vi = np.zeros(n-1)

    # Initial conditions
    for i in range(0, n-1):
        di[i] = r
        ai[i] = -1
        ui[i] = mt.sin(mt.pi * i * h)

    # Sparse matrix A setup
    A = diags([ai, di, ai], [-1, 0, 1], shape=(n-1, n-1)).toarray()

    # Time stepping
    for z in range(m):
        ti = time.time()
        vi = scipy.sparse.linalg.spsolve(A, ui)
        t = z * k
        for i in range(1, n-1):
            ui[i] = abs(mt.e**(-(mt.pi**2) * t) * (mt.sin(mt.pi * i * h)) - vi[i])

        tf = time.time()
        T1 = tf - ti

    max_error = max(ui)
    max_errors.append(max_error)
    times.append(T1)

    print(f"\n n = m = {n}")
    print(f"max error (e{z1}):", max_error)
    print(f"time cost (t{z1}):", T1)

# Plotting max errors over iterations
plt.figure(figsize=(10, 6))
plt.plot(range(1,6), max_errors, marker='o', linestyle='-', color='b')
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Max Error')
plt.title('Max Error Evolution over Iterations')
plt.grid(True)
plt.show()

    