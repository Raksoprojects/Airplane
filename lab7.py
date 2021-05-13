import numpy as np
from matplotlib import pyplot as plt


px = []
py = []
  

with open('lab_7_dane/measurements12.txt', 'r') as file:
    for line in file.readlines():
        x = line.split('\t')
        px.append(float(x[0]))
        py.append(float(x[1][0:-1]))

p = np.sqrt(np.square(px)+np.square(py))
t = np.arange(0, len(px), step=1)



# s[n+1] = F*s[n]+ G*q[n]
# z[k] = H*s[n] + w[n]
# s = [x,y,vx,vy].T
# q = [qx, qy].T -> Q
# w = [wx, wy].T ->: R
# z

# vx[n+1] = vx[n] + qx[n]
# vy[n+1] = vy[n] + qy[n]
# x[n+1] = x[n] + T*vx[n]
# y[n+1] = y[n] + T*vy[n]


T = 1
F = np.array([[1,0,T,0],
              [0,1,0,T],
              [0,0,1,0],
              [0,0,0,1]])

G = np.array([[0,0],
              [0,0],
              [1,0],
              [0,1]]) 

H = np.array([[1,0,0,0],
              [0,1,0,0]])


# samolot
p_x = px[0]
p_y = py[0]
vx = 0
vy = 0
s = np.array([p_x, p_y, 0, 0]).T
s_n = np.array([p_x, p_y, 0, 0]).T
P = np.identity(4, dtype=float) * 5

Q = np.array([[0.25,0], [0, 0.25]])
R = np.array([[2.0, 0], [0, 2.0]])
s_n_list = []
s_list = []
I = np.identity(4)

for i in t:
    
    znn = np.dot(H,s_n) + R
    s_n = np.reshape(np.dot(F, s_n), (4,-1)) + np.dot(G, Q)
    vx = vx +Q[0]
    vy = vy +Q[1]
    # x = x + T*vx
    # y = y + T*vy

    s = F*s
    P = np.dot(np.dot(F,P),F.T) + np.dot(np.dot(G,Q),G.T)
    z_n = np.dot(H,s)
    z = np.array([px[i],py[i]]).T
    e = np.reshape(z,(2,1)) - z_n
    S = np.dot(np.dot(H,P),H.T) + R
    K = np.dot(np.dot(P,H.T),np.linalg.inv(S))
    s = s +np.dot(K,e)
    P = (I - np.dot(K,H))

    s_n_list.append(np.mean(s_n))
    s_list.append(np.mean(s))

plt.plot(t,s_list)
# plt.plot(px,py,'x')
plt.plot(t,s_n_list)
plt.show()