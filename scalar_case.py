# here we calculate the posterior variance in a simple scalar case of VIVID using symbolic computation

from sympy import symbols,Matrix,lambdify

b, p, h,r = symbols('b,p,h,r')
M1 = Matrix(([b+p, b*h], [b*h, h**2*b+r]))
M2 = Matrix(([b, b*h]))
K = M2.T*M1.inv()
H_tilde = Matrix(([1, h]))
M3 = K*H_tilde
AA = (1-M3[0])*b 

QQ = r*b/(h*h*b+r)

f=lambdify((b,p,h,r), AA) 
Q=lambdify((b,h,r), QQ)

f_list = []
q_list = []
b_list = []

for i in range(20):
  b = i*0.1
  f_list.append(f(b,1,1,1))
  q_list.append(Q(b,1,1))
  b_list.append(b)


import matplotlib.pyplot as plt
plt.plot(b_list,f_list,'b',label='$Tr(\mathbf{A}^{{VIVID}}_t)$')
plt.plot(b_list,q_list,'r',label='$Tr(\mathbf{A}^{{DA}}_t)$')
plt.legend(fontsize = 16)
plt.xlabel('$\mathbf{B}_t$',fontsize = 16)
