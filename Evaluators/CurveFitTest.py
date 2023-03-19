import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

def model(x,u):
    """定义拟合的曲线
    :param x:输入值自变量
    :param u:输入值函数的参数
    :return:返回值因变量
    """
    return x[0] * (u ** 2 + x[1] * u) / (u ** 2 + x[2] * u + x[3])

def fun(x,u,y):
    return model(x,u) - y

def jac(x,u,y):
    J = np.empty((u.size,x.size))
    den = u ** 2 + x[2] * u + x[3]
    num = u ** 2 + x[1] * u
    J[:,0] = num / den
    J[:,1] = x[0] * u / den
    J[:,2] = -x[0] * num * u / den ** 2
    J[:,3] = -x[0] * num / den ** 2
    return J

#输入值自变量
u = np.array([4.0, 2.0, 1.0, 5.0e-1, 2.5e-1, 1.67e-1, 1.25e-1, 1.0e-1,
              8.33e-2, 7.14e-2, 6.25e-2])
#输入值因变量
y = np.array([1.957e-1, 1.947e-1, 1.735e-1, 1.6e-1, 8.44e-2, 6.27e-2,
              4.56e-2, 3.42e-2, 3.23e-2, 2.35e-2, 2.46e-2])
#函数的参数
x0 = np.array([2.5, 3.9, 4.15, 3.9])
#利用jac矩阵结合最小二乘法来计算曲线的参数,设置参数的取值在(0,100)之间
res = least_squares(fun, x0, jac=jac, bounds=(0, 100), args=(u, y), verbose=1)

#需要预测值得输入值
u_test = np.linspace(0, 5)
#利用计算的曲线参数来计算预测值
y_test = model(res.x, u_test)
plt.plot(u, y, 'o', markersize=4, label='data')
plt.plot(u_test, y_test, label='fitted model')
plt.xlabel("u")
plt.ylabel("y")
plt.legend(loc='lower right')
plt.show()
