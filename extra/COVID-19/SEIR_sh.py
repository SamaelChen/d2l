# %%
import plotly.graph_objects as go
import plotly as ply
import scipy.integrate as spi
import numpy as np
import matplotlib.pyplot as plt
import random
import datetime

# %%
# N为人群总数
N_sh = 24870895
N_cn = 693051
# β为传染率系数
beta_sh = 0.6
beta_cn = 0.8
# gamma为恢复率系数
gamma_sh = 0.1
gamma_cn = 0.2
# delta为死亡率系数
delta_sh = 0.0
delta_cn = 0.0
# lambda为潜伏转阳率系数
lambda_sh = 0.1
lambda_cn = 0.3333
# Te为疾病潜伏期
Te = 14
# I_0为感染者的初始人数
I_sh_0 = 1
I_cn_0 = 12
# E_0为潜伏者的初始人数
E_sh_0 = 0
E_cn_0 = 80
# R_0为治愈者的初始人数
R_0 = 0
# S_0为易感者的初始人数
S_sh_0 = N_sh - I_sh_0 - E_sh_0 - R_0
S_cn_0 = N_cn - I_cn_0 - E_cn_0 - R_0
# D_0为死亡的初始人数
D_0 = 0
# T为传播时间
T = 150

# INI为初始状态下的数组
INI_sh = (S_sh_0, E_sh_0, I_sh_0, R_0, D_0)
INI_cn = (S_cn_0, E_cn_0, I_cn_0, R_0, D_0)
# %%


def funcSEIR(inivalue, _, N, beta, gamma, delta, lamb):
    Y = np.zeros(5)
    S, E, I, R, D = inivalue
    # 易感个体变化
    Y[0] = - (beta * S * I) / N
    # 潜伏个体变化（每日有一部分转为感染者）
    Y[1] = (beta * S * I) / N - E * lamb
    # 感染个体变化
    Y[2] = E * lamb - gamma * I * delta - I * gamma
    # 治愈个体变化
    Y[3] = gamma * I
    # 死亡个体变化
    Y[4] = I * gamma * delta
    return Y


def rmse(y_hat, y):
    return np.sqrt(np.sum((y_hat - y) ** 2))


def optim(beta, gamma, lamb, N, E, I, T, y_true,
          delta=0, R=0, D=0, iter=10000000000,
          toleration=10000000):
    beta_lower = beta[0]
    beta_upper = beta[1]
    gamma_lower = gamma[0]
    gamma_upper = gamma[1]
    lamb_lower = lamb[0]
    lamb_upper = lamb[1]
    E_lower = E[0]
    E_upper = E[1]
    I_lower = I[0]
    I_upper = I[1]
    T_range = np.arange(0, T)
    min_rmse = float('inf')
    tol = 1
    for i in range(iter):
        tol += 1
        beta = random.uniform(beta_lower, beta_upper)
        gamma = random.uniform(gamma_lower, gamma_upper)
        lamb = random.uniform(lamb_lower, lamb_upper)
        E = random.randint(E_lower, E_upper)
        I = random.randint(I_lower, I_upper)
        S = N - E - I - R
        INI = (S, E, I, R, D)
        RES = spi.odeint(funcSEIR, INI, T_range, args=(
            N, beta, gamma, delta, lamb))
        loss = rmse(RES[:, 2], y_true)
        print('current iter {:d} loss: {:.3f}, min loss: {:.3f}'.format(
            i+1, loss, min_rmse), end='\r')
        if loss < min_rmse:
            min_rmse = loss
            best_beta = beta
            best_gamma = gamma
            best_lambda = lamb
            best_delta = delta
            best_S = S
            best_E = E
            best_I = I
            best_R = R
            best_D = D
            tol = 1
        if tol >= toleration:
            break
    print('current iter {:d} loss: {:.3f}'.format(i+1, min_rmse))
    print(best_beta, best_gamma, best_lambda, best_delta,
          best_S, best_E, best_I, best_R, best_D)
    return(best_beta, best_gamma, best_lambda, best_delta,
           best_S, best_E, best_I, best_R, best_D)


# %%
y_true = np.array([12, 25, 77, 96, 107, 166, 170, 199,
                   244, 359, 389, 517, 773, 810, 910,
                   1014, 1047, 1131, 1481, 2333, 2947,
                   3704, 4091, 4484, 5617, 6574])
# %%
beta, gamma, lamb, delta, S, E, I, R, D = optim([0.0, 1.0], [0.0, 1.0], [0.0, 1.0], N_cn,
                                                [0, 1000], [12, 12], T=25, y_true=y_true)
# %%
T_range = np.arange(0, T + 1)
INI_cn = (S, E, I, R, D)
RES = spi.odeint(funcSEIR, INI_cn, T_range, args=(
    N_cn, beta, gamma, delta, lamb))
# RES = spi.odeint(funcSEIR, INI_cn, T_range, args=(
#     N_cn, beta_cn, gamma_cn, delta_cn, lambda_cn))

plt.figure(figsize=(16, 9))
plt.plot(RES[:, 0], label='Susceptible', marker='.')
plt.plot(RES[:, 1], label='Exposed', marker='.')
plt.plot(RES[:, 2], label='Infection', marker='.')
plt.plot(RES[:, 3], label='Recovery', marker='.')

plt.title('SEIR Model')
plt.legend()
plt.xlabel('Day')
plt.ylabel('Number')
plt.show()

# %%
base = datetime.datetime.today() - datetime.timedelta(days=1)
date_list = [(base - datetime.timedelta(days=x)).strftime('%Y%m%d')
             for x in range(len(y_true))]
plt.figure(figsize=(16, 9))
plt.plot(date_list[::-1], y_true, label='True Infection', marker='^')
plt.plot(RES[:(len(y_true)+1), 2], label='Infection', marker='.')
plt.xticks(rotation=45)
plt.title('SEIR Model')
plt.legend()
plt.ylabel('Number')
plt.show()

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=date_list[::-1], y=y_true,
                         mode='lines+markers',
                         name='真实'))
next_day = datetime.datetime.today().strftime('%Y%m%d')
fig.add_trace(go.Scatter(x=date_list[::-1]+[next_day], y=RES[:len(y_true)+1, 2],
                         mode='lines+markers',
                         name='预测'))
fig.show()
# %%
