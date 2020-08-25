import numpy as np
from scipy.linalg import expm
import copy

lambd, mu = 0.02, 2


def f_probability(x):
    return 1/(1 + np.exp(-x))


def removing_probability(A, M, action, theta):
    P, s, vh = np.linalg.svd(A)
    return f_probability(P[action].dot(M).dot(theta))


def newAM(A, M, remove=True):
    if remove:
        P, s, _ = np.linalg.svd(A)


def reward(A, M, action, theta, remove=True):
    n = np.shape(A)[0]
    P, s, _ = np.linalg.svd(A)
    if not remove:
        return sum(f_probability(P[i].dot(np.linalg.expm(lambd * np.diag(s) - mu * np.identity(n))) \
                                 .dot(M).dot(theta)) - f_probability(P[i].dot(M).dot(theta)) for i in range(n))
    else:
        A_tilde = copy.copy(A)
        A_tilde[action] = 0
        A_tilde[:, action] = 0
        P_tilde, s_tilde, _ = np.linalg.svd(A_tilde)
        return sum(f_probability(P_tilde[i].dot(np.linalg.expm(lambd * np.diag(s_tilde) - mu * np.identity(n))) \
                                 .dot(P_tilde).dot(P).dot(M).dot(theta)) - f_probability(P[i].dot(M).dot(theta)) \
                   for i in range(n))


def one_step(A, M, action, theta):
    n = np.shape(A)[0]
    P, s, _ = np.linalg.svd(A)
    p = f_probability(P[action].dot(M).dot(theta))
    remove = np.random.binomial(1, p)
    if not remove:
        A_next = A
        M_next = expm(lambd * np.diag(s) - mu * np.identity(n)).dot(M)
        r = -sum(f_probability(P[i].dot(M_next).dot(theta)) - f_probability(P[i].dot(M).dot(theta)) for i in range(n))
    else:
        A_next = copy.copy(A)
        A_next[action] = 0
        A_next[:, action] = 0
        P_tilde, s_tilde, _ = np.linalg.svd(A_next)
        M_next = expm(lambd * np.diag(s_tilde) - mu * np.identity(n)).dot(P_tilde).dot(P).dot(M)
        r = sum(f_probability(P[i].dot(M_next).dot(theta)) - f_probability(P[i].dot(M).dot(theta)) for i in range(n))
    return A_next, M_next, r, remove


n = 50
n_simulation = 100
T = 50
A = np.ones((n, n))
for i in range(n):
    A[i, i] = 0
_, _, M = np.linalg.svd(A)
feasible_set = [i for i in range(n)]
theta = np.ones((50, 1))
total_reward = [0] * len(feasible_set)

for i, action in enumerate(feasible_set):
    for _ in range(n_simulation):
        A_next, M_next, r, remove = one_step(A, M, action, theta)
        total_reward[i] = +r
        nodes = copy.copy(feasible_set)
        if remove:
            nodes.remove(action)
        for _ in range(T - 1):
            action = np.random.choice(nodes)
            A_next, M_next, r, remove = one_step(A_next, M_next, action, theta)
            total_reward[i] += r
            if remove:
                nodes.remove(action)

print(feasible_set[np.argmax(total_reward)])