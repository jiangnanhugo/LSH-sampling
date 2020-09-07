import numpy as np
import numpy.random as rand
import numpy.linalg as lin
import time
from scipy.special import softmax as sftmx
from lsh_sampling_self_attention import hyperplane_lsh

def softmax(x):
    c = np.tile(np.max(x, axis=1, keepdims=True), x.shape[1])
    exp = np.exp(x - c)
    z = np.tile(np.sum(exp, axis=1, keepdims=True), x.shape[1])
    softmax = exp / z
    return softmax


def normal_softmax(R):
    return softmax(np.dot(R, R.T))


def lsh_softmax(R, seed=32):
    """
    reformer LSH attention implementation.
    :param R: the shared QK matrix
    :param seed: random seef value
    :return:
    """
    rand.seed(seed=seed)

    N = R.shape[0]
    M = R.shape[1]

    A = rand.randn(M, M)
    R1 = np.dot(R, A)  # rotate randomly
    R1 /= (np.tile(lin.norm(R1, axis=1, keepdims=True), M))  # normalize (maps to Sphere)

    h = np.argmax(np.concatenate([-R1, R1], axis=1), axis=1)  # calc hash
    n_bucket = len(np.unique(h))

    sorted_h_idx = np.argsort(h)

    m = 2 * N / n_bucket

    Q = np.zeros((N, N))
    for i in range(int(m)):
        t = int(N // m)
        if i < int(m) - 1:
            sorted_idx = sorted_h_idx[t * i: t * (i + 1)]
        else:
            sorted_idx = sorted_h_idx[t * i:]

        R_small = R[sorted_idx, :]
        tmp = np.dot(R_small, R_small.T)

        Q[np.ix_(sorted_idx, sorted_idx)] = softmax(tmp)

    return Q


if __name__ == "__main__":
    L = 100
    dk = 1024
    Q = np.random.randn(L, dk)
    V = np.random.randn(L, dk)
    start = time.time()
    normal_sm = np.matmul(normal_softmax(Q), V)
    print("numpy implemented self-attention time: {}".format(time.time() - start))
    start = time.time()
    lsh_sm = np.matmul(lsh_softmax(Q), V)
    print("reformer LSH self-attention time: {}".format(time.time() - start))

    start = time.time()
    out = np.matmul(sftmx(np.matmul(Q, np.transpose(Q)), axis=1), V)
    print("scipy self-attention time: {}".format(time.time() - start))
    start = time.time()
    attention = np.zeros((L, dk))
    hyperplane_lsh(Q, Q, V, attention, L, dk)
    print("LSH-sampling self-attention time: {}".format(time.time() - start))