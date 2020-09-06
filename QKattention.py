
import numpy as np
import time
from scipy.special import softmax
# Hypercube LSH for approximate near neighbors
# https://arxiv.org/pdf/1702.05760.pdf
# it claims to works better than hyperplane LSH in larger d
def hyperplane_lsh(Q, K, V, attention, L, dk):
    # num_plane: number of planes
    num_plane = 3
    n_buckets = 2<< (num_plane-1)
    Q_buckets=[[]for _ in range(n_buckets)]
    K_buckets = [[]for _ in range(n_buckets)]
    random_hyperplane = np.random.randn(dk, num_plane)
    rQ = np.dot(Q, random_hyperplane) > 0
    rK = np.dot(K, random_hyperplane) > 0


    for i in range(L):
        key_num=0
        for j, x in enumerate(rQ[i, :]):
            key_num += (x==True) << j
        Q_buckets[key_num].append(i)
    for i in range(L):
        key_num=0
        for j, x in enumerate(rK[i, :]):
            key_num += (x==True) << j
        K_buckets[key_num].append(i)

    for i in range(n_buckets):
        # print("id={}\nQi's={}\nKi's={}".format(i, Q_buckets[i], K_buckets[i]))
        q_vectors = Q[Q_buckets[i],:]
        k_vectors = K[K_buckets[i], :]
        v_vectors = V[K_buckets[i], :]

        block_qk=np.exp(np.dot(q_vectors, np.transpose(k_vectors)))
        Z=1/np.sum(block_qk, axis=-1)
        repeat_Z=np.tile(Z[:, np.newaxis], (1, len(K_buckets[i])))
        normalized=np.multiply(block_qk,repeat_Z)
        qkv = np.matmul(normalized,v_vectors)
        attention[Q_buckets[i], :]=qkv


if __name__ == "__main__":
    L=10000
    dk=512
    Q=np.random.randn(L,dk)
    K=np.random.randn(L,dk)
    V = np.random.randn(L, dk)
    attention = np.zeros((L, dk))
    start= time.time()
    hyperplane_lsh(Q, K, V, attention, L, dk)
    print("LSH-sampling self-attention time: {}".format(time.time() - start))
    start=time.time()
    out=np.matmul(softmax(np.matmul(Q, np.transpose(K)), axis=1),V)
    print(time.time() - start)


