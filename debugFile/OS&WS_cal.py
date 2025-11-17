# this file is prepared for project 026
# Created by iboxl

# M = 100
# K = 100
# N = 100
# m = 10
# n = 10
# k = 10

# M = 784
# K = 1152
# N = 128
# m = 32
# n = 32
# k = 128

M = 3136
K = 24
N = 144
m = 32
n = 16
k = 6


ib = 8
wb = 8
ob = 32

load_w = K * N * wb
load_i = M * K * ib
store_o = (((M*N - m*n)*(K / k)) + m*n) * ob
print(f'{load_w} + {load_i} + {store_o}')
print(load_w + load_i + store_o)

load_w = K * N * M/m * wb
load_i = M * K * N/n * ib
store_o = M*N * ob
print(f'{load_w} + {load_i} + {store_o}')
print(load_w + load_i + store_o)
