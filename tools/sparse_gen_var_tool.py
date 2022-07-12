import math
import random
from random import gauss
import os
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt


class Matrix:
    rows = 0
    cols = 0
    nnz = 0
    comment = ""
    col_idx = []
    row_idx = []
    val = []


mean_list = []
var_list = []


def generate(rows, cols, row_nnzs, mean, var):
    mtx = Matrix()

    mtx.rows = rows
    mtx.cols = cols
    mtx.nnz = 0
    mtx.comment = "% row nnz mean {:*>6}, row nnz val {:*>6}.\n".format(mean, var)

    col_idx = []
    row_idx = []
    val = []

    for r in range(1, rows+1):
        # print(row_nnzs[r-1])
        nnz_per_row = int(np.round(row_nnzs[r-1]))
        mtx.nnz += nnz_per_row
        # print(nnz_per_row)
        col_idx = col_idx + random.sample(range(1, cols+1), nnz_per_row)
        row_idx = row_idx + [r] * nnz_per_row
        val = val + [1] * nnz_per_row

    mtx.row_idx = row_idx
    mtx.col_idx = col_idx
    mtx.val = val

    if mtx.nnz != len(row_idx) or len(row_idx) != len(col_idx):
        print("nnz != len(row_idx) or len(row_idx) != len(col_idx)")
        sys.exit(0)

    return mtx


def write_matrix(mtx, path):
    file = open(path, 'w')

    file.writelines(mtx.comment)
    file.writelines("{} {} {}\n".format(mtx.rows, mtx.cols, mtx.nnz))

    for i in range(mtx.nnz):
        file.writelines("{} {} {}\n".format(
            mtx.row_idx[i], mtx.col_idx[i], mtx.val[i]))

    file.close()


for i in range(15):
    mean_list.append(2 ** (i+1))
    var_list.append(2 ** (i+1))

for mean in mean_list:
    for var in var_list:
        randomval = np.random.normal(mean, math.sqrt(var), 10000)
        print(mean, var)
        print(np.max(randomval), np.min(randomval))

        if np.min(randomval) < 0 or np.max(randomval) > 10000:
            continue
        print(randomval)
        mtx = generate(10000, 10000, randomval, mean, var)
        print("mean, var, cal mean", mean, var, np.mean(randomval.astype(int)))
        # print(mtx.comment)
        # print(mtx.row_idx)
        # print(mtx.col_idx)
    
        write_matrix(
            mtx, "/public/home/ictapp_w/xpj/10000_gauss/10000r_mean{}_var{}.mtx".format(mean, var))

        print("var:{} mean:{} case done.".format(var, mean))
        os.system("echo var:{} mean:{} case done.".format(var, mean))
