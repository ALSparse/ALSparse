import os
import sys
import random
import argparse
import sys

class Matrix:
    rows    = 0
    cols    = 0
    nnz     = 0
    comment = ""
    col_idx = []
    row_idx = []
    val     = []

def uniform_generate(rows, cols, nnz_per_row, file_path):
    mtx = Matrix()

    mtx.rows    = rows
    mtx.cols    = cols
    mtx.nnz     = rows * nnz_per_row
    mtx.comment = "% each row has {} nnz\n".format(nnz_per_row)

    col_idx = []
    row_idx = []
    val     = []

    for r in range(1, rows+1):
        col_idx = col_idx + random.sample(range(1, cols+1), nnz_per_row)
        row_idx = row_idx + [r] * nnz_per_row
        val     = val     + [1] * nnz_per_row

    mtx.row_idx = row_idx
    mtx.col_idx = col_idx
    mtx.val     = val

    if mtx.nnz != len(row_idx) or len(row_idx) != len(col_idx):
        print("nnz != len(row_idx) or len(row_idx) != len(col_idx)")
        sys.exit(0)

    return mtx


def write_matrix(mtx, path):
    file = open(path, 'w')

    file.writelines(mtx.comment)
    file.writelines("{} {} {}\n".format(mtx.rows, mtx.cols, mtx.nnz))

    for i in range(mtx.nnz):
        file.writelines("{} {} {}\n".format(mtx.row_idx[i], mtx.col_idx[i], mtx.val[i]))

    file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='convert mtx file to binary')

    parser.add_argument('start', type=int, help='start idx.')
    parser.add_argument('end',   type=int, help='end idx.')
    args = parser.parse_args()

    start = args.start
    end   = args.end

    print(start, end)
    os.system("echo from {} to {}.".format(start, end))

    for i in range(start, end):
        os.system("echo {} nnzper start.".format(i))

        mtx = uniform_generate(10000, 10000, i, "w")
        # print(mtx.comment)
        # print(mtx.row_idx)
        # print(mtx.col_idx)

        write_matrix(mtx, "/public/home/ictapp_w/xpj/10000_perrow/10000r_{}per.mtx".format(i))

        print("{} nnzper done.".format(i))
        os.system("echo {} nnzper done.".format(i))
