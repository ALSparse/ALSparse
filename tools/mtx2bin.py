import sys
import os
import argparse
import logging
from typing import Pattern
import numpy as np
parser = argparse.ArgumentParser(description='convert mtx file to binary')

parser.add_argument('input', type=str, help='src mtx name.')
parser.add_argument('-output', type=str, help='dst mtx name.')
args = parser.parse_args()

src_path = args.input
dst_path = args.output
mtx_name = os.path.basename(src_path)
src_dir = os.path.dirname(src_path)

if(dst_path is None):
    dst_path = src_dir + "/" + mtx_name.split(".")[0] + ".bin"

if(not os.path.exists(src_path)):
    logging.error(u"输入矩阵不存在")
    exit()

mtx_file = open(src_path, "r")
out_file = open(dst_path, "wb")
'''
%%MatrixMarket matrix array complex general
%%MatrixMarket matrix array real general
%%MatrixMarket matrix coordinate complex general
%%MatrixMarket matrix coordinate complex Hermitian
%%MatrixMarket matrix coordinate complex symmetric
%%MatrixMarket matrix coordinate integer general
%%MatrixMarket matrix coordinate integer symmetric
%%MatrixMarket matrix coordinate pattern general
%%MatrixMarket matrix coordinate pattern symmetric
%%MatrixMarket matrix coordinate real general
%%MatrixMarket matrix coordinate real skew-symmetric
%%MatrixMarket matrix coordinate real symmetric
'''

'''

定义header:
{
  int64 rows; //总行数
  int64 cols; //总列数
  int64 nnzs; //有效nnz个数
  int64 real_nnz; //总共的nnz个数
  int64 field_per_nnz; //complex:2, integer/float:1, pattern 0, real:1
  int64 num_type; //float:0, integer 1;
  int64 mtx_sym;  //general:0, sym:1, Hermitian:2 
  int64 reserved;
}

filed_per_nnz:
  complex:2列浮点
  real:1列浮点
  integer:1列整数
  pattern:0列

定义每一个nnz
{
  int32 row;
  int32 col;
  num_type val[filed_per_nnz];
}
'''
num_type = {"complex": 0, "real": 0, "integer": 1, "pattern": 0}
field_per_nnz = {"complex": 2, "real": 1, "integer": 1, "pattern": 0}
# pattern_type = {"integer":lambda x:np.int64(x),
#                 "complex":lambda x:np.float64(x),
#                 "real":lambda x:np.float64(x),
#                 "pattern":None}
pattern_type = {"integer": "int64",
                "complex": "float64",
                "real": "float64",
                "pattern": None}
first_line = mtx_file.readline().strip().split(" ")
if(first_line[0].startswith('%')):

    if(first_line[-3] != "coordinate"):
        logging.error(u"输入不是矩阵")
        exit()
    mtx_pattern = first_line[-2]
    mtx_sym = first_line[-1]
    print(mtx_name, mtx_pattern, mtx_sym)
else:
    mtx_pattern = "real"
    mtx_sym = "general"
    print(mtx_name, mtx_pattern, mtx_sym)
    mtx_file.seek(0)

# 找到第一个非注释的行
# rows,cols,lines
rows = -1
cols = -1
lines = -1
while True:
    line = mtx_file.readline()
    if(line[0] != "%"):
        arr = [int(num) for num in line.split(" ")] + [0]
        np_arr = np.array(arr, dtype='int64')
        rows, cols, lines = np_arr[0], np_arr[1], np_arr[2]
        print(rows, cols, lines)
        break

header_np = np.concatenate((np_arr[0:3], np.array([0]), np.array(
    [field_per_nnz[mtx_pattern], num_type[mtx_pattern]]), np.zeros(2).astype("int64")))
nnzs = lines
if(mtx_sym == "general"):
    header_np[6] = 0
elif (mtx_sym == "Hermitian"):
    header_np[6] = 2
else:
    header_np[6] = 1
print("(rows, cols, nnz, realnnz, value per nnz, numerical type, pattern )")
print(header_np)
out_file.write(header_np.tobytes())
for i in range(lines):
    line = mtx_file.readline().strip().split(" ")
    ori = np.array(line[0:2]).astype("int32")
    sym = np.array(list(reversed(line[0:2]))).astype("int32")
    val = np.array(line[2:]).astype(pattern_type[mtx_pattern])
    # print(ori,val)
    # nnz: (rows,cols)
    out_file.write(ori.tobytes())
    # nnz: (rows,cols,val[0],val[1])
    out_file.write(val.tobytes())
    if(header_np[6] != 0 and ori[0] != ori[1]):
        nnzs += 1
# 可以更新nnz个数
print("real nnz", nnzs)
header_np[3] = nnzs
out_file.seek(24, 0)
out_file.write(nnzs.tobytes())

mtx_file.close()
out_file.close()
