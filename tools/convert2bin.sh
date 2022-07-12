mtx_root="/home/zjj/mtx"
files=$(find ${mtx_root} -mindepth 1 -maxdepth 1 -type d)
for file in ${files}; do
  mtx_name=$(basename ${file})
  mtx_file=${mtx_root}/${mtx_name}/${mtx_name}.mtx
  echo "preprocessing ${mtx_file}"
  python3 ./mtx2bin.py ${mtx_file}
done
