#include <stdint.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <ios>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "index_engine.hpp"
// struct {};
struct bin_header {
  int64_t rows;  //总行数
  int64_t cols;  //总列数
  int64_t nnzs;  //总nnz个数
  int64_t real_nnz;
  int64_t field_per_nnz;  // complex:2, integer/float(double):1, pattern 0, real:1
  int64_t num_type;       // float(double):0, integer 1;
  int64_t mtx_sym;        // general:0, sym:1, Hermitian:2
  int64_t reserved;
};
class MtxBinWriter {
 private:
  std::string file_path_;
  std::ofstream ofs_;
  bin_header& header_;

 public:
  MtxBinWriter(std::string file_path, bin_header& header)
      : ofs_(file_path, std::ofstream::binary), header_(header) {}

  void write(char* array, int N) {
    if (!ofs_.is_open()) {
      std::cerr << "file not open yet\n" << std::endl;
    }
    ofs_.write(reinterpret_cast<char*>(array), N);
  }
};
class MtxBinReader {
 private:
  std::string file_path_;
  std::ifstream ifs_;

 public:
  MtxBinReader(std::string file_path) : ifs_(file_path, std::ifstream::binary) {}

  void read_header(char* array, int N) {}
};

template <typename T_index, typename T_value>
class MatrixGenerator {
 protected:
  unsigned int seed_;
  int rows_;
  int cols_;
  double density_;
  // minimal distance between two adjcent nz
  int stride_;
  bool generated_row_nnz_ = false;
  bool generated_matrix_ = false;
  int current_row_;
  int min_nnz_row_;
  int max_nnz_row_;
  int64_t total_nnz;
  std::unique_ptr<MtxBinWriter> writer_;
  bin_header header_;

  std::vector<size_t> nnz_per_row_;
  bool nnz_per_row_shuffle_ = true;
  bool nnz_per_row_copied = false;

  std::vector<T_index> row_index_;
  std::vector<T_index> col_index_;
  std::vector<T_value> value_;

  std::unique_ptr<IndexSimulator<T_index>> index_simulator_;
  std::unique_ptr<IDataSimulator<T_value>> value_simulator_;
  std::unique_ptr<IDataSimulator<double>> row_nnz_density_simulator_;

  DisType index_stride_distr_type_;
  DisType row_nnz_distr_type_;

 public:
  MatrixGenerator(int rows, int cols, const std::vector<size_t>& nnz_per_row, int stride = 1)
      : rows_(rows),
        cols_(cols),
        nnz_per_row_(nnz_per_row),
        stride_(stride),
        current_row_(0),
        total_nnz(0),
        nnz_per_row_copied(true),
        row_nnz_distr_type_(DisType::None),
        index_stride_distr_type_(DisType::Const) {
    header_.rows = static_cast<int64_t>(rows);
    header_.cols = static_cast<int64_t>(cols);
    header_.field_per_nnz = static_cast<int64_t>(1);
    if (sizeof(T_value) == 4)
      header_.num_type = static_cast<int64_t>(0);
    else
      header_.num_type = static_cast<int64_t>(1);
    header_.mtx_sym = static_cast<int64_t>(0);
    if (nnz_per_row.size() != rows_) {
      std::cerr << "nnz_per_row.size() != rows_" << std::endl;
      exit(-1);
    }
    // matrix index should be one based
    index_simulator_ = std::make_unique<IndexSimulator<T_index>>(0, static_cast<T_index>(cols_),
                                                                 stride, stride, DisType::Const);
    value_simulator_ = std::make_unique<NormDataSimulator<T_value>>(-1.0f, 1.0f, 0.0, 1.0);
    generated_row_nnz_ = true;
    min_nnz_row_ = *std::min_element(nnz_per_row_.begin(), nnz_per_row_.end());
    max_nnz_row_ = *std::max_element(nnz_per_row_.begin(), nnz_per_row_.end());
    density_ =
        std::accumulate(nnz_per_row_.begin(), nnz_per_row_.end(), 0) * 1.0 / (rows_ * 1.0 * cols_);
  }
  MatrixGenerator(int rows, int cols, const size_t nnz_per_row, int stride = 1)
      : rows_(rows),
        cols_(cols),
        nnz_per_row_(std::vector<size_t>(rows, nnz_per_row)),
        stride_(stride),
        current_row_(0),
        total_nnz(0),
        nnz_per_row_copied(true), 
        row_nnz_distr_type_(DisType::None), 
        index_stride_distr_type_(DisType::Const) {
    header_.rows = static_cast<int64_t>(rows); 
    header_.cols = static_cast<int64_t>(cols); 
    header_.field_per_nnz = static_cast<int64_t>(1);
    if (sizeof(T_value) == 4)
      header_.num_type = static_cast<int64_t>(0);
    else
      header_.num_type = static_cast<int64_t>(1);
    header_.mtx_sym = static_cast<int64_t>(0);
    if (nnz_per_row_.size() != rows_) {
      std::cerr << "nnz_per_row.size() != rows_" << std::endl;
      exit(-1);
    }
    // matrix index should be one based
    index_simulator_ = std::make_unique<IndexSimulator<T_index>>(0, static_cast<T_index>(cols_),
                                                                 stride, stride, DisType::Const);
    value_simulator_ = std::make_unique<NormDataSimulator<T_value>>(-1.0f, 1.0f, 0.0, 1.0);
    generated_row_nnz_ = true;
    min_nnz_row_ = *std::min_element(nnz_per_row_.begin(), nnz_per_row_.end());
    max_nnz_row_ = *std::max_element(nnz_per_row_.begin(), nnz_per_row_.end());
    density_ =
        std::accumulate(nnz_per_row_.begin(), nnz_per_row_.end(), 0) * 1.0 / (rows_ * 1.0 * cols_);
  }
  MatrixGenerator(int rows, int cols, double density, int min_nnz_row, int max_nnz_row,
                  std::unique_ptr<IDataSimulator<double>> row_nnz_density_simulator, int stride = 1)
      : rows_(rows),
        cols_(cols),
        density_(density),
        min_nnz_row_(min_nnz_row),
        max_nnz_row_(max_nnz_row),
        stride_(stride),
        current_row_(0),
        total_nnz(0),
        nnz_per_row_copied(false),
        row_nnz_density_simulator_(std::move(row_nnz_density_simulator)) {
    header_.rows = static_cast<int64_t>(rows);
    header_.cols = static_cast<int64_t>(cols);
    header_.field_per_nnz = static_cast<int64_t>(1);
    if (sizeof(T_value) == 4)
      header_.num_type = static_cast<int64_t>(0);
    else
      header_.num_type = static_cast<int64_t>(1);
    header_.mtx_sym = static_cast<int64_t>(0);
    // matrix index should be one based
    index_simulator_ = std::make_unique<IndexSimulator<T_index>>(0, static_cast<T_index>(cols_),
                                                                 stride, stride, DisType::Const);
    value_simulator_ = std::make_unique<NormDataSimulator<T_value>>(-1.0f, 1.0f, 0.0, 1.0);
    generated_row_nnz_ = false;
    std::cout << "ctor OK\n";
  }

  MatrixGenerator(int rows, int cols, double density, int min_nnz_row, int max_nnz_row,
                  std::unique_ptr<IDataSimulator<double>> row_nnz_density_simulator,
                  std::unique_ptr<IndexSimulator<T_index>> index_simulator, int stride = 1)
      : rows_(rows),
        cols_(cols),
        density_(density),
        min_nnz_row_(min_nnz_row),
        max_nnz_row_(max_nnz_row),
        stride_(stride),
        current_row_(0),
        total_nnz(0),
        nnz_per_row_copied(false),
        row_nnz_density_simulator_(std::move(row_nnz_density_simulator)),
        index_simulator_(std::move(index_simulator)) {
    header_.rows = static_cast<int64_t>(rows);
    header_.cols = static_cast<int64_t>(cols);
    header_.field_per_nnz = static_cast<int64_t>(1);
    if (sizeof(T_value) == 4)
      header_.num_type = static_cast<int64_t>(0);
    else
      header_.num_type = static_cast<int64_t>(1);
    header_.mtx_sym = static_cast<int64_t>(0);
    // matrix index should be one based
    value_simulator_ = std::make_unique<NormDataSimulator<T_value>>(-1.0f, 1.0f, 0.0, 1.0);
    generated_row_nnz_ = false;
  }
  // powerLaw
  // MatrixGenerator(int rows, int cols, double density = 0.005, bool nnz_per_row_shuffle = true,
  //                 DisType row_nnz_distr_type, int min_nnz_row = 1, int max_nnz_row = INT_MAX,
  //                 int stride = 1, float alpha = -2.0f)
  //     : rows_(rows),
  //       cols_(cols),
  //       density_(density),
  //       nnz_per_row_shuffle_(nnz_per_row_shuffle),
  //       stride_(stride),
  //       min_nnz_row_(min_nnz_row),
  //       max_nnz_row_(max_nnz_row),
  //       total_nnz(0),
  //       row_nnz_distr_type_(row_nnz_distr_type),
  //       index_stride_distr_type_(DisType::Const) {
  //   header_.rows = static_cast<int64_t>(rows);
  //   header_.cols = static_cast<int64_t>(cols);
  //   header_.field_per_nnz = static_cast<int64_t>(1);
  //   if (sizeof(T_value) == 4)
  //     header_.num_type = static_cast<int64_t>(0);
  //   else
  //     header_.num_type = static_cast<int64_t>(1);
  //   header_.mtx_sym = static_cast<int64_t>(0);

  //   current_row_ = 0;
  //   seed_ = (std::random_device()());
  //   // matrix index should be one based
  //   index_simulator_ = std::make_unique<IndexSimulator<T_index>>(0, static_cast<T_index>(cols_),
  //                                                                stride, stride, DisType::Const);
  //   value_simulator_ = std::make_unique<NormDataSimulator<T_value>>(-1.0f, 1.0f, 0.0, 1.0);
  //   // powerLaw default
  //   if (row_nnz_distr_type == DisType::Const) {
  //     row_nnz_density_simulator_ = std::make_unique<ConstDataSimulator<double>>(1.0);
  //   } else if (row_nnz_distr_type == DisType::PowerLaw) {
  //     row_nnz_density_simulator_ =
  //         std::make_unique<FloatPowerLawDataSimulator<double>>(density_, 2.0 * density_, alpha);
  //   } else if (row_nnz_distr_type == DisType::Uniform) {
  //     row_nnz_density_simulator_ =
  //         std::make_unique<FloatUniformDataSimulator<double>>(density_, 2.0 * density_);
  //   } else if (row_nnz_distr_type == DisType::Norm) {
  //     row_nnz_density_simulator_ =
  //         std::make_unique<NormDataSimulator<double>>(0, 1000.0, 500.0, 1.0);
  //   }
  // }
  std::vector<size_t>& get_nnz_per_row() { return nnz_per_row_; };

  void initialize_nnz_distribute() {}
  void reset() {
    if (!nnz_per_row_copied) {
      nnz_per_row_.clear();
      generated_row_nnz_ = false;
    }
    current_row_ = 0;
    total_nnz = 0;
    generated_matrix_ = false;
  }
  void next_row() { current_row_++; }
  void generate_nnz_per_row() {
    if (generated_row_nnz_ || nnz_per_row_copied) {
      return;
    }
    std::vector<double> density_row;
    double total_density = 0.;
    for (int i = 0; i < rows_; i++) {
      double d = row_nnz_density_simulator_->next_num();
      density_row.push_back(d);
      total_density += d;
    }
    double scale = 1.0 / total_density;
    for (auto& x : density_row) {
      x = x * scale;
    }
    double nnz_expected = static_cast<double>(rows_ * 1.0 * cols_ * density_);
    std::cout << "expected nnz " << nnz_expected << std::endl;
    for (int i = 0; i < rows_; i++) {
      nnz_per_row_.push_back(static_cast<size_t>(density_row[i] * nnz_expected));
    }
    generated_row_nnz_ = true;
    if (nnz_per_row_shuffle_)
      std::shuffle(nnz_per_row_.begin(), nnz_per_row_.end(), std::mt19937(std::random_device()()));

    std::cout << "computed nnz is " << std::accumulate(nnz_per_row_.begin(), nnz_per_row_.end(), 0)
              << " " << std::endl;
  }
  virtual ~MatrixGenerator() {}
  virtual void generate_next_row(bool diag_centric = false) {
    if (!generated_row_nnz_) {
      std::cerr << "Error: not generate nnz_per_row yet" << std::endl;
      return;
    }
    if (current_row_ >= rows_) {
      total_nnz = row_index_.size();
      std::cerr << "reach final row!!" << std::endl;
      return;
    }

    int nnz_current_row = nnz_per_row_[current_row_];
    if (diag_centric) {
      int radius = (nnz_current_row - 1) / 2;
      T_index starting_index = current_row_ - radius > 0 ? current_row_ - radius : 0;
      index_simulator_->set_current_index(starting_index);
    } else {
      index_simulator_->set_random_index();
    }
    for (int i = 0; i < nnz_current_row; total_nnz++, i++) {
      // std::cout << "row " << current_row_ << " has " << nnz_current_row << " nz\n";
      T_index col_idx = index_simulator_->next_num();
      T_value val = value_simulator_->next_num();
      if (col_idx >= cols_) {
        break;
      }
      // one based index
      row_index_.push_back(static_cast<T_index>(current_row_) + 1);
      col_index_.push_back(static_cast<T_index>(col_idx) + 1);
      value_.push_back(static_cast<T_value>(val));
    }
    current_row_++;
  }
  void generate_matrix(bool diag_centric = false) {
    if (generated_matrix_) {
      std::cout << "reset original matrix first!\n";
      return;
      // reset();
    }
    if (!generated_row_nnz_) {
      generate_nnz_per_row();
    }
    for (int i = 0; i < rows_; i++) {
      generate_next_row(diag_centric);
    }
    generated_matrix_ = true;
    std::cout << "actual nnz is " << total_nnz << std::endl;
  }
  void matrix_to_file(const std::string& path) {
    if (current_row_ != rows_) {
      std::cerr << "does not finish row nnz generations yet\n" << std::endl;
      return;
    }
    header_.nnzs = total_nnz;
    header_.real_nnz = total_nnz;
    writer_ = std::make_unique<MtxBinWriter>(path, header_);
    writer_->write(reinterpret_cast<char*>(&header_), sizeof(bin_header));

    for (int i = 0; i < total_nnz; i++) {
      writer_->write(reinterpret_cast<char*>(row_index_.data() + i), sizeof(T_index));
      writer_->write(reinterpret_cast<char*>(col_index_.data() + i), sizeof(T_index));
      writer_->write(reinterpret_cast<char*>(value_.data() + i), sizeof(T_value));
    }
  }
};
