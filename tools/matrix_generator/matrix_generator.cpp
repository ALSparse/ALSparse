#include "matrix_generator.hpp"

#include <string>
#include <vector>
int main(int argc, const char* argv[]) {
  int rows = 5000;
  double density = 0.0005;
  double alpha0 = -3.0;

  int stride = 1;
  int min_nnz_row = 1;
  bool diag_centric = false;
  bool nnz_shuffle = true;
  DisType nnz_dist_type = DisType::PowerLaw;
  std::unique_ptr<IDataSimulator<double> > nnz_density_simulator;
  std::vector<std::string> arg_str_arr;

  for (int i = 1; i < argc; i++) {
    arg_str_arr.emplace_back(argv[i]);
  }
  std::string Promt =
      "matrix_generator output distr_type m [n] [density] [diag_centric] [stride] [min_nnz_row] "
      "[max_nnz_row] [nnz_shuffle] \n\talpha for powerLaw is defaulted to -3.0";

  if (arg_str_arr.size() < 3) {
    std::cerr << "Usage : " << Promt << "\n";
    exit(0);
  }
  std::string file_path = arg_str_arr[0];
  rows = stoi(arg_str_arr[2]);

  int cols = rows;
  int max_nnz_row = cols / 2;

  if (arg_str_arr.size() > 3) {
    cols = stoi(arg_str_arr[3]);
  }
  if (arg_str_arr.size() > 4) {
    density = stod(arg_str_arr[4]);
  }
  if (arg_str_arr.size() > 5) {
    if (arg_str_arr[5] == "true") {
      diag_centric = true;
    } else if (arg_str_arr[5] == "false") {
      diag_centric = false;
    } else {
      std::cerr << "Undefined diag_centric value" << std::endl;
    }
  }

  if (arg_str_arr.size() > 6) {
    stride = stoi(arg_str_arr[6]);
  }
  if (arg_str_arr.size() > 7) {
    min_nnz_row = stoi(arg_str_arr[7]);
  }
  if (arg_str_arr.size() > 8) {
    max_nnz_row = stoi(arg_str_arr[8]);
  }
  if (arg_str_arr.size() > 9) {
    if (arg_str_arr[9] == "true") {
      nnz_shuffle = true;
    } else if (arg_str_arr[9] == "false") {
      nnz_shuffle = false;
    } else {
      std::cerr << "Undefined nnz_shuffle value" << std::endl;
    }
  }
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
  std::string distr_type = arg_str_arr[1];
  if (distr_type == "PowerLaw") {
    nnz_dist_type = DisType::PowerLaw;
    nnz_density_simulator =
        std::make_unique<FloatPowerLawDataSimulator<double> >(density, 2.0 * density, alpha0);
  } else if (distr_type == "Norm") {
    nnz_dist_type = DisType::Norm;
    nnz_density_simulator = std::make_unique<NormDataSimulator<double> >(0, 1000.0, 500.0, 1.0);
  } else if (distr_type == "Uniform") {
    nnz_dist_type = DisType::Uniform;
    nnz_density_simulator =
        std::make_unique<FloatUniformDataSimulator<double> >(density, 2.0 * density);
  } else if (distr_type == "Const") {
    nnz_dist_type = DisType::Const;
    nnz_density_simulator = std::make_unique<ConstDataSimulator<double> >(1.0);
  } else {
    std::cerr << "Unsupported dist type\n";
    exit(-1);
  }

  MatrixGenerator<int, double> MG(rows, cols, density, min_nnz_row, max_nnz_row,
                                  std::move(nnz_density_simulator), stride);
  MG.generate_matrix(diag_centric);
  MG.matrix_to_file(file_path);

  return 0;
}