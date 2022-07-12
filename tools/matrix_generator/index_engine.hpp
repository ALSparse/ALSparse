#include <stdint.h>

#include <numeric>
#include <random>
template <typename T>
inline T clip_data(const T& x, const T& min_, const T& max_) {
  if (x < min_) return min_;
  if (x > max_) return max_;
  return x;
}

enum class DisType { Uniform, Norm, PowerLaw, Const, None };
template <typename T>
class IDataSimulator {
 public:
  virtual ~IDataSimulator() {}
  virtual T next_num() = 0;
  IDataSimulator(DisType dist_type) : dist_type_(dist_type_) {}
  DisType get_type() { return dist_type_; }

 private:
  DisType dist_type_;
};

template <typename T>
class ConstDataSimulator : public IDataSimulator<T> {
 private:
  const T data_;

 public:
  ConstDataSimulator(const T& data)
      : IDataSimulator<T>(DisType::Const), data_(data) {}
  T next_num() override { return data_; }
};
// only generate integers
// the smaller the alpha is, the larger the variance is, and the larger the
// expectation is.
template <typename T>
class IntPowerLawDataSimulator : public IDataSimulator<T> {
 private:
  std::mt19937 gen_;
  std::uniform_real_distribution<double> dis_;
  float alpha_;
  double min_, max_, offset_;

 public:
  IntPowerLawDataSimulator(const T& min, const T& max, float alpha = -2.0f)
      : IDataSimulator<T>(DisType::PowerLaw),
        gen_(std::random_device()()),
        dis_(0, 1),
        alpha_(alpha) {
    min_ = 1.0;
    max_ = max - min + 1.0;
    offset_ = min - 1.0;
  }
  IntPowerLawDataSimulator(unsigned int seed, const T& min, const T& max,
                           float alpha = -2.0f)
      : IDataSimulator<T>(DisType::PowerLaw),
        gen_(seed),
        dis_(0, 1),
        alpha_(alpha) {
    min_ = 1.0;
    max_ = max - min + 1.0;
    offset_ = min - 1.0;
  }
  T next_num() override {
    double x = dis_(gen_);
    double y = (pow((pow(max_, alpha_ + 1) - pow(min_, alpha_ + 1)) * x +
                        pow(min_, alpha_ + 1),
                    1.0 / (alpha_ + 1.0)));
    return static_cast<T>(clip_data(round(y) + offset_, min_ - 1.0f, max_));
  }
};
template <typename T>
class FloatPowerLawDataSimulator : public IDataSimulator<T> {
 private:
  std::mt19937 gen_;
  std::uniform_real_distribution<double> dis_;
  float alpha_;
  double min_, max_, offset_;

 public:
  FloatPowerLawDataSimulator(const T& min, const T& max, float alpha)
      : IDataSimulator<T>(DisType::PowerLaw),
        gen_(std::random_device()()),
        dis_(0, 1),
        alpha_(alpha) {
    min_ = 1.0;
    max_ = max - min + 1.0;
    offset_ = min - 1.0;
  }
  FloatPowerLawDataSimulator(unsigned int seed, const T& min, const T& max,
                             float alpha)
      : IDataSimulator<T>(DisType::PowerLaw),
        gen_(seed),
        dis_(0, 1),
        alpha_(alpha) {
    min_ = 1.0;
    max_ = max - min + 1.0;
    offset_ = min - 1.0;
  }
  T next_num() override {
    double x = dis_(gen_);
    double y = (pow((pow(max_, alpha_ + 1) - pow(min_, alpha_ + 1)) * x +
                        pow(min_, alpha_ + 1),
                    1.0 / (alpha_ + 1.0)));
    return clip_data(y + offset_, 0., max_);
  }
};

// only generate integers
template <typename T>
class IntUniformDataSimulator : public IDataSimulator<T> {
 private:
  std::mt19937 gen_;
  std::uniform_int_distribution<T> dis_;
  T min_, max_;

 public:
  IntUniformDataSimulator(T min, T max)
      : IDataSimulator<T>(DisType::Uniform),
        gen_(std::random_device()()),
        dis_(min, max),
        max_(max),
        min_(min) {}
  T next_num() override {
    T x = dis_(gen_);
    return clip_data(static_cast<T>(x), min_, max_);
  }
  IntUniformDataSimulator(unsigned int seed, T min, T max)
      : IDataSimulator<T>(DisType::Uniform),
        gen_(seed),
        dis_(min, max),
        max_(max),
        min_(min) {}
};
template <typename T>
class FloatUniformDataSimulator : public IDataSimulator<T> {
 private:
  std::mt19937 gen_;
  std::uniform_real_distribution<T> dis_;
  T min_, max_;

 public:
  FloatUniformDataSimulator(T min, T max)
      : IDataSimulator<T>(DisType::Uniform),
        gen_(std::random_device()()),
        dis_(min, max),
        max_(max),
        min_(min) {}
  T next_num() override {
    T x = dis_(gen_);
    return clip_data(static_cast<T>(x), min_, max_);
  }
  FloatUniformDataSimulator(unsigned int seed, T min, T max)
      : IDataSimulator<T>(DisType::Uniform),
        gen_(seed),
        dis_(min, max),
        max_(max),
        min_(min) {}
};

template <typename T>
class NormDataSimulator : public IDataSimulator<T> {
 private:
  std::mt19937 gen_;
  std::normal_distribution<double> dis_;
  double mu_, sigma_;
  T min_, max_;

 public:
  NormDataSimulator(T min, T max, double mu, double sigma)
      : IDataSimulator<T>(DisType::Norm),
        gen_(std::random_device()()),
        dis_(mu, sigma),
        max_(max),
        min_(min) {}
  T next_num() override {
    double x = dis_(gen_);
    return clip_data(static_cast<T>(x), min_, max_);
  }
  NormDataSimulator(unsigned int seed, T min, T max, double mu, double sigma)
      : IDataSimulator<T>(DisType::Norm),
        gen_(seed),
        dis_(mu, sigma),
        max_(max),
        min_(min) {}
};

template <typename T>
class IndexSimulator : public IDataSimulator<T> {
 private:
  T current_index;
  // this distribution is dedicated to increment generation
  // TODO maybe we need other customized distribution instead of uniform?
  std::unique_ptr<IDataSimulator<T>> increment_simulator;

  // index incremnt boundary
  int min_inc_, max_inc_;

  // index boundary
  T min_, max_;

 public:
  void clear() { current_index = 0; }
  void set_current_index(const T& idx) {
    if (idx > max_ || idx < min_) {
      std::cerr << "set index out of bound\n" << std::endl;
      return;
    }
    current_index = idx;
  }
  void set_random_index() {
    std::mt19937 gen((std::random_device()()));
    std::uniform_int_distribution<T> dis(min_, max_);
    T idx = dis(gen);
    if (idx > max_ || idx < min_) {
      std::cerr << "set index out of bound\n" << std::endl;
      return;
    }
    current_index = idx;
  }
  // IndexSimulator(T min, T max, DisType inc = DisType::PowerLaw, float alpha =
  // -2.0)
  //     : min_(min),
  //       max_(max),
  //       current_index(0),
  //       min_inc_(1),
  //       max_inc_(max),
  //       // avr_inc_(static_cast<T>((min + max) * 0.5f)),
  //       gen_(std::random_device()()),
  // {
  //   if (inc == DisType::Uniform)
  //     increment_simulator = std::make_unique<IntUniformDataSimulator<T>
  //     >(min_inc_, max_inc_);
  //   else if (inc == DisType::Norm) {
  //     double mu = static_cast<double>((min + max) * 0.5f);
  //     double sigma = static_cast<double>(max_inc_ - min_inc_);
  //     increment_simulator = std::make_unique<NormDataSimulator<T> >(min_inc_,
  //     max_inc_, mu, sigma);
  //   } else if (inc == DisType::PowerLaw) {
  //     increment_simulator =
  //         std::make_unique<IntPowerLawDataSimulator<T> >(min_inc_, max_inc_,
  //         alpha);
  //   } else if (inc == DisType::Const) {
  //     increment_simulator = std::make_unique<ConstDataSimulator<T> >();
  //   }
  // }
  IndexSimulator(T min, T max, int min_inc = 1, int max_inc = 1,
                 DisType inc = DisType::PowerLaw, float alpha = -2.0)
      : IDataSimulator<T>(inc),
        min_(min),
        max_(max),
        current_index(0),
        min_inc_(min_inc),
        max_inc_(max_inc) {
    if (inc == DisType::Uniform)
      increment_simulator =
          std::make_unique<IntUniformDataSimulator<T>>(min_inc_, max_inc_);
    else if (inc == DisType::Norm) {
      double mu = static_cast<double>((min + max) * 0.5f);
      double sigma = static_cast<double>(max_inc_ - min_inc_);
      increment_simulator =
          std::make_unique<NormDataSimulator<T>>(min_inc_, max_inc_, mu, sigma);
    } else if (inc == DisType::PowerLaw) {
      increment_simulator = std::make_unique<IntPowerLawDataSimulator<T>>(
          min_inc_, max_inc_, alpha);
    } else if (inc == DisType::Const) {
      increment_simulator = std::make_unique<ConstDataSimulator<T>>(min_inc);
    }
  }

  T next_num() override {
    if (current_index > max_ || current_index < min_) {
      return static_cast<T>(-1);
    }
    int stride = increment_simulator->next_num();
    auto ret_index = current_index;
    current_index = clip_data(current_index + stride, min_, max_);
    return ret_index;
  }
};
