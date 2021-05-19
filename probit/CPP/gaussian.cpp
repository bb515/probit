#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <math.h>
#include <random>
#include <algorithm>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <thread>
#include <future>

namespace py = pybind11;

using Eigen::MatrixXd;
using Eigen::VectorXd;

double trunc_left_std(double left_bound) {
    double alpha = (left_bound + sqrt(pow(left_bound, 2) + 4.0)) / 2.0;

    std::random_device rd;
    std::default_random_engine generator(rd());
    std::exponential_distribution<double> exp_rv(alpha);
    std::uniform_real_distribution<double> uni_rv(0.0, 1.0);

    double z, ro, u;

    while (true) {
        z = exp_rv(generator) + left_bound;
        ro = exp(-pow(alpha - z, 2) / 2);
        u = uni_rv(generator);

        if (u <= ro)
            return z;
    }
}

double trunc_right_std(double right_bound) { return -trunc_left_std(-right_bound); }

double trunc_left(double mu, double sigma2, double left_bound) {
    double sigma = sqrt(sigma2);
    return mu + sigma * trunc_left_std((left_bound - mu) / sigma);
}

double trunc_right(double mu, double sigma2, double right_bound) { return - trunc_left(-mu, sigma2, -right_bound); }

double sample_by_uni(double left_bound, double right_bound) {
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<double> uni_rv(0.0, 1.0);

    double z, ro, u;

    while (true) {
        z = left_bound + (right_bound - left_bound) * uni_rv(generator);

        if (left_bound <= 0 && 0 <= right_bound)
            ro = -pow(z, 2) / 2.0;
        else if (right_bound < 0)
            ro = (pow(right_bound, 2) - pow(z, 2)) / 2.0;
        else
            ro = (pow(left_bound, 2) - pow(z, 2)) / 2.0;

        u = log(uni_rv(generator));

        if (u <= ro)
            return z;
    }
}

double sample_by_exp(double left_bound, double right_bound) {
    double z;

    while (true) {
        z = trunc_left_std(left_bound);

        if (z <= right_bound)
            return z;
    }
}

double sample_by_norm(double left_bound, double right_bound) {
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::normal_distribution<double> norm_rv(0.0, 1.0);
    double z;

    while (true) {
        z = norm_rv(generator);

        if (left_bound <= z && z <= right_bound)
            return z;
    }
}

double norm_cdf(double x) {
    return (1.0 + erf(x * M_SQRT1_2)) / 2.0;
}

double trunc_std(double left_bound, double right_bound) {
    if (left_bound * right_bound < 0) {
        if (right_bound - left_bound < M_SQRT2 * sqrt(M_PI))
            return sample_by_uni(left_bound, right_bound);
        else
            return sample_by_norm(left_bound, right_bound);
    } else {
        double mult = (left_bound >= 0) ? 1.0 : -1.0;

        left_bound *= mult;
        right_bound *= mult;

        if (left_bound > right_bound)
            std::swap(left_bound, right_bound);

        double alpha = (left_bound + sqrt(pow(left_bound, 2) + 4)) / 2.0;

        if (log(alpha) + alpha * left_bound / 2.0 + log(right_bound - left_bound) <= (1 + pow(left_bound, 2)) / 2.0)
            return mult * sample_by_uni(left_bound, right_bound);
        else
            return mult * sample_by_exp(left_bound, right_bound);
    }
}

double truncated(double mu, double sigma2, double left_bound, double right_bound) {
    double sigma = sqrt(sigma2);
    left_bound = (left_bound - mu) / sigma;
    right_bound = (right_bound - mu) / sigma;
    return mu + sigma * trunc_std(left_bound, right_bound);
}


PYBIND11_MODULE(gaussian, m) {
    m.doc() = "Gaussian sampling library"; // optional module docstring

    m.def("truncated", &truncated, "Sampling from a Gaussian truncated on an interval");
}
