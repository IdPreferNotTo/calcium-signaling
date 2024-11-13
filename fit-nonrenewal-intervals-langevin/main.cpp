#include <iostream>
#include <cmath>
#include <random>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

float delta(int a, int b) {
    if (a == b) {
        return 1.;
    } else {
        return 0;
    }
}


class Puff_current {
private:
    const double ip3_;
    const double ca0_;
    const double caT_;
    const int kNumCha_;
    const int kNumCls_;

    const float kRateOpnSingle_;
    const float kRateOpn_ = float(kNumCha_) * kRateOpnSingle_;
    const float kRateRef_;
    float rateOpn_;
    const float kRateCls_;

    MatrixXf transitionMatrix_;
    vector<double> means_;
    vector<double> intensities_;
    vector<double> derivativesIntesities_;

public:
    Puff_current(int numCha, int numCls, float rateOpnSingle, float rateRef, float rateCls, double ip3, double ca0,
                 double caT) :
            kNumCha_(numCha), kNumCls_(numCls), kRateOpnSingle_(rateOpnSingle), kRateRef_(rateRef),
            kRateCls_(rateCls), ip3_(ip3), ca0_(ca0), caT_(caT) {
        rateOpn_ = kRateOpn_;
        setTransitionMatrix();
        setMeans();
        setIntensities();
        setDerivationsIntesities();
    };

    void updateRateOpen(double ca) {
        rateOpn_ = kRateOpn_ * pow(ca / ca0_, 3) * (1. + pow(ca0_, 3)) / (1. + pow(ca, 3)) * pow(ip3_ / 1., 3) *
                   (1. + pow(1., 3)) / (1. + pow(ip3_, 3));
    }

    void setTransitionMatrix() {
        int n = kNumCha_ + kNumCls_;
        MatrixXf m = ArrayXXf::Zero(n, n);
        // Fill the matrix column-wise.
        for (int i = 0; i < n; i++) {
            if (i < kNumCls_ - 1) {
                m(i, i) = -kRateRef_;
                m(i + 1, i) = kRateRef_;
            }
            if (i == kNumCls_ - 1) {
                m(i, i) = -rateOpn_;
                for (int k = 0; k < kNumCha_; k++) {
                    m(i + k + 1, i) = rateOpn_ / float(kNumCha_);
                }
            }
            if (i > kNumCls_ - 1) {
                m(i, i) = -kRateCls_;
                if (i + 1 < n) {
                    m(i + 1, i) = kRateCls_;
                } else {
                    m(0, i) = kRateCls_;
                }
            }
        }
        // Finally the last row is subsitute by the normalization condition, i.e. (1, ..., 1).
        for (int i = 0; i < n; i++) {
            m(n - 1, i) = 1.;
        }
        transitionMatrix_ = m;
    }

    void updateRates(double ca) {
        updateRateOpen(ca);
    }

    double mean(double ca) {
        double tOpn, tCls;

        updateRates(ca);
        tOpn = (kNumCha_ + 1.) / (2. * kRateCls_);
        tCls = 1. / rateOpn_ + (kNumCls_ - 1.) / kRateRef_;
        return (kNumCha_ + 2.) / 3 * tOpn / (tOpn + tCls);
    }

    double intensity(double ca) {
        // D = \sum_i \sum_k x_i x_k f(k->i)p0(k) where p0(k) is the steady state probability and
        // f(k->i) = \int_0^\infty dt [p(i, t | k, 0) - p_0(i)] and p(i, t | k, 0) the probability
        // of state i at time t given state k at time 0.
        updateRates(ca);
        setTransitionMatrix();
        const int n = kNumCha_ + kNumCls_;
        // Define inhomogeneity b
        VectorXf b(n);
        for (int i = 0; i < n; i++) {
            b(i) = 0;
            if (i == n - 1) {
                b(i) = 1;
            }
        }
        // solve transitionMatrix * p0 = b
        VectorXf p0 = transitionMatrix_.colPivHouseholderQr().solve(b);

        // Define inhomogeneities bf
        MatrixXf bf(n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                bf(i, j) = p0(i) - delta(i, j);
                if (i == n - 1) {
                    bf(i, j) = 0;
                }
            }
        }

        // solve tMatrix * f = bf
        MatrixXf f = transitionMatrix_.colPivHouseholderQr().solve(bf);

        // Define values x(i) of the states i in [1, ..., n]
        VectorXf xs(n);
        for (int i = 0; i < n; i++) {
            if (i < kNumCls_) {
                xs(i) = 0;
            } else {
                int j = i - kNumCls_;
                xs(i) = kNumCha_ - j;
            }
        }

        float D = 0;
        float sum_over_i;
        for (int k = 0; k < n; k++) {
            sum_over_i = 0;
            for (int i = 0; i < n; i++) {
                sum_over_i += xs(i) * f(i, k); //f(i, k) = f(k -> i)
            }
            D += xs(k) * p0(k) * sum_over_i;
        }
        return D;
    }

    void setMeans() {
        // Calculate the mean for 1000 Ca values between 0 and CaT. Later on the mean is picked instead of calculated.
        double mean_ca;
        double dCa = 1. / 1000.;
        for (int i = 0; i < 501; i++) {
            mean_ca = mean(i * dCa);
            means_.push_back(mean_ca);
        }
    }

    void setIntensities() {
        // Calculate the noise intensities for 1000 Ca values between 0 and CaT. Later on the intensity is picked instead
        // of calculated.
        double intensity_ca;
        double dCa = 1. / 1000.;
        for (int i = 0; i < 501; i++) {
            intensity_ca = intensity(i * dCa);
            intensities_.push_back(intensity_ca);
        }
    }

    void setDerivationsIntesities() {
        double derivativesIntesities_ca;
        double dCa = 1 / 1000.;
        double D_left, D_right;
        for (int i = 0; i < 501; i++) {
            D_left = intensity((i - 1) * dCa);
            D_right = intensity((i + 1) * dCa);
            derivativesIntesities_ca = (D_right - D_left) / (2 * dCa);
            derivativesIntesities_.push_back(derivativesIntesities_ca);
        }
    }


    double getMean(double ca) const {
        int idx;
        double ca_at_idx, mean;
        // Find idx. Example: Say ca = 0.7543 then the idx is 754 (with resolution dCa = 0.001)
        double dCa = 1. / 1000.;
        double imax = 500;
        idx = int(imax * (ca / caT_));
        if (idx <= 0) {
            return 0;
        }
        ca_at_idx = idx * dCa;
        // linear interpolation
        mean = means_[idx] + (means_[idx + 1] - means_[idx]) * (ca - ca_at_idx) / dCa;
        return mean;
    }

    double getIntensity(double ca) const {
        int idx;
        double ca_idx, intensity;
        // Find idx. Example: Say ca = 0.7543 then the idx is 754 (with resolution dCa = 0.001)
        double dCa = 1. / 1000.;
        double imax = 500;
        idx = int(imax * (ca / caT_));
        if (idx <= 0) {
            return 0;
        }
        ca_idx = idx * dCa;

        // linear interpolation
        intensity = intensities_[idx] + (intensities_[idx + 1] - intensities_[idx]) * (ca - ca_idx) / dCa;
        return intensity;
    }

    double getDerivativeIntensity(double ca) const {
        int idx;
        double ca_idx, derivative;
        // Find idx. Example: Say ca = 0.7543 then the idx is 754 (with resolution dCa = 0.001)
        double dCa = 1. / 1000.;
        double imax = 500;
        idx = int(imax * (ca / caT_));
        if (idx <= 0) {
            return 0;
        }
        ca_idx = idx * dCa;
        // linear interpolation
        derivative = derivativesIntesities_[idx] +
                     (derivativesIntesities_[idx + 1] - derivativesIntesities_[idx]) * (ca - ca_idx) / dCa;
        return derivative;
    }
};


int main(int argc, char *argv[]) {
    // ------------------------ Numerical parameter --------------------------------------------------------------------
    const auto dt = 0.01;

    float tau_i = atof(argv[1]);
    float j = atof(argv[2]);
    float tau_er = atof(argv[3]);
    float eps = atof(argv[4]);

    //------------------------- Parameters that determine Cell properties ----------------------------------------------
    const auto K = 10;
    const int N = 5;
    const int M = 3;
    const float r_opn_single = 0.1;
    const float r_ref = 20.;
    const float r_cls = 50.;

    const float ip3 = 1.0;
    const float ciR = 0.2;
    const float ciT = 0.5;

    // -------------------------- Noise parameters & random number generator -------------------------------------------
    double rng;
    const double mu = 0.0;
    const double stddev = 1.0;
    std::random_device rd;
    std::mt19937 generator(rd());
    //Better seed from random_device instead of clock in case one runs many simulations in a short periode of time
    std::normal_distribution<double> dist(mu, stddev);
    //------------------------------------------------------------------------------------------------------------------
    Puff_current j_puff(N, M, r_opn_single, r_ref, r_cls, ip3, ciR, ciT);
    double j_leak = 0;
    double m_j_puff = 0; // mean
    double sqrt2d_j_puff = 0; // sqrt 2 of the puff current
    double n_j_puff = 0; // noise
    double s_drift_j_puff = 0; // stratonovich drift

    const int n_max = 1000;
    const int s_max = 25;
    double interspike_intervals[n_max][s_max];

    for (int i = 0; i < n_max; i++) {
        for (int ii = 0; ii < s_max; ii++) {
            interspike_intervals[i][ii] = 0;
        }
    }

    for (int i = 0; i < n_max; i++) {
        double t = 0;
        double t_since_spike = 0;
        double ci = 0.2;
        double cer = 1.;

        int ii = 0;
        while (ii < s_max) {
            rng = dist(generator);
            j_leak = -(ci - ciR  * cer) / tau_i;
            m_j_puff = j * float(K) * cer * j_puff.getMean(ci);
            sqrt2d_j_puff = j * cer * sqrt(2 * K * j_puff.getIntensity(ci));
            s_drift_j_puff = (1. / 2.) * pow(j * cer, 2) * K * j_puff.getDerivativeIntensity(ci);
            n_j_puff = sqrt2d_j_puff * rng;

            ci += (j_leak + m_j_puff + s_drift_j_puff) * dt + n_j_puff * sqrt(dt);
            cer += -(cer - 1.) / tau_er * dt;

            if (ci >= ciT) {
                cer += -eps * cer;
                ci = ciR * cer;
                interspike_intervals[i][ii] = t_since_spike;
                t_since_spike = 0;
                ii += 1;
            }

            t += dt;
            t_since_spike += dt;
        }
    }

    for (int i = 0; i < n_max; i++){
        for(int ii = 0; ii < s_max; ii++){
            cout << interspike_intervals[i][ii] << " ";
        }
        cout << "\n";
    }

    return 0;
}

