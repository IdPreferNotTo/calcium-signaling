#include <iostream>
#include <pwd.h>
#include <cmath>
#include <random>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;
namespace pt = boost::property_tree;

float delta(int a, int b){
    if(a ==b){
        return 1.;
    }
    else{
        return 0;
    }
}

vector<double> range(double min, double max, size_t N) {
    vector<double> range;
    double delta = (max-min)/double(N-1);
    for(int i=0; i<N; i++) {
        range.push_back(min + i*delta);
    }
    return range;
}

class J_puff {
private:
    const double ip3_;
    const double ca_r_;
    const double ca_t_;
    const int n_;
    const int m_;

    const float r_opn_;
    const float r_ref_;
    float r_opn_ca_;
    float r_ref_ca_;
    const float r_cls_;

    MatrixXf transition_matrix_;
    vector<double> means_;
    vector<double> intensities_;
    vector<double> derivatives_intesities_;

public:
    J_puff(int n, int m, float r_opn_single, float r_ref, float r_cls, double ip3, double ca_r, double ca_t) :
            n_(n), m_(m), r_opn_(n*r_opn_single), r_ref_(r_ref), r_cls_(r_cls), ip3_(ip3), ca_r_(ca_r), ca_t_(ca_t){
        r_opn_ca_ = r_opn_;
        r_ref_ca_ = r_ref_;
        set_transition_matrix();
        set_means();
        set_intensities();
        set_derivations_intesities();
    };

    void update_r_opn(double ca) {
        r_opn_ca_ = r_opn_ * pow(ca / ca_r_, 3) * (1. + pow(ca_r_, 3)) / (1. + pow(ca, 3))
                    * pow(ip3_ / 1., 3) * (1. + pow(1., 3)) / (1. + pow(ip3_, 3));
    }

    void set_transition_matrix() {
        MatrixXf M = ArrayXXf::Zero(n_ + m_, n_ + m_);
        // Fill the matrix column-wise.
        for (int i = 0; i < n_ + m_ ; i++) {
            if (i < m_ - 1) {
                M(i, i) = -r_ref_ca_;
                M(i + 1, i) = r_ref_ca_;
            }
            if (i == m_ - 1) {
                M(i, i) = -r_opn_ca_;
                for (int k = 0; k < n_; k++) {
                    M(i + k + 1, i) = r_opn_ca_ / float(n_);
                }
            }
            if (i > m_ - 1) {
                M(i, i) = -r_cls_;
                if (i + 1 < n_ + m_) {
                    M(i + 1, i) = r_cls_;
                } else {
                    M(0, i) = r_cls_;
                }
            }
        }
        // Finally the last row is subsitute by the normalization condition, i.e. (1, ..., 1).
        for(int i = 0; i < n_ + m_; i++){
            M(n_ + m_ - 1, i) = 1.;
        }
        transition_matrix_ = M;
    }

    void update_rates(double ca){
        update_r_opn(ca);
    }

    double mean(double ca) {
        double t_opn, t_cls;

        update_rates(ca);
        t_opn = (n_ + 1.) / (2. * r_cls_);
        t_cls = 1. / r_opn_ca_ + (m_ - 1.) / r_ref_ca_;
        return (n_ + 2.) / 3 * t_opn / (t_opn + t_cls);
    }

    double intensity(double ca) {
        // D = \sum_i \sum_k x_i x_k f(k->i)p0(k) where p0(k) is the steady state probability and
        // f(k->i) = \int_0^\infty dt [p(i, t | k, 0) - p_0(i)] and p(i, t | k, 0) the probability
        // of state i at time t given state k at time 0.
        update_rates(ca);
        set_transition_matrix();
        // Define inhomogeneity b
        VectorXf b(n_ + m_);
        for(int i = 0; i < n_ + m_; i++){
            b(i) = 0;
            if(i == n_ + m_ - 1){
                b(i) = 1;
            }
        }
        // solve transitionMatrix * p0 = b
        VectorXf p0 = transition_matrix_.colPivHouseholderQr().solve(b);

        // Define inhomogeneities bf
        MatrixXf bf(n_ + m_, n_ + m_);
        for(int i = 0; i < n_ + m_; i++) {
            for(int j = 0; j < n_ + m_; j++) {
                bf(i, j) = p0(i) - delta(i, j);
                if(i == n_ + m_ - 1){
                    bf(i, j) = 0;
                }
            }
        }

        // solve tMatrix * f = bf
        MatrixXf f = transition_matrix_.colPivHouseholderQr().solve(bf);

        // Define values x(i) of the states i in [1, ..., n]
        VectorXf xs(n_ + m_);
        for(int i = 0; i < n_ + m_; i++){
            if(i < m_) {
                xs(i) = 0;
            }
            else{
                xs(i) = n_ + m_ - i;
            }
        }

        float D = 0;
        float sum_over_i;
        for(int k = 0; k < n_ + m_; k++) {
            sum_over_i = 0;
            for (int i = 0; i < n_ + m_; i++) {
                sum_over_i += xs(i)*f(i, k); //f(i, k) = f(k -> i)
            }
            D += xs(k) * p0(k) * sum_over_i;
        }
        return D;
    }

    void set_means(){
        // Calculate the mean for 1000 Ca values between 0 and CaT. Later on the mean is picked instead of calculated.
        double mean_ca;
        double dCa = 1./1000.;
        for(int i = 0; i < 501; i++){
            mean_ca = mean(i*dCa);
            means_.push_back(mean_ca);
        }
    }

    void set_intensities() {
        // Calculate the noise intensities for 1000 Ca values between 0 and CaT. Later on the intensity is picked instead
        // of calculated.
        double intensity_ca;
        double dCa = 1. / 1000.;
        for (int i = 0; i < 501; i++) {
            intensity_ca = intensity(i * dCa);
            intensities_.push_back(intensity_ca);
        }
    }

    void set_derivations_intesities() {
        double derivatives_intesities_ca;
        double d_ca = 1 / 1000.;
        double D_left, D_right;
        for (int i = 0; i < 501; i++) {
            D_left = intensity((i - 1) * d_ca);
            D_right = intensity((i + 1) * d_ca);
            derivatives_intesities_ca = (D_right - D_left) / (2 * d_ca);
            derivatives_intesities_.push_back(derivatives_intesities_ca);
        }
    }

    double get_mean(double ca) const{
        int idx;
        double ca_at_idx, mean;
        // Find idx. Example: Say ca = 0.7543 then the idx is 754 (with resolution dCa = 0.001)
        double dCa = 1./1000.;
        double imax = 500;
        idx = int(imax*(ca/ca_t_));
        ca_at_idx = idx*dCa;

        // linear interpolation
        mean = means_[idx] + (means_[idx+1] - means_[idx])*(ca - ca_at_idx)/dCa;
        return mean;
    }

    double get_intensity(double ca) const{
        int idx;
        double ca_idx, intensity;
        // Find idx. Example: Say ca = 0.7543 then the idx is 754 (with resolution dCa = 0.001)
        double dCa = 1./1000.;
        double imax = 500;
        idx = int(imax*(ca/ca_t_));
        ca_idx = idx*dCa;

        // linear interpolation
        intensity = intensities_[idx] + (intensities_[idx+1] - intensities_[idx])*(ca - ca_idx)/dCa;
        return intensity;
    }

    double get_derivative_intensity(double ca) const{
        int idx;
        double ca_idx, derivative;
        // Find idx. Example: Say ca = 0.7543 then the idx is 754 (with resolution dCa = 0.001)
        double dCa = 1./1000.;
        double imax = 500;
        idx = int(imax*(ca/ca_t_));
        ca_idx = idx*dCa;

        // linear interpolation
        derivative = derivatives_intesities_[idx] + (derivatives_intesities_[idx + 1] - derivatives_intesities_[idx]) * (ca - ca_idx) / dCa;
        return derivative;
    }

    double f_func(double ca, float tau, float j, float cer, int N, float a) const {
        if (ca == 0) {
            return -(ca - ca_r_ * cer) / tau;
        } else {
            double f = get_mean(ca);
            double der_D = get_derivative_intensity(ca);
            return -(ca - ca_r_ * cer) / tau + j * cer * N * f + a * j * j * cer * cer * N * der_D;
        }
    }

    double d_func(double ca, float tau, float j, float cer, int N) const {
        if (ca == 0) {
            return 0;
        } else {
            double D = get_intensity(ca);
            return j * j * cer * cer * N * D;
        }
    }

    double g_func(double ca, float tau, float j, float cer, int N, float a) const {
        double f, d;
        f = f_func(ca, tau, j, cer, N, a);
        d = d_func(ca, tau, j, cer, N);
        return f / d;
    }

    double h_func(double ca, float tau, float j, float cer, int N, float a) const {
        double dca = 0.0001;
        double ca_tmp, g, h;

        h = 0;
        if (ca > ca_r_ * cer) {
            ca_tmp = ca_r_ * cer;
            while (ca_tmp <= ca) {
                g = g_func(ca_tmp, tau, j, cer, N, a);
                h += g * dca;
                ca_tmp += dca;
            }
        }
        if (ca < ca_r_ * cer) {
            ca_tmp = ca;
            while (ca_tmp <= ca_r_ * cer) {
                g = g_func(ca_tmp, tau, j, cer, N, a);
                h -= g * dca;
                ca_tmp += dca;
            }
        }
        return h;
    }

    vector<vector<double>> probability_density(float tau, float j, float cer, int N, float a) const{
        std::vector<double> cas;
        cas = range(0.5, 0.2 * cer, 10001);
        double dca = cas.at(0) - cas.at(1);

        std::vector<double> p0s;
        double integral = 0;
        double h, d;

        for (auto &ca: cas) {
            h = h_func(ca, tau, j, cer, N, a);
            d = d_func(ca, tau, j, cer, N);
            if (ca == 0.5) {
                integral += 0;
            } else {
                if (ca >= ca_r_ * cer) {
                    integral += exp(-h) * dca;
                }
            }
            p0s.push_back(integral * exp(h) / d);
        }

        double sum = 0;
        for(auto& p0 : p0s){
            sum += p0*dca;
        }
        vector<vector<double>> cas_p0s;
        for(int i = 0; i < 14001; i++){
            vector<double> v = {cas[i], p0s[i]/sum};
            cas_p0s.push_back(v);
        }
        return cas_p0s;
    };
};


int main(int argc, char *argv[]) {
    struct passwd *pw = getpwuid(getuid());
    const char *homedir = pw->pw_dir;

    float tau = atof(argv[1]);
    float j = atof(argv[2]);
    float cer = atof(argv[3]);


    float ip3 = 1.;
    float a = 0.5;
    float caR = 0.2;
    float caT = 0.5;
    float r_opn_single = 0.1;
    float r_ref = 20;
    float r_cls = 50;
    int K = 10;
    int N = 5;
    int M = 3;

    J_puff jpuff(N, M, r_opn_single, r_ref, r_cls, ip3, caR, caT);
    vector<vector<double>> cas_p0s = jpuff.probability_density(tau, j, cer,K, a);


    std::string path;
    path = std::string(homedir) + "/Data/calcium/theory/probability_density/";
    char parameters[200];
    std::sprintf(parameters, "p0_ip%.2f_tau%.2e_j%.2e_K%d.dat", ip3, tau, j, K);


    // -----------------------------------------------------------------------------------------------------------------
    std::string out_file;
    out_file = path + parameters;
    std::ofstream file;
    file.open(out_file);
    if (!file.is_open()) {
        std::cout << "Could not open file at: " << out_file << std::endl;
        std::cout << "This is where I am: " << std::string(homedir) << std::endl;
        return 1;
    }
    for(auto& ca_p0: cas_p0s){
        file << ca_p0[0] << " " << ca_p0[1] << "\n";
    }

    //------------------------------------------------------------------------------------------------------------------
    return 0;
}
