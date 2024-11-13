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


class puffCurrent {
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
    puffCurrent(int numCha, int numCls, float rateOpnSingle, float rateRef, float rateCls, double ip3, double ca0, double caT) :
            kNumCha_(numCha), kNumCls_(numCls), kRateOpnSingle_(rateOpnSingle), kRateRef_(rateRef),
            kRateCls_(rateCls), ip3_(ip3), ca0_(ca0), caT_(caT){
        rateOpn_ = kRateOpn_;
        setTransitionMatrix();
        setMeans();
        setIntensities();
        setDerivationsIntesities();
    };

    void updateRateOpen(double ca) {
        rateOpn_ = kRateOpn_ * pow(ca / ca0_, 3) * (1. + pow(ca0_, 3)) / (1. + pow(ca, 3)) * pow(ip3_ / 1., 3) * (1. + pow(1., 3)) / (1. + pow(ip3_, 3));
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
        for(int i = 0; i < n; i++){
            m(n-1, i) = 1.;
        }
        transitionMatrix_ = m;
    }

    void updateRates(double ca){
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
        for(int i = 0; i < n; i++){
            b(i) = 0;
            if(i == n-1){
                b(i) = 1;
            }
        }
        // solve transitionMatrix * p0 = b
        VectorXf p0 = transitionMatrix_.colPivHouseholderQr().solve(b);

        // Define inhomogeneities bf
        MatrixXf bf(n, n);
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                bf(i, j) = p0(i) - delta(i, j);
                if(i == n-1){
                    bf(i, j) = 0;
                }
            }
        }

        // solve tMatrix * f = bf
        MatrixXf f = transitionMatrix_.colPivHouseholderQr().solve(bf);

        // Define values x(i) of the states i in [1, ..., n]
        VectorXf xs(n);
        for(int i = 0; i < n; i++){
            if(i<kNumCls_) {
                xs(i) = 0;
            }
            else{
                int j = i - kNumCls_;
                xs(i) = kNumCha_ - j;
            }
        }

        float D = 0;
        float sum_over_i;
        for(int k = 0; k < n; k++) {
            sum_over_i = 0;
            for (int i = 0; i < n; i++) {
                sum_over_i += xs(i)*f(i, k); //f(i, k) = f(k -> i)
            }
            D += xs(k) * p0(k) * sum_over_i;
        }
        return D;
    }

    void setMeans(){
        // Calculate the mean for 1000 Ca values between 0 and CaT. Later on the mean is picked instead of calculated.
        double mean_ca;
        double dCa = 1./1000.;
        for(int i = 0; i < 501; i++){
            mean_ca = mean(i*dCa);
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

    void setDerivationsIntesities(){
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


    double getMean(double ca) const{
        int idx;
        double ca_at_idx, mean;
        // Find idx. Example: Say ca = 0.7543 then the idx is 754 (with resolution dCa = 0.001)
        double dCa = 1./1000.;
        double imax = 500;
        idx = int(imax*(ca/caT_));
        if(idx <= 0){
            return 0;
        }
        ca_at_idx = idx*dCa;
        // linear interpolation
        mean = means_[idx] + (means_[idx+1] - means_[idx])*(ca - ca_at_idx)/dCa;
        return mean;
    }

    double getIntensity(double ca) const{
        int idx;
        double ca_idx, intensity;
        // Find idx. Example: Say ca = 0.7543 then the idx is 754 (with resolution dCa = 0.001)
        double dCa = 1./1000.;
        double imax = 500;
        idx = int(imax*(ca/caT_));
        if(idx <= 0) {
            return 0;
        }
        ca_idx = idx*dCa;

        // linear interpolation
        intensity = intensities_[idx] + (intensities_[idx+1] - intensities_[idx])*(ca - ca_idx)/dCa;
        return intensity;
    }

    double getDerivativeIntensity(double ca) const{
        int idx;
        double ca_idx, derivative;
        // Find idx. Example: Say ca = 0.7543 then the idx is 754 (with resolution dCa = 0.001)
        double dCa = 1./1000.;
        double imax = 500;
        idx = int(imax*(ca/caT_));
        if(idx <= 0){
            return 0;
        }
        ca_idx = idx*dCa;
        // linear interpolation
        derivative = derivativesIntesities_[idx] + (derivativesIntesities_[idx + 1] - derivativesIntesities_[idx]) * (ca - ca_idx) / dCa;
        return derivative;
    }
};


int main(int argc, char *argv[]) {
    struct passwd *pw = getpwuid(getuid());
    const char *homedir = pw->pw_dir;

    string paraFile = "../parameter/" + string(argv[1]) + ".json";
    pt::ptree desc;
    pt::json_parser::read_json(paraFile, desc);

    // Temporary Parameter
    const auto run = desc.get<int>("num parameter.run");
    const string output = desc.get<string>("num parameter.output");

    // ------------------------ Numerical parameter --------------------------------------------------------------------
    const auto dt = desc.get<double>("num parameter.dt langevin");
    const auto t_out = desc.get<double>("num parameter.t_out");
    const auto k_max_spikes = desc.get<int>("num parameter.max spikes");
    const auto k_max_time = desc.get<double>("num parameter.max time");

    //------------------------- Parameters that determine Cell properties ----------------------------------------------
    const auto k_K = desc.get<int>("cell.num cluster");
    const auto tau = desc.get<float>("cell.timeconstant");
    const auto j = desc.get<float>("cell.blip current");
    const auto k_ip3 = desc.get<float>("cell.ip3");
    const auto k_c0 = desc.get<float>("cell.calcium rest");
    const auto k_cT = desc.get<float>("cell.calcium threshold");

    // ------------------------ Parameters that determine if Ca is fixed -----------------------------------------------
    const auto k_bool_ca_fix = desc.get<bool>("calcium fix.on");
    const auto k_cF = desc.get<double>("calcium fix.value");

    // ------------------------ Parameters that determine IPI density --------------------------------------------------
    const auto k_N = desc.get<int>("cluster.num channel");
    const auto k_M = desc.get<int>("cluster.number closed states");
    const auto k_r_opn_s = desc.get<float>("cluster.rate open single");
    const auto k_r_ref = desc.get<float>("cluster.rate ref");
    const auto k_r_cls = desc.get<float>("cluster.rate close");

    //------------------------- Parameters that determine Adaptatopm properties ----------------------------------------
    const auto k_bool_cer = desc.get<bool>("adaptation.on");
    const auto k_tau_er = desc.get<float>("adaptation.timeconstant");
    const auto k_eps = desc.get<float>("adaptation.amplitude");

    //-------------------------- Open file for puff dynamics and spike dynamics ----------------------------------------
    std::string path;
    if (output == "local") {
        path = "../out/";
    } else {
        path = "/neurophysics/lukasra/Data/";
    }

    char parameters[200];
    if(k_bool_cer) {
        if(k_bool_ca_fix){
            std::sprintf(parameters,
                         "_cafix%.2f_ip%.2f_taua%.2e_ampa%.2e_tau%.2e_j%.2e_K%d_%d.dat",
                         k_cF, k_ip3, k_tau_er, k_eps, tau, j, k_K, run);
        }
        else {
            std::sprintf(parameters,
                         "_ip%.2f_taua%.2e_ampa%.2e_tau%.2e_j%.2e_K%d_%d.dat",
                         k_ip3, k_tau_er, k_eps, tau, j, k_K, run);
        }
    }
    else{
        if(k_bool_ca_fix){
            std::sprintf(parameters,
                         "_cafix%.2f_ip%.2f_tau%.2e_j%.2e_K%d_%d.dat",
                         k_cF, k_ip3, tau, j, k_K, run);
        }
        else {
            std::sprintf(parameters, "_ip%.2f_tau%.2e_j%.2e_K%d_%d.dat",
                         k_ip3, tau, j, k_K, run);
        }
    }

    // -----------------------------------------------------------------------------------------------------------------
    std::string out_file;
    out_file = path + "ca_langevin" + parameters;
    std::ofstream file;
    file.open(out_file);
    if (!file.is_open()) {
        std::cout << "Could not open file at: " << out_file << std::endl;
        std::cout << "This is where I am: " << std::string(homedir) << std::endl;
        return 1;
    }

    std::string out_file_spikes;
    out_file_spikes = path + "spike_times_langevin" + parameters;
    std::ofstream file_spikes;
    file_spikes.open(out_file_spikes);
    if (!file_spikes.is_open()) {
        std::cout << "Could not open file at: " << out_file_spikes << std::endl;
        std::cout << "This is where I am: " << std::string(homedir) << std::endl;
        return 1;
    }

    file << "# Parameter" << "\n";
    file << "# dt: " << dt << "\n";
    file << "# r_open: " << k_r_opn_s << ", r_close: " << k_r_cls << ", r_refractory:" << k_r_ref << "\n";
    file << "#N_Cluster: " << k_K << ", N_Channel: " << k_N << ", N_Closed: " << k_M << "\n";
    file << "# Ca rest: " << k_c0 << ", Ca Threshold" << k_cT << "\n";
    file << "# t | ci | jpuff_mean | jpuff_std | cer " << std::endl;
    //------------------------------------------------------------------------------------------------------------------

    // -------------------------- Noise parameters & random number generator -------------------------------------------
    double rng;
    const double mu = 0.0;
    const double stddev = 1.0;
    std::random_device rd;
    std::mt19937 generator(rd());
    //Better seed from random_device instead of clock in case one runs many simulations in a short periode of time
    std::normal_distribution<double> dist(mu, stddev);
    //------------------------------------------------------------------------------------------------------------------

    double t = 0;
    int spike_count = 0;
    double t_since_spike = 0;
    double t_since_print = 0;


    double j_leak;
    double mean_j_puff;
    double intensity_j_puff;
    double noise_j_puff;
    double strat_drift;

    double ci = k_c0;
    if(k_bool_ca_fix){
        ci = k_cF;
    }
    double cer = 1.;

    puffCurrent jpuff(k_N, k_M, k_r_opn_s, k_r_ref, k_r_cls, k_ip3, k_c0, k_cT);

    while (t < k_max_time && spike_count < k_max_spikes) {

        rng = 0; //dist(generator);
        j_leak = -(ci - k_c0 * cer) / tau;
        mean_j_puff = j * k_K * cer * jpuff.getMean(ci);
        intensity_j_puff = j * cer * sqrt(2 * k_K * jpuff.getIntensity(ci));
        strat_drift = (1. / 2.) * pow(j * cer, 2) * k_K * jpuff.getDerivativeIntensity(ci);
        noise_j_puff = intensity_j_puff * rng;
        if (t_out < t_since_print && t < 10000) {
            file << fixed << setprecision(2) << t << setprecision(4) << " " << ci << " " << mean_j_puff << " " << noise_j_puff << " " << cer << "\n";
            t_since_print = 0;
        }

        if(!k_bool_ca_fix) {
            ci += (j_leak + mean_j_puff + strat_drift) * dt + noise_j_puff * sqrt(dt);
        }

        if(k_bool_cer){
            cer += (1. - cer) / k_tau_er * dt;
        }

        if(t > 10000 && spike_count == 0){
            return 0;
        }

        if (ci >= k_cT) {
            file << fixed << setprecision(1) << t << setprecision(4) << " " << k_cT << " " << 0 << " " << 0 << " " << cer << "\n";
            if(k_bool_cer){
                cer += -k_eps * cer;
            }
            ci = k_c0 * cer;

            file_spikes << setprecision(4) << t_since_spike << " ";
            t_since_spike = 0;
            spike_count += 1;
        }

        t += dt;
        t_since_print += dt;
        t_since_spike += dt;
    }
    return 0;
}
