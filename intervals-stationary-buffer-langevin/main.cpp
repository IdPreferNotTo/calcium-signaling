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


class PuffCurrent {
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
    PuffCurrent(int numCha, int numCls, float rateOpnSingle, float rateRef, float rateCls, double ip3, double ca0, double caT) :
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
    const auto kMaxSpikes = desc.get<int>("num parameter.max spikes");
    const auto kMaxTime = desc.get<double>("num parameter.max time");

    //------------------------- Parameters that determine Cell properties ----------------------------------------------
    const auto kNumClu = desc.get<int>("cell.num cluster");
    const auto tau = desc.get<float>("cell.timeconstant");
    const auto j = desc.get<float>("cell.blip current");
    const auto kIp3 = desc.get<float>("cell.ip3");
    const auto kCa0 = desc.get<float>("cell.calcium rest");
    const auto kCaT = desc.get<float>("cell.calcium threshold");

    // ------------------------ Parameters that determine if Ca is fixed -----------------------------------------------
    const auto kBoolCaFixed = desc.get<bool>("calcium fix.on");
    const auto kCaFix = desc.get<double>("calcium fix.value");

    // ------------------------ Parameters that determine IPI density --------------------------------------------------
    const auto kNumCha = desc.get<int>("cluster.num channel");
    const auto kNumCls = desc.get<int>("cluster.number closed states");
    const auto krOpnSingle = desc.get<float>("cluster.rate open single");
    const auto krRef = desc.get<float>("cluster.rate ref");
    const auto krCls = desc.get<float>("cluster.rate close");
    const auto kp = desc.get<float>("buffer.kp");
    const auto km = desc.get<float>("buffer.km");
    const auto bT = desc.get<float>("buffer.bT");

    //------------------------- Parameters that determine Adaptatopm properties ----------------------------------------
    const auto boolAdap = desc.get<bool>("adaptation.on");
    const auto tauA = desc.get<float>("adaptation.timeconstant");
    const auto ampA = desc.get<float>("adaptation.amplitude");

    //-------------------------- Open file for puff dynamics and spike dynamics ----------------------------------------
    std::string path;
    if(output =="local"){
        path = "../out/";
    }
    else{
        path = "/neurophysics/lukasra/Data/";
    }

    char parameters[200];
    if(boolAdap) {
        if(kBoolCaFixed){
            std::sprintf(parameters,
                         "_cafix%.2f_ip%.2f_taua%.2e_ampa%.2e_tau%.2e_j%.2e_K%d_%d.dat",
                         kCaFix, kIp3, tauA, ampA, tau, j, kNumClu, run);
        }
        else {
            std::sprintf(parameters,
                         "_ip%.2f_taua%.2e_ampa%.2e_tau%.2e_j%.2e_K%d_%d.dat",
                         kIp3, tauA, ampA, tau, j, kNumClu, run);
        }
    }
    else{
        if(kBoolCaFixed){
            std::sprintf(parameters,
                         "_cafix%.2f_ip%.2f_tau%.2e_j%.2e_K%d_%d.dat",
                         kCaFix, kIp3, tau, j, kNumClu, run);
        }
        else {
            std::sprintf(parameters, "_bt%.2f_ip%.2f_tau%.2e_j%.2e_K%d_%d.dat",
                         bT, kIp3, tau, j, kNumClu, run);
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
    out_file_spikes = path + "spike_times_langevin_buffer" + parameters;
    std::ofstream file_spikes;
    file_spikes.open(out_file_spikes);
    if (!file_spikes.is_open()) {
        std::cout << "Could not open file at: " << out_file_spikes << std::endl;
        std::cout << "This is where I am: " << std::string(homedir) << std::endl;
        return 1;
    }

    file << "# Parameter" << "\n";
    file << "# dt: " << dt << "\n";
    file << "# r_open: " << krOpnSingle << ", r_close: " << krCls << ", r_refractory:" << krRef << "\n";
    file << "#N_Cluster: " << kNumClu << ", N_Channel: " << kNumCha << ", N_Closed: " << kNumCls << "\n";
    file << "# Ca rest: " << kCa0 << ", Ca Threshold" << kCaT << "\n";
    file << "# t | ca | jpuff_mean | jpuff_std | adap " << std::endl;
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
    int spikeCount = 0;
    double t_since_spike = 0;
    double t_since_print = 0;

    double jLeak;
    double meanJpuff;
    double sqrt2DJpuff;
    double noiseJpuff;
    double stratDrift;

    double dca, dcb;
    double cb = 0;

    double ca = kCa0;
    if(kBoolCaFixed){
        ca = kCaFix;
    }
    double adap = 1.;

    PuffCurrent jpuff(kNumCha, kNumCls, krOpnSingle, krRef, krCls, kIp3, kCa0, kCaT);

    while (t < kMaxTime && spikeCount < kMaxSpikes) {

        rng = dist(generator);
        jLeak = -(ca - kCa0) / tau;
        meanJpuff = j * kNumClu * adap * jpuff.getMean(ca);
        sqrt2DJpuff = j * adap * sqrt(2 * kNumClu * jpuff.getIntensity(ca));
        stratDrift = (1./2.)*pow(j * adap, 2) * kNumClu * jpuff.getDerivativeIntensity(ca);
        noiseJpuff = sqrt2DJpuff * rng;
        if (t_out < t_since_print && t < 10000) {
            file << fixed << setprecision(1) << t << setprecision(4) << " " << ca << " " << meanJpuff << " " << noiseJpuff << " " << adap << " " << cb << "\n";
            t_since_print = 0;
        }

        if(!kBoolCaFixed) {
            //double K = km/kp;
            //double beta = 1 + K*bT/((K + kCa0)*(K + kCa0));
            //dca = (1./beta)*((jLeak + meanJpuff + stratDrift) * dt + noiseJpuff * sqrt(dt));
            dca = (jLeak + meanJpuff + stratDrift -kp*ca*(bT - cb) + km*cb) * dt + noiseJpuff * sqrt(dt);
            dcb = (kp*ca*(bT - cb) - km*cb) * dt;
            ca += dca;
            cb += dcb;
        }

        if(boolAdap){
            adap += (1. - adap) / tauA * dt;
        }

        if (ca >= kCaT) {
            file << fixed << setprecision(1) << t << setprecision(4) << " " << kCaT << " " << 0.000 << " " << 0.000 << " " << adap << " " << cb <<  "\n";
            ca = kCa0;
            cb = kCa0*bT/(km/kp + kCa0);

            if(boolAdap){
                adap += -ampA * adap;
            }
            /*float tRef = 0;
            while(tRef < 10.){
                tRef += dt;
                t += dt;
                if(boolAdap){
                    adap += (1. - adap) / tauA * dt;
                }
            }*/

            file_spikes << setprecision(4) << t_since_spike << " ";
            t_since_spike = 0;
            spikeCount += 1;
        }

        t += dt;
        t_since_print += dt;
        t_since_spike += dt;
    }
    return 0;
}
