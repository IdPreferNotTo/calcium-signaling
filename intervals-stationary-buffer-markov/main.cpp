#include <iostream>
#include <vector>
#include <random>
#include <pwd.h>
#include <cmath>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

class Cluster {
private:
    const int kNumCha_;
    const int kNumClosedStates_;
    const float kRateOpenSingle_;
    const float kRateOpen_ = float(kNumCha_) * kRateOpenSingle_;
    const float kRateRef_;
    const float kRateClose_;
    const float kCaRest_;
    const float kIp3_;
    int kIdx_;

    static std::random_device rd;
    static unsigned seed;
    static std::mt19937 generator;
    std::uniform_int_distribution<int> intDist_;

    double rateOpen_ = kRateOpen_;
    double rateRef_ = kRateRef_;
    int state_ = 0;
public:
    Cluster(const int idx, const int numCha, const int numClosedStates, const float rateOpenSingle,
            const float rateRef, const float rateClose, const float Ip3, const float caRest) :
            kIdx_(idx), kNumCha_(numCha), kNumClosedStates_(numClosedStates), kRateOpenSingle_(rateOpenSingle),
            kRateRef_(rateRef), kRateClose_(rateClose), kIp3_(Ip3), kCaRest_(caRest),
            intDist_(std::uniform_int_distribution<>(1, numCha)) {
    }

    int state() const {
        return state_;
    }

    int idx() const {
        return kIdx_;
    }

    void update_rate_open(double Ca) {
        rateOpen_ = kRateOpen_ * pow(Ca / kCaRest_, 3.) * (1 + pow(kCaRest_, 3.)) / (1 + pow(Ca, 3.)) * pow(kIp3_ / 1., 3) * (1. + pow(1., 3)) / (1. + pow(kIp3_, 3));
    }

    void open() {
        // opens a random of channels between drawn from a uniform distribution between 1 and n.
        state_ = intDist_(generator);
    }

    void close() {
        if (state_ != 1) {
            state_ -= 1;
        } else {
            state_ = -(kNumClosedStates_-1);
        }
    }

    void refractory() {
        state_ += 1;
    }

    double rate_open() const {
        // rate_open is the cumulative transition probability to leave the closed state (x_i = 0) into ANY open state
        return rateOpen_;
    }

    double rate_close() const {
        // rate_close is the transition probability from n open channel to n-1 open channel (x_i = n -> x_i=n-1)
        return kRateClose_;
    }

    double rate_ref() const {
        // rate_ref is the transition probability from the m'th to m'th + 1 refractory state
        return rateRef_;
    }


};

std::random_device Cluster::rd;
unsigned Cluster::seed = rd();
std::mt19937 Cluster::generator(Cluster::seed);

class Observer {
private:
    const double &t_;
    std::ofstream &file_;
public:
    Observer(const double &t, std::ofstream &file) : t_(t), file_(file) {

    }

    void write(int state, int idx, int n) {
        file_ << std::setprecision(10) << t_ << " " << state << " " << idx << " " << n << "\n";
    }
};

using namespace std;
namespace pt = boost::property_tree;

int main(int argc, char *argv[]) {
    // Average interpuff interval depends on Ip3, ci and the number of channel. We assume that the IPIs measured
    // represent an ensemble average (i.e. channel number n = <n>), at the resting [ci^+] (ci = 0.2 = Ca_rest/Kca)
    // with intermediate IP3 concentration (ip3 = 1 = Ip3/Kip3).
    // The average interpuff interval however still differs among different cell types.
    struct passwd *pw = getpwuid(getuid());
    const char *homedir = pw->pw_dir;

    string paraFile = "../parameter/" + string(argv[1]) + ".json";
    pt::ptree desc;
    pt::json_parser::read_json(paraFile, desc);

    // Temporary Parameter
    const auto run = desc.get<int>("num parameter.run");
    const string output  = desc.get<string>("num parameter.output");
    const bool output_puff = desc.get<bool>("num parameter.output puff");

    // ------------------------ Numerical parameter --------------------------------------------------------------------
    const auto dt = desc.get<double>("num parameter.dt");
    const auto t_out = desc.get<double>("num parameter.t_out");
    const auto kMaxSpikes = desc.get<int>("num parameter.max spikes");
    const auto kMaxTime = desc.get<double>("num parameter.max time");

    // ------------------------ Fixed Parameters for all Simulations ---------------------------------------------------
    const auto kIp3 = desc.get<float>("cell.ip3");
    const auto kCa0 = desc.get<float>("cell.calcium rest");
    const auto kCaT = desc.get<float>("cell.calcium threshold");

    // ------------------------ Parameters that determine if ci is fixed -----------------------------------------------
    const auto kBoolCaFixed = desc.get<bool>("calcium fix.on");
    const auto kCaFix = desc.get<double>("calcium fix.value");

    // ------------------------ Parameters that determine IPI density --------------------------------------------------
    const auto kNumCha = desc.get<int>("cluster.num channel");
    const auto kNumCls = desc.get<int>("cluster.number closed states");
    const auto krOpnSingle = desc.get<float>("cluster.rate open single");
    const auto krRef = desc.get<float>("cluster.rate ref");
    const auto krCls = desc.get<float>("cluster.rate close");

    //------------------------- Parameters that determine Cell properties ----------------------------------------------
    const auto kNumClu = desc.get<int>("cell.num cluster");
    const auto tau = desc.get<float>("cell.timeconstant");
    const auto j = desc.get<float>("cell.blip current");
    const auto kp = desc.get<float>("buffer.kp");
    const auto km = desc.get<float>("buffer.km");
    const auto bT = desc.get<float>("buffer.bT");

    //------------------------- Parameters that determine Adaptatopm properties ----------------------------------------
    const auto boolAdap = desc.get<bool>("adaptation.on");
    const auto tauA = desc.get<float>("adaptation.timeconstant");
    const auto eps = desc.get<float>("adaptation.amplitude");

    // ------------------------ Parameters for output file -------------------------------------------------------------
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
                         kCaFix, kIp3, tauA, eps, tau, j, kNumClu, run);
        }
        else {
            std::sprintf(parameters,
                         "_bt%.2f_ip%.2f_taua%.2e_ampa%.2e_tau%.2e_j%.2e_K%d_%d.dat",
                         bT, kIp3, tauA, eps, tau, j, kNumClu, run);
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

    //open file for puff dynamics and spike dynamics
    std::string out_file_puff;
    out_file_puff = path + "puff_markov" + parameters;
    std::ofstream filePuff;
    filePuff.open(out_file_puff);
    if (!filePuff.is_open()) {
        std::cout << "Could not open file at: " << out_file_puff << std::endl;
        std::cout << "This is where I am: " << std::string(homedir) << std::endl;
        return 1;
    }
    filePuff << "# Parameter" << "\n";
    filePuff << "# dt: " << dt << "\n";
    filePuff << "# r_open: " << krOpnSingle << ", r_close: " << krCls << ", r_refractory:" << krRef << "\n";
    filePuff << "# N_Clu: " << kNumClu << ", N_Cha: " << kNumCha <<  ", N_Closed: " << kNumCls << "\n";
    filePuff << "# ci rest: " << kCa0 << ", ci Threshold" << 1 << "\n";
    filePuff << "# t | state | idx " << std::endl;


    std::string out_file_calcium;
    out_file_calcium = path + "ca_markov" + parameters;
    std::ofstream fileCalcium;
    fileCalcium.open(out_file_calcium);
    if (!fileCalcium.is_open()) {
        std::cout << "Could not open file at: " << out_file_calcium << std::endl;
        std::cout << "This is where I am: " << std::string(homedir) << std::endl;
        return 1;
    }
    fileCalcium << "# Parameter" << "\n";
    fileCalcium << "# dt: " << dt << "\n";
    fileCalcium << "# r_open: " << krOpnSingle << ", r_close: " << krCls << ", r_refractory:" << krRef << "\n";
    fileCalcium << "# N_Clu: " << kNumClu << ", N_Cha: " << kNumCha << ", N_Closed: " << kNumCls << "\n";
    fileCalcium << "# ci rest: " << kCa0 << ", ci Threshold" << 1 << "\n";
    fileCalcium << "# t | ci | j_puff | adap |" << std::endl;

    std::string out_file_spikes;
    out_file_spikes = path + "spike_times_markov_buffer" + parameters;
    std::ofstream file_spikes;
    file_spikes.open(out_file_spikes);
    if (!file_spikes.is_open()) {
        std::cout << "Could not open file at: " << out_file_spikes << std::endl;
        std::cout << "This is where I am: " << std::string(homedir) << std::endl;
        return 1;
    }

    // ------------------------ Random number generator ----------------------------------------------------------------
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<double> uniform_double(0, 1);
    //------------------------------------------------------------------------------------------------------------------
    std::vector<Cluster> clusters;
    for (int i = 0; i < kNumClu; i++) {
        //int numCha = poisson_int(generator);
        //numCha += 1;
        int numCha = kNumCha;
        Cluster cluster(i, numCha, kNumCls, krOpnSingle, krRef, krCls, kIp3, kCa0);
        clusters.push_back(cluster);
    }

    int countSpikes = 0;

    double t = 0;
    double t_since_print = 0;
    double t_since_spike = 0;
    double tRef = 0;
    double tNormalization;

    double ci;
    double cb = 0;
    double dca;
    double dcb;

    if(kBoolCaFixed){
        ci = kCaFix;
    }
    else{
        ci = kCa0;
    }

    double cer = 1;
    double jleak;
    double jpuff;


    double transitionProbability;
    double rnd;

    int openChannel;
    int sumOpenChannel = 0;


    Observer observer(t, filePuff);


    for (auto &cluster: clusters) {
        cluster.update_rate_open(ci);
    }

    while (t < kMaxTime && countSpikes < kMaxSpikes) {
        // Main Loop
        openChannel = 0;

        for (auto &cluster: clusters) {
            //  Calculate transitions for every cluster
            rnd = uniform_double(generator);
            if (cluster.state() < 0) {
                transitionProbability = cluster.rate_ref();
                if (rnd < transitionProbability * dt) {
                    cluster.refractory();
                    if(output_puff && t < 5000){
                        int n = 0;
                        for(auto &ii: clusters){
                            if(ii.state() >= 0){
                                n += ii.state();
                            }
                        }
                        observer.write(cluster.state(), cluster.idx(), n);
                    }
                }
            }
            else if (cluster.state() == 0) {
                transitionProbability = cluster.rate_open();
                if (rnd < transitionProbability * dt) {
                    cluster.open();
                    int n = 0;
                    for(auto &ii: clusters){
                        if(ii.state() >= 0){
                            n += ii.state();
                        }
                    }
                    if(output_puff && t < 5000) {
                        observer.write(cluster.state(), cluster.idx(), n);
                    }
                }
            }
            else {
                transitionProbability = cluster.rate_close();
                if (rnd < transitionProbability * dt) {
                    cluster.close();
                    int n = 0;
                    for(auto &ii: clusters){
                        if(ii.state() >= 0){
                            n += ii.state();
                        }
                    }
                    if(output_puff && t < 5000) {
                        observer.write(cluster.state(), cluster.idx(), n);
                    }
                }
            }
        }

        for (auto &cluster: clusters) {
            // Count number of open channel
            if (cluster.state() > 0) {
                openChannel += cluster.state();
            }
        }

        //  Calculate cell wide Calcium dynamics
        jleak = -(ci - kCa0 * cer) / tau;
        jpuff = j * float(openChannel) * cer;
        if(!kBoolCaFixed){
            dca = (jleak + jpuff - kp * ci * (bT - cb) + km * cb) * dt;
            dcb = (kp * ci * (bT - cb) - km * cb) * dt;
            ci += dca;
            cb += dcb;

        }
        if(boolAdap){
            cer += (1. - cer) / tauA * dt;
        }
        // Update opening rate according to the current ci concentration
        for (auto &cluster: clusters) {
            cluster.update_rate_open(ci);
        }
        sumOpenChannel += openChannel;
        if (t_since_print > t_out && t < 5000) {
            tNormalization = t_since_print / dt;
            // Print data every 100 time-steps (or 0.1 seconds). The refractory period is not included.
            fileCalcium << fixed << setprecision(2) << t << " " << setprecision(4) << ci << " " << cb << " " << j * cer * float(sumOpenChannel) / tNormalization << " " << cer << "\n";
            sumOpenChannel = 0;
            t_since_print = 0;
        }
        if (t > 5000 && countSpikes == 0){
            return 1;
        }
        // If the ci Threshold is crossed do stuff
        if (ci > kCaT) {
            countSpikes += 1;
            // If Caclium Threshold is crossed reset ci and print data
            fileCalcium << t << " " << kCaT << " " << cb << " " << 0 << " " << cer << "\n";
            ci = kCa0 * cer;
            cb = kCa0 * cer * bT/(km/kp + kCa0 * cer);
            if(boolAdap) {
                cer += - eps * cer;
            }
            // Simulate some refractory period with ci = ci Rest. This is essentially a white noise approximation
            file_spikes << setprecision(4) << t_since_spike << " ";
            t_since_spike = 0;
            tRef = 0;
            for (auto &cluster: clusters) {
                cluster.update_rate_open(kCa0);
            }
            while(tRef < 10.){
                // Rates have been updated to the resting ci concentration. The puff model is given some time (spike duration ~10s)
                // to adjust to the resting ci concentration. Essentially the model is assume to be in its stationary state.

                for (auto &cluster: clusters) {
                    //  Calculate transitions for every cluster
                    rnd = uniform_double(generator);
                    if (cluster.state() < 0) {
                        transitionProbability = cluster.rate_ref();
                        if (rnd < transitionProbability * dt) {
                            cluster.refractory();
                        }
                    }
                    else if (cluster.state() == 0) {
                        transitionProbability = cluster.rate_open();
                        if (rnd < transitionProbability * dt) {
                            cluster.open();
                        }
                    }
                    else {
                        transitionProbability = cluster.rate_close();
                        if (rnd < transitionProbability * dt) {
                            cluster.close();
                        }
                    }
                }
                tRef += dt;
            }
        }

        t += dt;
        t_since_print += dt;
        t_since_spike += dt;
    }
    return 0;
}
