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
        rateOpen_ =
                kRateOpen_ * pow(Ca / kCaRest_, 3.) * (1 + pow(kCaRest_, 3.)) / (1 + pow(Ca, 3.)) * pow(kIp3_ / 1., 3) *
                (1. + pow(1., 3)) / (1. + pow(kIp3_, 3));
    }

    void open() {
        // opens a random of channels between drawn from a uniform distribution between 1 and n.
        state_ = intDist_(generator);
    }

    void close() {
        if (state_ != 1) {
            state_ -= 1;
        } else {
            state_ = -(kNumClosedStates_ - 1);
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

using namespace std;
namespace pt = boost::property_tree;

int main(int argc, char *argv[]) {
    // Average interpuff interval depends on Ip3, Ca and the number of channel. We assume that the IPIs measured
    // represent an ensemble average (i.e. channel number n = <n>), at the resting [Ca^+] (ca = 0.2 = Ca_rest/Kca)
    // with intermediate IP3 concentration (ip3 = 1 = Ip3/Kip3).
    // The average interpuff interval however still differs among different cell types.
    struct passwd *pw = getpwuid(getuid());
    const char *homedir = pw->pw_dir;

    string paraFile = "../parameter/" + string(argv[1]) + ".json";
    pt::ptree desc;
    pt::json_parser::read_json(paraFile, desc);

    // Temporary Parameter
    const auto run = desc.get<int>("num parameter.run");
    const string output = desc.get<string>("num parameter.output");
    const bool output_puff = desc.get<bool>("num parameter.output puff");

    // ------------------------ Numerical parameter --------------------------------------------------------------------
    const auto dt = desc.get<double>("num parameter.dt");

    // ------------------------ Fixed Parameters for all Simulations ---------------------------------------------------
    const auto k_ip3 = desc.get<float>("cell.ip3");
    const auto k_c0 = desc.get<float>("cell.calcium rest");
    const auto k_cT = desc.get<float>("cell.calcium threshold");

    // ------------------------ Parameters that determine IPI density --------------------------------------------------
    const auto k_N = desc.get<int>("cluster.num channel");
    const auto k_M = desc.get<int>("cluster.number closed states");
    const auto k_r_opn_s = desc.get<float>("cluster.rate open single");
    const auto k_r_ref = desc.get<float>("cluster.rate ref");
    const auto k_r_cls = desc.get<float>("cluster.rate close");

    //------------------------- Parameters that determine Cell properties ----------------------------------------------
    const auto k_K = desc.get<int>("cell.num cluster");
    const auto tau = desc.get<float>("cell.timeconstant");
    const auto j = desc.get<float>("cell.blip current");

    //------------------------- Parameters that determine Adaptatopm properties ----------------------------------------
    const auto k_tau_er = desc.get<float>("adaptation.timeconstant");
    const auto k_eps = desc.get<float>("adaptation.amplitude");

    // ------------------------ Parameters for output file -------------------------------------------------------------
    std::string path;
    if (output == "local") {
        path = "../out/";
    } else {
        path = "/neurophysics/lukasra/Data/";
    }
    char parameters[200];
    std::sprintf(parameters, "_ip%.2f_taua%.2e_ampa%.2e_tau%.2e_j%.2e_K%d_%d.dat", k_ip3, k_tau_er, k_eps, tau, j, k_K,
                 run);

    // comment this block and use the block above instead for transient spike times.
    std::string out_file_adap;
    out_file_adap = path + "transient_adaptation_markov" + parameters;
    std::ofstream file_adap;
    file_adap.open(out_file_adap);
    if (!file_adap.is_open()) {
        std::cout << "Could not open file at: " << out_file_adap << std::endl;
        std::cout << "This is where I am: " << std::string(homedir) << std::endl;
        return 1;
    }

    // ------------------------ Random number generator ----------------------------------------------------------------
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<double> uniform_double(0, 1);
    //------------------------------------------------------------------------------------------------------------------

    for (int i = 0; i < 100; i++) {
        cout << i << endl;
        std::vector<Cluster> clusters;
        for (int ii = 0; ii < k_K; ii++) {
            Cluster cluster(ii, k_N, k_M, k_r_opn_s, k_r_ref, k_r_cls, k_ip3, k_c0);
            clusters.push_back(cluster);
        }

        double t = 0;
        double t_since_spike = 0;
        double ci = k_c0;
        double cer = 1;
        double j_leak, j_puff;
        double transition_prob;
        double rnd;
        int nr_open_channels;
        int count = 0;

        for (auto &cluster: clusters) {
            cluster.update_rate_open(ci);
        }

        while (t < 1000) {
            // Main Loop
            nr_open_channels = 0;
            for (auto &cluster: clusters) {
                //  Calculate transitions for every cluster
                rnd = uniform_double(generator);
                if (cluster.state() < 0) {
                    transition_prob = cluster.rate_ref();
                    if (rnd < transition_prob * dt) {
                        cluster.refractory();
                    }
                } else if (cluster.state() == 0) {
                    transition_prob = cluster.rate_open();
                    if (rnd < transition_prob * dt) {
                        cluster.open();
                    }
                } else {
                    transition_prob = cluster.rate_close();
                    if (rnd < transition_prob * dt) {
                        cluster.close();
                    }
                }
            }

            for (auto &cluster: clusters) {
                // Count number of open channel
                if (cluster.state() > 0) {
                    nr_open_channels += cluster.state();
                }
            }
            //  Calculate cell wide Calcium dynamics
            j_leak = -(ci - k_c0 * cer) / tau;
            j_puff = j * float(nr_open_channels) * cer;
            ci += (j_leak + j_puff) * dt;
            cer += (1. - cer) / k_tau_er * dt;

            // Update opening rate according to the current ci concentration
            for (auto &cluster: clusters) {
                cluster.update_rate_open(ci);
            }

            if (ci > k_cT) {
                cer += -k_eps * cer;
                ci = k_c0 * cer;

                t_since_spike = 0;
                for (auto &cluster: clusters) {
                    cluster.update_rate_open(k_c0 * cer);
                }
            }

            if (count == 1000){
                file_adap << fixed << setprecision(4) << cer << " ";
                count = 0;
            }
            count += 1;
            t += dt;
            t_since_spike += dt;
        }
        file_adap << endl;


    }
    return 0;
}
