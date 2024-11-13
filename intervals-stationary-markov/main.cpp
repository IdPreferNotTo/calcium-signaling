#include <iostream>
#include <vector>
#include <random>
#include <pwd.h>
#include <cmath>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

class cluster {
private:
    const int k_num_cha_;
    const int k_num_closed_states_;
    const float k_rate_open_single_;
    const float k_rate_open_ = float(k_num_cha_) * k_rate_open_single_;
    const float k_rate_ref_;
    const float k_rate_close_;
    const float k_ca_rest_;
    float k_ip3_;
    int k_idx_;

    static std::random_device rd;
    static unsigned seed;
    static std::mt19937 generator;
    std::uniform_int_distribution<int> int_dist_;

    double rate_open_ = k_rate_open_;
    double rate_ref_ = k_rate_ref_;
    int state_ = 0;
public:
    cluster(const int idx, const int numCha, const int numClosedStates, const float rateOpenSingle,
            const float rateRef, const float rateClose, const float Ip3, const float caRest) :
            k_idx_(idx), k_num_cha_(numCha), k_num_closed_states_(numClosedStates), k_rate_open_single_(rateOpenSingle),
            k_rate_ref_(rateRef), k_rate_close_(rateClose), k_ip3_(Ip3), k_ca_rest_(caRest),
            int_dist_(std::uniform_int_distribution<>(1, numCha)) {
    }

    void set_ip3(float ip3) {
        k_ip3_ = ip3;
    }

    int state() const {
        return state_;
    }

    int idx() const {
        return k_idx_;
    }

    void update_rate_open(double Ca) {
        rate_open_ = k_rate_open_ * pow(Ca / k_ca_rest_, 3.) * (1 + pow(k_ca_rest_, 3.)) / (1 + pow(Ca, 3.)) * pow(k_ip3_ / 1., 3) * (1. + pow(1., 3)) / (1. + pow(k_ip3_, 3));
    }

    void open() {
        // opens a random of channels between drawn from a uniform distribution between 1 and n.
        state_ = int_dist_(generator);
    }

    void close() {
        if (state_ != 1) {
            state_ -= 1;
        } else {
            state_ = -(k_num_closed_states_ - 1);
        }
    }

    void refractory() {
        state_ += 1;
    }

    double rate_open() const {
        // rate_open is the cumulative transition probability to leave the closed state (x_i = 0) into ANY open state
        return rate_open_;
    }

    double rate_close() const {
        // rate_close is the transition probability from n open channel to n-1 open channel (x_i = n -> x_i=n-1)
        return k_rate_close_;
    }

    double rate_ref() const {
        // rate_ref is the transition probability from the m'th to m'th + 1 refractory state
        return rate_ref_;
    }


};

std::random_device cluster::rd;
unsigned cluster::seed = rd();
std::mt19937 cluster::generator(cluster::seed);

class observer {
private:
    const double &t_;
    std::ofstream &file_;
public:
    observer(const double &t, std::ofstream &file) : t_(t), file_(file) {

    }

    void write(int state, int idx, int n) {
        file_ << std::setprecision(10) << t_ << " " << state << " " << idx << " " << n << "\n";
    }
};

using namespace std;
namespace pt = boost::property_tree;

int main(int argc, char *argv[]) {
    // Average interpuff interval depends on Ip3, ci and the number of channel. We assume that the IPIs measured
    // represent an ensemble average (i.e. channel number n = <n>), at the resting [ci^+] (ca = 0.2 = Ca_rest/Kca)
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
    const auto k_max_spikes = desc.get<int>("num parameter.max spikes");
    const auto k_max_time = desc.get<double>("num parameter.max time");

    // ------------------------ Fixed Parameters for all Simulations ---------------------------------------------------
    const auto k_ip3 = desc.get<float>("cell.ip3");
    const auto k_c0 = desc.get<float>("cell.calcium rest");
    const auto k_cT = desc.get<float>("cell.calcium threshold");

    // ------------------------ Parameters that determine if ci is fixed -----------------------------------------------
    const auto k_bool_ca_fix = desc.get<bool>("calcium fix.on");
    const auto k_cF = desc.get<double>("calcium fix.value");

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
    const auto k_bool_cer = desc.get<bool>("adaptation.on");
    const auto k_tau_er = desc.get<float>("adaptation.timeconstant");
    const auto k_eps = desc.get<float>("adaptation.amplitude");

    // ------------------------ Parameters for output file -------------------------------------------------------------
    std::string path;
    if(output =="local"){
        path = "../out/";
    }
    else{
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

    //open file for puff dynamics and spike dynamics
    std::string out_file_puff;
    out_file_puff = path + "puff_markov" + parameters;
    std::ofstream file_puff;
    file_puff.open(out_file_puff);
    if (!file_puff.is_open()) {
        std::cout << "Could not open file at: " << out_file_puff << std::endl;
        std::cout << "This is where I am: " << std::string(homedir) << std::endl;
        return 1;
    }
    file_puff << "# Parameter" << "\n";
    file_puff << "# dt: " << dt << "\n";
    file_puff << "# r_open: " << k_r_opn_s << ", r_close: " << k_r_cls << ", r_refractory:" << k_r_ref << "\n";
    file_puff << "# N_Clu: " << k_K << ", N_Cha: " << k_N << ", N_Closed: " << k_M << "\n";
    file_puff << "# ci rest: " << k_c0 << ", ci Threshold" << 1 << "\n";
    file_puff << "# t | state | idx " << std::endl;


    std::string out_file_calcium;
    out_file_calcium = path + "ca_markov" + parameters;
    std::ofstream file_ca;
    file_ca.open(out_file_calcium);
    if (!file_ca.is_open()) {
        std::cout << "Could not open file at: " << out_file_calcium << std::endl;
        std::cout << "This is where I am: " << std::string(homedir) << std::endl;
        return 1;
    }
    file_ca << "# Parameter" << "\n";
    file_ca << "# dt: " << dt << "\n";
    file_ca << "# r_open: " << k_r_opn_s << ", r_close: " << k_r_cls << ", r_refractory:" << k_r_ref << "\n";
    file_ca << "# N_Clu: " << k_K << ", N_Cha: " << k_N << ", N_Closed: " << k_M << "\n";
    file_ca << "# ci rest: " << k_c0 << ", ci Threshold" << 1 << "\n";
    file_ca << "# t | ci | j_puff | adap |" << std::endl;

    std::string out_file_spikes;
    out_file_spikes = path + "spike_times_markov" + parameters;
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
    std::vector<cluster> clusters;
    for (int i = 0; i < k_K; i++) {
        cluster cluster(i, k_N, k_M, k_r_opn_s, k_r_ref, k_r_cls, 0, k_c0);
        clusters.push_back(cluster);
    }

    int count_spikes = 0;

    double t = 0;
    double t_since_print = 0;
    double t_since_spike = 0;
    double t_norm;

    double ci;
    if(k_bool_ca_fix){
        ci = k_cF;
    }
    else{
        ci = k_c0;
    }

    double cer = 1;
    double j_leak;
    double j_puff;

    double transition_prob;
    double rnd;

    int nr_opn_channels;
    int sum_opn_channels = 0;

    observer observer(t, file_puff);

    for (auto &cluster: clusters) {
            cluster.update_rate_open(ci);
    }
    while (t < k_max_time && count_spikes < k_max_spikes) {
        if(t >= 0){
            for (auto &cluster: clusters) {
                cluster.set_ip3(k_ip3);
            }
        }
        // Main Loop
        nr_opn_channels = 0;
        for (auto &cluster: clusters) {
            //  Calculate transitions for every cluster
            rnd = uniform_double(generator);
            if (cluster.state() < 0) {
                transition_prob = cluster.rate_ref();
                if (rnd < transition_prob * dt) {
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
                transition_prob = cluster.rate_open();
                if (rnd < transition_prob * dt) {
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
                transition_prob = cluster.rate_close();
                if (rnd < transition_prob * dt) {
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
                nr_opn_channels += cluster.state();
            }
        }

        //  Calculate cell wide Calcium dynamics
        j_leak = -(ci - k_c0 * cer) / tau;
        j_puff = j * float(nr_opn_channels) * cer;
        if(!k_bool_ca_fix){
            ci += (j_leak + j_puff) * dt;
        }
        if(k_bool_cer){
            cer += ((1. - cer) / k_tau_er - 0.01*(j_leak + j_puff) )* dt;
        }
        // Update opening rate according to the current ci concentration
        for (auto &cluster: clusters) {
            cluster.update_rate_open(ci);
        }
        sum_opn_channels += nr_opn_channels;
        if (t_since_print > t_out && t < 10000) {
            t_norm = t_since_print / dt;
            // Print data every 100 time-steps (or 0.1 seconds). The refractory period is not included.
            file_ca << fixed << setprecision(2) << t << " " << setprecision(4) << ci << " " << float(sum_opn_channels) / t_norm << " " << cer << "\n";
            sum_opn_channels = 0;
            t_since_print = 0;
        }
        if (t > 2000 && count_spikes == 0){
            return 1;
        }
        // If the ci Threshold is crossed do stuff
        if(!k_bool_ca_fix) {
            if (ci > k_cT) {
                count_spikes += 1;
                if (k_bool_cer) {
                    cer += -k_eps * cer;
                }
                ci = k_c0 * cer;
                if (t < 10000) {
                    file_ca << t << " " << ci << " " << 69.0 << " " << cer << "\n";
                }

                // Simulate some refractory period with ci = ci Rest. This is essentially a white noise approximation
                file_spikes << setprecision(8) << t_since_spike << " ";
                t_since_spike = 0;
                for (auto &cluster: clusters) {
                    cluster.update_rate_open(k_c0 * cer);
                }
            }
        }
        t += dt;
        t_since_print += dt;
        t_since_spike += dt;
    }
    return 0;
}
