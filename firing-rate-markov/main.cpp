#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <boost/property_tree/ptree.hpp>

class cluster {
private:
    const int kNumCha_;
    const int kNumClosedStates_;
    const float kRateOpenSingle_;
    const float kRateOpen_ = float(kNumCha_) * kRateOpenSingle_;
    const float kRateRef_;
    const float kRateClose_;
    const float kCaRest_;
    float kIp3_;
    int kIdx_;

    static std::random_device rd;
    static unsigned seed;
    static std::mt19937 generator;
    std::uniform_int_distribution<int> intDist_;

    double rateOpen_ = kRateOpen_;
    double rateRef_ = kRateRef_;
    int state_ = 0;
public:
    cluster(const int idx, const int numCha, const int numClosedStates, const float rateOpenSingle,
            const float rateRef, const float rateClose, const float Ip3, const float caRest) :
            kIdx_(idx), kNumCha_(numCha), kNumClosedStates_(numClosedStates), kRateOpenSingle_(rateOpenSingle),
            kRateRef_(rateRef), kRateClose_(rateClose), kIp3_(Ip3), kCaRest_(caRest),
            intDist_(std::uniform_int_distribution<>(1, numCha)) {
    }

    void set_ip3(float ip3) {
        kIp3_ = ip3;
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

std::random_device cluster::rd;
unsigned cluster::seed = rd();
std::mt19937 cluster::generator(cluster::seed);


using namespace std;
namespace pt = boost::property_tree;

int main(int argc, char *argv[]) {
    // ------------------------ Numerical parameter --------------------------------------------------------------------
    const auto dt = 0.001;
    const auto kMaxSpikes = 5000;
    const auto kMaxTime = 1000000;

    float tau = atof(argv[1]);
    float j = atof(argv[2]);
    float cer = atof(argv[3]);

    //------------------------- Parameters that determine Cell properties ----------------------------------------------
    const auto kNumClu = 10;
    const int kNumCha = 5;
    const int kNumCls = 3;
    const float krOpnSingle = 0.1;
    const float krRef = 20.;
    const float krCls = 50.;

    const float kIp3 = 1.0;
    const float kCa0 = 0.2;
    const float kCaT = 0.5;

    // ------------------------ Random number generator ----------------------------------------------------------------
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<double> uniform_double(0, 1);
    //------------------------------------------------------------------------------------------------------------------
    std::vector<cluster> clusters;
    for (int i = 0; i < kNumClu; i++) {
        //int numCha = poisson_int(generator);
        //numCha += 1;
        int numCha = kNumCha;
        cluster cluster(i, numCha, kNumCls, krOpnSingle, krRef, krCls, kIp3, kCa0);
        clusters.push_back(cluster);
    }

    int spikeCount = 0;

    double t = 0;
    double tSinceSpike = 0;
    double ca = kCa0;
    double jleak;
    double jpuff;
    double transitionProbability;
    double rnd;

    int openChannel;
    int sumOpenChannel = 0;

    std::vector<double> interspikeIntervals;

    for (auto &cluster: clusters) {
        cluster.update_rate_open(ca);
    }

    while (t < kMaxTime && spikeCount < kMaxSpikes) {
        // Main Loop
        openChannel = 0;

        for (auto &cluster: clusters) {
            //  Calculate transitions for every cluster
            rnd = uniform_double(generator);
            if (cluster.state() < 0) {
                transitionProbability = cluster.rate_ref();
                if (rnd < transitionProbability * dt) {
                    cluster.refractory();
                }
            } else if (cluster.state() == 0) {
                transitionProbability = cluster.rate_open();
                if (rnd < transitionProbability * dt) {
                    cluster.open();
                    int n = 0;
                    for (auto &ii: clusters) {
                        if (ii.state() >= 0) {
                            n += ii.state();
                        }
                    }
                }
            } else {
                transitionProbability = cluster.rate_close();
                if (rnd < transitionProbability * dt) {
                    cluster.close();
                    int n = 0;
                    for (auto &ii: clusters) {
                        if (ii.state() >= 0) {
                            n += ii.state();
                        }
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
        jleak = -(ca - kCa0) / tau;
        jpuff = j * float(openChannel) * cer;
        ca += (jleak + jpuff) * dt;

        // Update opening rate according to the current ca concentration
        for (auto &cluster: clusters) {
            cluster.update_rate_open(ca);
        }
        sumOpenChannel += openChannel;
        if (t > 5000 && spikeCount == 0) {
            cout << 0 << endl;
            return 0.;
        }
        // If the ca Threshold is crossed do stuff
        if (ca >= kCaT) {
            ca = kCa0;
            interspikeIntervals.push_back(tSinceSpike);
            tSinceSpike = 0;
            spikeCount += 1;
        }

        t += dt;
        tSinceSpike += dt;
    }
    double sum = 0;
    for(auto &isi: interspikeIntervals){
        sum += isi;
    }

    double meanIsi;
    meanIsi = sum / interspikeIntervals.size();
    cout << 1. / meanIsi << endl;
    return 0;
}
