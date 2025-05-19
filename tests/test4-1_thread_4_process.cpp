/**
 * @brief run 1 thread on 4 different data - check for thread safety
 */

#include <iostream>
#include "MapReduceClient.h"
#include "MapReduceFramework.h"
#include <chrono>
#include <thread>
#include <algorithm>
#include <set>

unsigned int unique_keys = 100;

class elements : public K1, public K2, public K3, public V1, public V2, public V3 {
public:
    elements(int i) { num = i; }
    bool operator<(const K1 &other) const { return num < dynamic_cast<const elements&>(other).num; }
    bool operator<(const K2 &other) const { return num < dynamic_cast<const elements&>(other).num; }
    bool operator<(const K3 &other) const { return num < dynamic_cast<const elements&>(other).num; }
    bool operator<(const V1 &other) const { return num < dynamic_cast<const elements&>(other).num; }
    bool operator<(const V2 &other) const { return num < dynamic_cast<const elements&>(other).num; }
    bool operator<(const V3 &other) const { return num < dynamic_cast<const elements&>(other).num; }
    int num;
};

class tester : public MapReduceClient {
public:
    void map(const K1* key, const V1* val, void* context) const override {
        int input = (static_cast<const elements*>(key)->num) % unique_keys;
        emit2(new elements(input), new elements(1), context);
    }
    void reduce(const IntermediateVec* pairs, void* context) const override {
        emit3(
                new elements(static_cast<const elements*>(pairs->at(0).first)->num),
                new elements(static_cast<int>(pairs->size())),
                context
        );
        for (const auto &pair : *pairs) {
            delete pair.first;
            delete pair.second;
        }
    }
};

struct ThreadSafe { unsigned int threadsNumber; InputVec inputVec; OutputVec outputVec; };

int main() {
    unsigned int numOfProcess = 4;
    unsigned int numOfThreads = 1;
    std::vector<ThreadSafe> contexts(numOfProcess);
    std::vector<InputVec> in_items_vec(numOfProcess);
    std::vector<OutputVec> out_items_vec(numOfProcess);
    std::vector<JobHandle> jobs(numOfProcess);
    tester client;

    // prepare inputs
    for (unsigned int l = 0; l < numOfProcess; ++l) {
        std::srand(l);
        for (int j = 0; j < 100000; ++j) {
            int r = std::rand();
            in_items_vec[l].push_back({ new elements(r), nullptr });
        }
        contexts[l] = { numOfThreads, in_items_vec[l], out_items_vec[l] };
    }

    // launch jobs
    for (unsigned i = 0; i < numOfProcess; ++i) {
        jobs[i] = startMapReduceJob(client, contexts[i].inputVec, contexts[i].outputVec, contexts[i].threadsNumber);
    }
    for (auto &job : jobs) closeJobHandle(job);

    // sort & print & cleanup
    for (unsigned m = 0; m < numOfProcess; ++m) {
        auto &out = contexts[m].outputVec;
        std::sort(out.begin(), out.end(), [](auto &a, auto &b){ return *a.first < *b.first; });
        for (auto &p : out) {
            std::cout << "thread " << m+1 << " out:\t" << static_cast<elements*>(p.second)->num << '\n';
        }
        for (auto &p : out) { delete p.first; delete p.second; }
        for (auto &p : contexts[m].inputVec) { delete p.first; delete p.second; }
    }
    return 0;
}
