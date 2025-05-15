// MapReduceFramework.cpp
#include "MapReduceFramework.h"
#include "MapReduceClient.h"
#include "Barrier.h"

#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream> // for system error reporting

// forward‐declares from your file:
struct ThreadContext;
struct JobContext;

// Helper: report and exit on system failure
static void sysError(const char *msg)
{
    std::cerr << "system error: " << msg << "\n";
    std::exit(1);
}

// -- Internals (not in the header) ----------------------------------

struct ThreadContext
{
    JobContext *job; // back‐pointer
    int id;          // 0 … multiThreadLevel-1
};

struct JobContext
{
    // user parameters
    const MapReduceClient *client;
    const InputVec *input;
    OutputVec *output;
    int numThreads;

    // threads & contexts
    std::vector<std::thread> threads;
    std::vector<ThreadContext> tcs;

    // per-thread buffers
    std::vector<IntermediateVec> intermediates;

    // shuffle queue
    std::vector<IntermediateVec> shuffleQueue;
    std::mutex shuffleMux;
    std::atomic<size_t> shuffleCount{0};

    // barrier for end‐of‐sort
    Barrier barrier;

    // output‐vector protection
    std::mutex outMux;

    // progress tracking
    std::atomic<size_t> mapIndex{0};
    size_t totalItems;
    std::atomic<size_t> shuffledSoFar{0};

    std::atomic<size_t> totalIntermediates{0}; // incremented in emit2
    std::atomic<size_t> totalOutput{0};        // incremented in emit3

    // make sure we only join once
    std::atomic<bool> threadsJoined{false};

    JobContext(const MapReduceClient &c, const InputVec &in, OutputVec &out, int n)
        : client(&c), input(&in), output(&out), numThreads(n), tcs(n), intermediates(n), barrier(n), totalItems(in.size())
    {
    }
};

// Helper: report and exit on system failure
static void sysError(const char *msg)
{
    std::cerr << "system error: " << msg << "\n";
    std::exit(1);
}

// Worker function
static void worker(ThreadContext *tc)
{
    JobContext *job = tc->job;
    int id = tc->id;

    // ---- MAP PHASE ----
    job->client->map(
        nullptr, nullptr, nullptr); // ensure UNDEFINED_STAGE → MAP_STAGE
    while (true)
    {
        size_t idx = job->mapIndex++;
        if (idx >= job->totalItems)
            break;
        const auto &pair = job->input->at(idx);
        job->client->map(pair.first, pair.second, tc);
    }

    // ---- SORT PHASE ----
    std::sort(
        job->intermediates[id].begin(),
        job->intermediates[id].end(),
        [](auto &a, auto &b)
        { return *(a.first) < *(b.first); });

    // wait for everyone
    job->barrier.barrier();

    // ---- SHUFFLE PHASE (only thread 0) ----
    if (id == 0)
    {
        for (;;)
        {
            // assemble one key‐group from all intermediates
            IntermediateVec group;
            for (int t = 0; t < job->numThreads; ++t)
            {
                auto &vec = job->intermediates[t];
                if (vec.empty())
                    continue;
                auto kv = vec.back();
                vec.pop_back();
                group.push_back(kv);
            }
            if (group.empty())
                break;
            job->shuffleMux.lock();
            job->shuffleQueue.push_back(std::move(group));
            job->shuffleMux.unlock();
            job->shuffledSoFar++;
        }
    }

    // ---- REDUCE PHASE ----
    // spin until everything is shuffled
    while (job->shuffledSoFar.load() < job->shuffleCount.load())
    { /* busy‐wait */
    }

    // now pop groups and reduce
    for (;;)
    {
        job->shuffleMux.lock();
        if (job->shuffleQueue.empty())
        {
            job->shuffleMux.unlock();
            break;
        }
        auto group = std::move(job->shuffleQueue.back());
        job->shuffleQueue.pop_back();
        job->shuffleMux.unlock();

        job->client->reduce(&group, tc);
    }

    // swap output into the shared vector
    // (emit3 will handle locking, see below)
}

// -- Framework API -------------------------------------------------

JobHandle startMapReduceJob(const MapReduceClient &client,
                            const InputVec &inputVec,
                            OutputVec &outputVec,
                            int multiThreadLevel)
{
    // allocate and initialize
    JobContext *job = nullptr;
    try
    {
        job = new JobContext(client, inputVec, outputVec, multiThreadLevel);
    }
    catch (...)
    {
        sysError("allocating JobContext");
    }

    // fill thread contexts
    for (int i = 0; i < multiThreadLevel; ++i)
    {
        job->tcs[i] = {job, i};
    }

    // launch threads
    try
    {
        for (int i = 0; i < multiThreadLevel; ++i)
        {
            job->threads.emplace_back(worker, &job->tcs[i]);
        }
    }
    catch (...)
    {
        sysError("spawning threads");
    }

    return static_cast<JobHandle>(job);
}

// Called by client.map(...) to emit (K2*,V2*)
void emit2(K2 *key, V2 *value, void *context)
{
    if (!context)
        sysError("emit2: null context");

    auto tc = static_cast<ThreadContext *>(context);
    auto job = tc->job;

    // 1) append into *this thread’s* intermediate buffer
    job->intermediates[tc->id].emplace_back(key, value);

    // 2) bump total intermediates for progress
    job->totalIntermediates.fetch_add(1, std::memory_order_relaxed);
}

// Called by client.reduce(...) to emit (K3*,V3*)
void emit3(K3 *key, V3 *value, void *context)
{
    if (!context)
        sysError("emit3: null context");

    auto tc = static_cast<ThreadContext *>(context);
    auto job = tc->job;

    // protect concurrent writes into the single outputVec
    {
        std::lock_guard<std::mutex> lg(job->outMux);
        job->output->emplace_back(key, value);
    }

    // bump total outputs for progress
    job->totalOutput.fetch_add(1, std::memory_order_relaxed);
}

void waitForJob(JobHandle job)
{
    if (!job)
        return;

    auto jc = static_cast<JobContext *>(job);

    // ensure we only join the threads a single time
    bool expected = false;
    if (jc->threadsJoined.compare_exchange_strong(expected, true))
    {
        // first caller will see expected==false, set it to true, and do the joins
        for (std::thread &t : jc->threads)
        {
            if (t.joinable())
            {
                t.join();
            }
        }
    }
    // subsequent calls see threadsJoined==true and do nothing
}

// … your existing includes, forward‐decls, sysError(), emit2, emit3, waitForJob …

// 1) First, make sure your JobContext has these members:
//
//    std::atomic<size_t> totalIntermediates{0};  // bumped in emit2
//    std::atomic<size_t> totalOutput{0};         // bumped in emit3
//
// (you already added threadsJoined earlier)

void getJobState(JobHandle job, JobState *state)
{
    if (!job || !state)
        return;
    auto jc = static_cast<JobContext *>(job);

    // 0% until any mapping begins
    size_t mapped = jc->mapIndex.load(std::memory_order_relaxed);
    if (mapped == 0)
    {
        state->stage = UNDEFINED_STAGE;
        state->percentage = 0.0f;
        return;
    }

    // MAP stage: still mapping input items
    size_t totalItems = jc->totalItems;
    if (mapped < totalItems)
    {
        state->stage = MAP_STAGE;
        state->percentage = (mapped * 100.0f) / totalItems;
        return;
    }

    // SHUFFLE stage: mapping done, but still shuffling intermediates
    size_t shuffled = jc->shuffledSoFar.load(std::memory_order_relaxed);
    size_t totalInt = jc->totalIntermediates.load(std::memory_order_relaxed);
    if (shuffled < totalInt)
    {
        state->stage = SHUFFLE_STAGE;
        // guard divide‐by‐zero if no intermediates at all
        state->percentage = totalInt
                                ? (shuffled * 100.0f) / totalInt
                                : 100.0f;
        return;
    }

    // REDUCE stage: shuffling done, but still reducing
    size_t reduced = jc->totalOutput.load(std::memory_order_relaxed);
    if (reduced < totalInt)
    {
        state->stage = REDUCE_STAGE;
        state->percentage = totalInt
                                ? (reduced * 100.0f) / totalInt
                                : 100.0f;
    }
    else
    {
        // all done
        state->stage = REDUCE_STAGE;
        state->percentage = 100.0f;
    }
}

void closeJobHandle(JobHandle job)
{
    if (!job)
        return;
    // wait for threads if needed
    waitForJob(job);

    // free all resources
    auto jc = static_cast<JobContext *>(job);
    delete jc;
}
