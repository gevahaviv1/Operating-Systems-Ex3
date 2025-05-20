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
#include <iostream>

// forward‐declares from your file:
struct ThreadContext;
struct JobContext;

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

    // barrier for end‐of‐sort
    Barrier barrier;

    // output‐vector protection
    std::mutex outMux;

    // progress tracking
    std::atomic<size_t> mapIndex{0};
    std::atomic<size_t> shuffledSoFar{0};
    std::atomic<size_t> totalGroups{0};
    std::atomic<size_t> reducedGroups{0};
    std::atomic<bool> shuffleComplete{false};
    size_t totalItems;

    std::atomic<size_t> totalIntermediates{0}; // incremented in emit2
    std::atomic<size_t> totalOutput{0};        // incremented in emit3

    // make sure we only join once
    std::atomic<bool> threadsJoined{false};

    JobContext(const MapReduceClient &c, const InputVec &in, OutputVec &out, int n)
        : client(&c), input(&in), output(&out), numThreads(n), tcs(n), intermediates(n), barrier(n), totalItems(in.size())
    {
    }
};

static void sysError(const char *msg)
{
    std::cerr << "system error: " << msg << "\n";
    std::exit(1);
}

static void worker(ThreadContext *tc)
{
    JobContext *job = tc->job;
    int id = tc->id;

    // — MAP —
    while (true)
    {
        size_t idx = job->mapIndex++;
        if (idx >= job->totalItems)
            break;
        const auto &kv = job->input->at(idx);
        job->client->map(kv.first, kv.second, tc);
    }

    // — SORT —
    auto &local = job->intermediates[id];
    std::sort(local.begin(), local.end(),
              [](auto &a, auto &b)
              { return *(a.first) < *(b.first); });

    // — BARRIER —
    job->barrier.barrier();

    // — SHUFFLE (thread 0 only) —
    if (id == 0)
    {
        for (;;)
        {
            // find max key
            K2 *maxKey = nullptr;
            for (int t = 0; t < job->numThreads; ++t)
            {
                auto &v = job->intermediates[t];
                if (!v.empty())
                {
                    K2 *k = v.back().first;
                    if (!maxKey || *maxKey < *k)
                        maxKey = k;
                }
            }
            if (!maxKey)
                break;

            // gather all equal‐key entries
            IntermediateVec group;
            for (int t = 0; t < job->numThreads; ++t)
            {
                auto &v = job->intermediates[t];
                while (!v.empty())
                {
                    K2 *k = v.back().first;
                    if (!(*k < *maxKey) && !(*maxKey < *k))
                    {
                        group.push_back(v.back());
                        v.pop_back();
                    }
                    else
                        break;
                }
            }

            // push to shared queue
            {
                std::lock_guard<std::mutex> lg(job->shuffleMux);
                job->shuffleQueue.emplace_back(std::move(group));
            }
            job->shuffledSoFar.fetch_add(group.size(), std::memory_order_relaxed);
            job->totalGroups.fetch_add(1, std::memory_order_relaxed);
        }
        // signal reducers
        job->shuffleComplete.store(true, std::memory_order_release);
    }

    // — REDUCE (all threads) —
    // wait for shuffle to finish
    while (!job->shuffleComplete.load(std::memory_order_acquire))
    { /* spin */
    }

    // consume and reduce groups
    for (;;)
    {
        std::unique_lock<std::mutex> lk(job->shuffleMux);
        if (job->shuffleQueue.empty())
            break;
        auto group = std::move(job->shuffleQueue.back());
        job->shuffleQueue.pop_back();
        lk.unlock();

        job->client->reduce(&group, tc);
    }
}

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
    job->reducedGroups.fetch_add(1, std::memory_order_relaxed);
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
        std::sort(jc->output->begin(), jc->output->end(),
                  [](auto &a, auto &b)
                  { return *(a.first) < *(b.first); });
    }
    // subsequent calls see threadsJoined==true and do nothing
}

void getJobState(JobHandle job, JobState *state)
{
    if (!job || !state)
        return;
    auto jc = static_cast<JobContext *>(job);

    size_t mapped = jc->mapIndex.load(std::memory_order_relaxed);
    size_t totalItems = jc->totalItems;

    size_t totalPairs = jc->totalIntermediates.load(std::memory_order_relaxed);
    size_t shuffled = jc->shuffledSoFar.load(std::memory_order_relaxed);

    size_t totalGroups = jc->totalGroups.load(std::memory_order_relaxed);
    size_t reduced = jc->reducedGroups.load(std::memory_order_relaxed);

    // --- Stage 1: MAP ---
    if (mapped == 0)
    {
        state->stage = MAP_STAGE;
        state->percentage = 0.0f;
        return;
    }
    if (mapped < totalItems)
    {
        state->stage = MAP_STAGE;
        state->percentage = (mapped * 100.0f) / totalItems;
        return;
    }
    // mapped == totalItems but shuffle not yet begun → give a final 100%
    if (totalGroups == 0)
    {
        state->stage = MAP_STAGE;
        state->percentage = 100.0f;
        return;
    }

    // --- Stage 2: SHUFFLE ---
    // Stay in shuffle until *at least one* reduce() has run
    if (reduced == 0)
    {
        state->stage = SHUFFLE_STAGE;
        state->percentage = totalPairs
                                ? (shuffled * 100.0f) / totalPairs
                                : 100.0f;
        return;
    }

    // --- Stage 3: REDUCE ---
    if (reduced < totalGroups)
    {
        state->stage = REDUCE_STAGE;
        state->percentage = (reduced * 100.0f) / totalGroups;
        return;
    }

    // all done
    state->stage = REDUCE_STAGE;
    state->percentage = 100.0f;
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
