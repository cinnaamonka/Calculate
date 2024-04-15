#include <iostream>
#include <random>
#include <vector>
#include <mutex>
#include <thread>
#include <chrono>
#include <numeric>
#include <functional>
#include <iomanip>
#include <future>
#include <algorithm>
#include <execution>

int in = 0;
int total = 0;

constexpr uint64_t ITERATIONS = static_cast<uint64_t>(1e8);

struct Vector2f
{
    float x, y;
};

Vector2f GetRandomPoint()
{
    static std::random_device rd;
    static std::default_random_engine eng(rd());
    static std::uniform_real_distribution<float> distr(-1, 1);
    return { distr(eng), distr(eng) };
}

double CalculatePi(uint64_t const start, uint64_t const end)
{
    double sum = 0;

    for (uint64_t i = start; i < end; i++)
    {
        int const sign = i % 2 ? -1 : 1;
        double const term = static_cast<double>(2 * i + 1);
        sum += sign / term;
    }
    return 4 * sum;
}

double CalculatePiMultithreaded(uint64_t const iterations,
    size_t const numThreads = std::thread::hardware_concurrency())
{
    std::vector<std::jthread*> threads(numThreads);
    uint64_t const chunk = iterations / numThreads;
    std::vector<double> sums(numThreads);

    for (size_t i = 0; i < numThreads; i++)
    {
        threads[i] = new std::jthread([=, &sums]()
            {
                auto const result = CalculatePi(i * chunk, (i + 1) * chunk);
                sums[i] = result;
            });
    }

    for (std::jthread* t : threads)
    {
        delete t;
    }

    std::sort(sums.begin(), sums.end(), [](double a, double b) { return b < a; });
    return std::accumulate(sums.begin(), sums.end(), 0.0);
}

double CalculatePiMultithreadedWithFutureAndPromises(uint64_t const iterations,
    size_t const numThreads = std::thread::hardware_concurrency())
{
    std::vector<std::jthread> threads(numThreads);
    uint64_t const chunk = iterations / numThreads;
    std::vector<std::promise<double>> promises(numThreads);
    std::vector<std::future<double>> futures;

    for (size_t i = 0; i < numThreads; ++i)
    {
        futures.push_back(promises[i].get_future());
        threads[i] = std::jthread([=, &promises](size_t threadIdx)
            {
                auto const result = CalculatePi(threadIdx * chunk, (threadIdx + 1) * chunk);
                promises[threadIdx].set_value(result);
            }, i);
    }

    std::vector<double> sums;
    for (size_t i = 0; i < numThreads; ++i)
    {
        sums.push_back(futures[i].get());
    }

    std::sort(sums.begin(), sums.end(), std::greater<double>());
    return std::accumulate(sums.begin(), sums.end(), 0.0);
}

double CalculatePiMultithreadedWithAsync(uint64_t const iterations,
    size_t const numThreads = std::thread::hardware_concurrency())
{
    std::vector<std::future<double>> futures;
    uint64_t const chunk = iterations / numThreads;

    for (size_t i = 0; i < numThreads; ++i)
    {
        futures.push_back(std::async(std::launch::async, [=]()
            {
                return CalculatePi(i * chunk, (i + 1) * chunk);
            }));
    }

    std::vector<double> sums;
    for (auto& future : futures)
    {
        sums.push_back(future.get());
    }

    std::sort(sums.begin(), sums.end(), std::greater<double>());
    return std::accumulate(sums.begin(), sums.end(), 0.0);
}

template <typename ExecutionPolicy>
double CalculatePiMultithreadedWithPolicy(uint64_t const iterations,
    size_t const numThreads = std::thread::hardware_concurrency(),
    ExecutionPolicy policy = std::execution::par)
{
    std::vector<std::future<double>> futures(numThreads);
    uint64_t const chunk = iterations / numThreads;

    std::vector<uint64_t> threadIndices(numThreads);
    std::iota(threadIndices.begin(), threadIndices.end(), 0);

    std::transform(threadIndices.begin(), threadIndices.end(), futures.begin(), [=](size_t threadIdx)
        {
            return std::async(std::launch::async, [=]() {
                return CalculatePi(threadIdx * chunk, (threadIdx + 1) * chunk);
                });
        });

    double result = std::transform_reduce(policy, futures.begin(), futures.end(), 0.0,
        std::plus<>(), [](std::future<double>& f) { return f.get(); });

    return result;
}

double CalculatePiMultithreadedWithSTL(uint64_t const iterations,
    size_t const numThreads = std::thread::hardware_concurrency())
{
    std::vector<std::future<double>> futures(numThreads);
    uint64_t const chunk = iterations / numThreads;

    std::vector<uint64_t> threadIndices(numThreads);
    std::iota(threadIndices.begin(), threadIndices.end(), 0);

    std::transform(threadIndices.begin(), threadIndices.end(), futures.begin(), [=](size_t threadIdx) {
        return std::async(std::launch::async, [=]() {
            return CalculatePi(threadIdx * chunk, (threadIdx + 1) * chunk);
            });
        });

    double result = std::transform_reduce(std::execution::par, futures.begin(), futures.end(), 0.0,
        std::plus<>(), [](std::future<double>& f) { return f.get(); });

    return result;
}

void CountPoints(int number, int& local_in, int& local_total)
{
    local_in = 0;
    for (int i = 0; i < number; ++i)
    {
        auto point = GetRandomPoint();
        if (point.x * point.x + point.y * point.y < 1.f) {
            ++local_in;
        }
    }
    local_total = number;
}

double MeasureTime(std::function<void(void)> const fn)
{
    auto t1 = std::chrono::high_resolution_clock::now();
    fn();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto delta = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    return delta.count();
}

void SingleThreadedExecution()
{
    auto start = std::chrono::steady_clock::now();

    CountPoints(ITERATIONS, in, total);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Single-threaded execution time: " << elapsed_seconds.count() << "s\n";
    std::cout << "Calculated value of Pi: " << 1.f * in / total * 4 << "\n";
}

int main()
{
    SingleThreadedExecution();

    double pi{};
    auto execTime = MeasureTime([&pi]() {
        pi = CalculatePiMultithreaded(ITERATIONS);
        });


    std::cout << "Multi-threaded execution time: " << std::setprecision(2) << execTime << " seconds" << std::endl;
    std::cout << "Calculated PI is: " << std::fixed << std::setprecision(5) << pi << std::endl;

    double pi2{};
    auto execTime2 = MeasureTime([&pi2]() {
        pi2 = CalculatePiMultithreadedWithFutureAndPromises(ITERATIONS);
        });


    std::cout << "Multi-threaded execution with future and promises time: " << std::setprecision(2) << execTime2 << " seconds" << std::endl;
    std::cout << "Calculated PI is: " << std::fixed << std::setprecision(5) << pi2 << std::endl;

    double pi3{};
    auto execTime3 = MeasureTime([&pi3]() {
        pi3 = CalculatePiMultithreadedWithAsync(ITERATIONS);
        });


    std::cout << "Multi-threaded execution with async time: " << std::setprecision(2) << execTime3 << " seconds" << std::endl;
    std::cout << "Calculated PI is: " << std::fixed << std::setprecision(5) << pi3 << std::endl;

    double pi4{};
    auto execTime4 = MeasureTime([&pi4]() {
        pi4 = CalculatePiMultithreadedWithSTL(ITERATIONS);
        });

    std::cout << "Multi-threaded execution with STL: " << std::setprecision(2) << execTime4 << " seconds" << std::endl;
    std::cout << "Calculated PI is: " << std::fixed << std::setprecision(5) << pi4 << std::endl;

    double pi5{};
    auto execTime5 = MeasureTime([&pi5]() {
        pi5 = CalculatePiMultithreadedWithPolicy(ITERATIONS, std::thread::hardware_concurrency(), std::execution::seq);
        });

    std::cout << "Parallel execution policy:\n";
    std::cout << "Result: " << pi5 << std::endl;
    std::cout << "Time taken: " << execTime5 << " seconds\n\n";

    double pi6{};
    auto execTime6 = MeasureTime([&pi6]() {
        pi6 = CalculatePiMultithreadedWithPolicy(ITERATIONS, std::thread::hardware_concurrency(), std::execution::seq);
        });

    std::cout << "Sequential execution policy:\n";
    std::cout << "Result: " << pi6 << std::endl;
    std::cout << "Time taken: " << execTime6 << " seconds\n\n";

    double pi7{};
    auto execTime7 = MeasureTime([&pi7]() {
        pi7 = CalculatePiMultithreadedWithPolicy(ITERATIONS, std::thread::hardware_concurrency(), std::execution::par_unseq);
        });

    std::cout << "Parallel unsequenced execution policy:\n";
    std::cout << "Result: " << pi7 << std::endl;
    std::cout << "Time taken: " << execTime7 << " seconds\n\n";

    double pi8{};
    auto execTime8 = MeasureTime([&pi8]() {
        pi8 = CalculatePiMultithreadedWithPolicy(ITERATIONS, std::thread::hardware_concurrency(), std::execution::unseq);
        });

    std::cout << "Unsequenced execution policy:\n";
    std::cout << "Result: " << pi8 << std::endl;
    std::cout << "Time taken: " << execTime8 << " seconds\n\n";

    return 0;
}
