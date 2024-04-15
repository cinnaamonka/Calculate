// Calculate.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <random>
#include <vector>
#include <mutex>
#include <thread>
#include <chrono>

int in = 0;
int total = 0;

struct vector2f
{
    float x, y;
};

vector2f getRandomPoint()
{
    static std::random_device rd;
    static std::default_random_engine eng(rd());
    static std::uniform_real_distribution<float> distr(-1, 1);
    return { distr(eng), distr(eng) };
}

void countPoints(int number, int& local_in, int& local_total)
{
    local_in = 0;
    for (int i = 0; i < number; ++i)
    {
        auto point = getRandomPoint();
        if (point.x * point.x + point.y * point.y < 1.f) {
            ++local_in;
        }
    }
    local_total = number;
}

void singleThreadedExecution()
{
    auto start = std::chrono::steady_clock::now();

    countPoints(100000000, in, total);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Single-threaded execution time: " << elapsed_seconds.count() << "s\n";
}

void multiThreadedExecution()
{
    auto start = std::chrono::steady_clock::now();

    const int num_threads = std::thread::hardware_concurrency();
    const int per_thread = 100000000 / num_threads;

    std::vector<std::thread> threads;
    std::vector<int> local_ins(num_threads);
    std::vector<int> local_totals(num_threads);

    for (int i = 0; i < num_threads; ++i)
    {
        threads.emplace_back(countPoints, per_thread, std::ref(local_ins[i]), std::ref(local_totals[i]));
    }

    for (auto& thread : threads)
    {
        thread.join();
    }

    for (int i = 0; i < num_threads; ++i)
    {
        in += local_ins[i];
        total += local_totals[i];
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Multi-threaded execution time: " << elapsed_seconds.count() << "s\n";
}

int main()
{
    singleThreadedExecution();
    multiThreadedExecution();

    std::cout << "Estimated value of Pi: " << 1.f * in / total * 4 << "\n";
    return 0;
}


