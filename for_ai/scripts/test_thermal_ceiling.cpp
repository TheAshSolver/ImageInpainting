#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>

std::atomic<bool> g_run{true};

void worker() {
    volatile double x = 1.0001;
    while (g_run.load(std::memory_order_relaxed)) {
        for (int i = 0; i < 50000; ++i) {
            x = x * 1.000001 + 0.000001;
        }
    }
}

int readTemp(const std::string& path) {
    std::ifstream f(path);
    int t = 0;
    if (f >> t) return t;
    return 0;
}

int main() {
    int numThreads = 7;
    std::cout << "[PROBE] Launching " << numThreads << " Oryon thermal threads..." << std::endl;
    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(worker);
    }

    auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < 30; ++s) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        int t_nsp = 0, t_gpu = 0, t_cpu = 0, t_sys = 0;
        
        // Find zone temps
        for (int i = 0; i < 80; ++i) {
            std::string tp = "/sys/class/thermal/thermal_zone" + std::to_string(i) + "/type";
            std::ifstream tf(tp);
            if (!tf) continue;
            std::string type;
            tf >> type;
            std::string tempPath = "/sys/class/thermal/thermal_zone" + std::to_string(i) + "/temp";
            if (type.find("nsphvx") != std::string::npos) {
                int t = readTemp(tempPath);
                if (t > t_nsp) t_nsp = t;
            } else if (type.find("gpuss") != std::string::npos) {
                int t = readTemp(tempPath);
                if (t > t_gpu) t_gpu = t;
            } else if (type == "cpu-1-1-1") {
                t_cpu = readTemp(tempPath);
            } else if (type == "sys-therm-0") {
                t_sys = readTemp(tempPath);
            }
        }

        std::cout << "[" << (s + 1) << "s] CPU: " << (t_cpu / 1000.0) 
                  << " C | NPU: " << (t_nsp / 1000.0) 
                  << " C | GPU: " << (t_gpu / 1000.0) 
                  << " C | SYS-0: " << (t_sys / 1000.0) << " C" << std::endl;

        if (t_nsp >= 82000 || t_gpu >= 82000 || t_cpu >= 88000) {
            std::cout << "[SAFETY] Reached 82 C! Stopping probe." << std::endl;
            break;
        }
    }

    g_run.store(false, std::memory_order_relaxed);
    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }
    std::cout << "[PROBE] Completed." << std::endl;
    return 0;
}
