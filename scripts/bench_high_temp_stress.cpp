#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <cstring>
#include <dlfcn.h>
#include <cstdlib>
#include <unistd.h>
#include <thread>
#include <atomic>
#include <iomanip>

// ==============================================================================
// SNPE C-API FUNCTION SIGNATURES (DYNAMIC LOADING VIA dlopen)
// ==============================================================================
typedef void* Handle_t;

typedef Handle_t (*DlContainer_Open_fn)(const char*);
typedef void (*DlContainer_Delete_fn)(Handle_t);
typedef Handle_t (*RuntimeList_Create_fn)();
typedef int (*RuntimeList_StringToRuntime_fn)(const char*);
typedef int (*RuntimeList_Add_fn)(Handle_t, int);
typedef void (*RuntimeList_Delete_fn)(Handle_t);
typedef Handle_t (*PlatformConfig_Create_fn)();
typedef void (*PlatformConfig_Delete_fn)(Handle_t);
typedef Handle_t (*SNPEBuilder_Create_fn)(Handle_t);
typedef Handle_t (*SNPEBuilder_SetRuntimeProcessorOrder_fn)(Handle_t, Handle_t);
typedef Handle_t (*SNPEBuilder_SetPerformanceProfile_fn)(Handle_t, int);
typedef Handle_t (*SNPEBuilder_SetProfilingLevel_fn)(Handle_t, int);
typedef Handle_t (*SNPEBuilder_SetPlatformConfig_fn)(Handle_t, Handle_t);
typedef Handle_t (*SNPEBuilder_Build_fn)(Handle_t);
typedef void (*SNPEBuilder_Delete_fn)(Handle_t);

typedef Handle_t (*SNPE_GetInputTensorNames_fn)(Handle_t);
typedef size_t (*StringList_Size_fn)(Handle_t);
typedef const char* (*StringList_At_fn)(Handle_t, size_t);
typedef Handle_t (*SNPE_GetInputDimensions_fn)(Handle_t, const char*);
typedef Handle_t (*Util_CreateITensor_fn)(Handle_t);
typedef void* (*ITensor_GetData_fn)(Handle_t);
typedef size_t (*ITensor_GetSize_fn)(Handle_t);
typedef Handle_t (*TensorMap_Create_fn)();
typedef int (*TensorMap_Add_fn)(Handle_t, const char*, Handle_t);
typedef int (*SNPE_ExecuteITensors_fn)(Handle_t, Handle_t, Handle_t);
typedef const char* (*Util_GetLastError_fn)();

// ==============================================================================
// SYSTEM & SENSOR TELEMETRY HELPERS
// ==============================================================================
struct MemoryStats {
    long vmRssKb = 0;
    long vmHwmKb = 0;
};

MemoryStats getProcessMemory() {
    MemoryStats stats;
    std::ifstream s("/proc/self/status");
    if (!s.is_open()) return stats;
    std::string line;
    while (std::getline(s, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::sscanf(line.c_str(), "VmRSS: %ld kB", &stats.vmRssKb);
        } else if (line.rfind("VmHWM:", 0) == 0) {
            std::sscanf(line.c_str(), "VmHWM: %ld kB", &stats.vmHwmKb);
        }
    }
    return stats;
}

int readSysfsInt(const std::string& path) {
    std::ifstream f(path);
    int val = 0;
    if (f >> val) return val;
    return -1;
}

long readSysfsLong(const std::string& path) {
    std::ifstream f(path);
    long val = 0;
    if (f >> val) return val;
    return -1;
}

std::vector<std::string> getThermalZones(const std::string& prefix1, const std::string& prefix2 = "") {
    std::vector<std::string> paths;
    for (int i = 0; i < 90; ++i) {
        std::string typePath = "/sys/class/thermal/thermal_zone" + std::to_string(i) + "/type";
        std::ifstream tf(typePath);
        if (!tf) continue;
        std::string tname;
        tf >> tname;
        if (tname.find(prefix1) != std::string::npos || (!prefix2.empty() && tname.find(prefix2) != std::string::npos)) {
            paths.push_back("/sys/class/thermal/thermal_zone" + std::to_string(i) + "/temp");
        }
    }
    return paths;
}

int getMaxTemperatureMilliC(const std::vector<std::string>& zones) {
    int maxT = 0;
    for (const auto& z : zones) {
        std::ifstream f(z);
        int t = 0;
        if (f >> t) {
            if (t > maxT) maxT = t;
        }
    }
    return maxT;
}

int getSysTherm0MilliC() {
    for (int i = 0; i < 90; ++i) {
        std::string tp = "/sys/class/thermal/thermal_zone" + std::to_string(i) + "/type";
        std::ifstream tf(tp);
        if (!tf) continue;
        std::string tname;
        tf >> tname;
        if (tname == "sys-therm-0") {
            return readSysfsInt("/sys/class/thermal/thermal_zone" + std::to_string(i) + "/temp");
        }
    }
    return -1;
}

// ==============================================================================
// 7-THREAD ORYON THERMAL INJECTOR (HEATING ENGINE UP TO 82°C)
// ==============================================================================
std::atomic<bool> g_heating{false};

void heaterWorker() {
    volatile double x = 1.0001;
    while (g_heating.load(std::memory_order_relaxed)) {
        for (int i = 0; i < 50000; ++i) {
            x = x * 1.000001 + 0.000001;
        }
    }
}

// Drives temperature up to targetDegC using 7 Oryon threads
double heatToTarget(int targetDegC, const std::vector<std::string>& computeZones, double maxSafetyDegC = 83.5) {
    auto t0 = std::chrono::high_resolution_clock::now();
    int currentMilliC = getMaxTemperatureMilliC(computeZones);
    if (currentMilliC >= targetDegC * 1000) {
        return 0.0;
    }

    g_heating.store(true, std::memory_order_relaxed);
    std::vector<std::thread> workers;
    int numWorkers = 7; // Use 7 cores to safely push thermal envelope past 78°C
    for (int i = 0; i < numWorkers; ++i) {
        workers.emplace_back(heaterWorker);
    }

    auto startWait = std::chrono::steady_clock::now();
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        int t = getMaxTemperatureMilliC(computeZones);
        int t_sys = getSysTherm0MilliC();

        // Safety cutoff
        if (t >= maxSafetyDegC * 1000 || t_sys >= maxSafetyDegC * 1000) {
            std::cerr << "[EMERGENCY SAFETY] Exceeded " << maxSafetyDegC << " C! Halting heater." << std::endl;
            break;
        }

        if (t >= targetDegC * 1000) {
            break;
        }

        // Timeout per step (60 seconds max to reach high temperatures)
        auto elapsedS = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - startWait
        ).count();
        if (elapsedS > 60) {
            std::cerr << "[THERMAL PLATEAU] Reached thermal equilibrium after 60s at " 
                      << (t / 1000.0) << " C." << std::endl;
            break;
        }
    }

    g_heating.store(false, std::memory_order_relaxed);
    for (auto& w : workers) {
        if (w.joinable()) w.join();
    }

    // Brief stabilization pause for CPU threads to quiesce
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

// ==============================================================================
// MAIN PROGRAM
// ==============================================================================
int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] 
                  << " <dlc_path> <runtime: dsp|gpu_float16> <model_name> [target_max_c=82] [samples_per_step=5] [csv_out_path]" 
                  << std::endl;
        return 1;
    }

    std::string dlcPath = argv[1];
    std::string runtimeStr = argv[2]; // "dsp" or "gpu_float16"
    std::string modelName = argv[3];
    int targetMaxC = (argc > 4) ? std::atoi(argv[4]) : 82;
    int samplesPerStep = (argc > 5) ? std::atoi(argv[5]) : 5;
    std::string csvOutPath = (argc > 6) ? argv[6] : "/data/local/tmp/lama/thermal_stress_telemetry.csv";

    std::cerr << "====================================================================" << std::endl;
    std::cerr << " Snapdragon 8 Elite High-Temperature Thermal Throttling Benchmark" << std::endl;
    std::cerr << " Model: " << modelName << " | Runtime: " << runtimeStr << std::endl;
    std::cerr << " Target Ceiling: " << targetMaxC << " C | Samples/Step: " << samplesPerStep << std::endl;
    std::cerr << " Governor Trip Point to verify: 78.0 C (sys-therm-0 Trip 2)" << std::endl;
    std::cerr << " Output CSV: " << csvOutPath << std::endl;
    std::cerr << "====================================================================" << std::endl;

    // Load SNPE library dynamically
    void* snpeLib = dlopen("/data/local/tmp/lama/lib/libSNPE.so", RTLD_NOW | RTLD_GLOBAL);
    if (!snpeLib) {
        std::cerr << "[FATAL] dlopen libSNPE.so failed: " << dlerror() << std::endl;
        return 1;
    }

    auto DlContainer_Open = (DlContainer_Open_fn)dlsym(snpeLib, "Snpe_DlContainer_Open");
    auto RuntimeList_Create = (RuntimeList_Create_fn)dlsym(snpeLib, "Snpe_RuntimeList_Create");
    auto RuntimeList_StringToRuntime = (RuntimeList_StringToRuntime_fn)dlsym(snpeLib, "Snpe_RuntimeList_StringToRuntime");
    auto RuntimeList_Add = (RuntimeList_Add_fn)dlsym(snpeLib, "Snpe_RuntimeList_Add");
    auto RuntimeList_Delete = (RuntimeList_Delete_fn)dlsym(snpeLib, "Snpe_RuntimeList_Delete");
    auto PlatformConfig_Create = (PlatformConfig_Create_fn)dlsym(snpeLib, "Snpe_PlatformConfig_Create");
    auto SNPEBuilder_Create = (SNPEBuilder_Create_fn)dlsym(snpeLib, "Snpe_SNPEBuilder_Create");
    auto SNPEBuilder_SetRuntimeProcessorOrder = (SNPEBuilder_SetRuntimeProcessorOrder_fn)dlsym(snpeLib, "Snpe_SNPEBuilder_SetRuntimeProcessorOrder");
    auto SNPEBuilder_SetPerformanceProfile = (SNPEBuilder_SetPerformanceProfile_fn)dlsym(snpeLib, "Snpe_SNPEBuilder_SetPerformanceProfile");
    auto SNPEBuilder_SetProfilingLevel = (SNPEBuilder_SetProfilingLevel_fn)dlsym(snpeLib, "Snpe_SNPEBuilder_SetProfilingLevel");
    auto SNPEBuilder_SetPlatformConfig = (SNPEBuilder_SetPlatformConfig_fn)dlsym(snpeLib, "Snpe_SNPEBuilder_SetPlatformConfig");
    auto SNPEBuilder_Build = (SNPEBuilder_Build_fn)dlsym(snpeLib, "Snpe_SNPEBuilder_Build");

    auto SNPE_GetInputTensorNames = (SNPE_GetInputTensorNames_fn)dlsym(snpeLib, "Snpe_SNPE_GetInputTensorNames");
    auto StringList_Size = (StringList_Size_fn)dlsym(snpeLib, "Snpe_StringList_Size");
    auto StringList_At = (StringList_At_fn)dlsym(snpeLib, "Snpe_StringList_At");
    auto SNPE_GetInputDimensions = (SNPE_GetInputDimensions_fn)dlsym(snpeLib, "Snpe_SNPE_GetInputDimensions");
    auto Util_CreateITensor = (Util_CreateITensor_fn)dlsym(snpeLib, "Snpe_Util_CreateITensor");
    auto ITensor_GetData = (ITensor_GetData_fn)dlsym(snpeLib, "Snpe_ITensor_GetData");
    auto ITensor_GetSize = (ITensor_GetSize_fn)dlsym(snpeLib, "Snpe_ITensor_GetSize");
    auto TensorMap_Create = (TensorMap_Create_fn)dlsym(snpeLib, "Snpe_TensorMap_Create");
    auto TensorMap_Add = (TensorMap_Add_fn)dlsym(snpeLib, "Snpe_TensorMap_Add");
    auto SNPE_ExecuteITensors = (SNPE_ExecuteITensors_fn)dlsym(snpeLib, "Snpe_SNPE_ExecuteITensors");
    auto ErrorCode_GetLastErrorString = (Util_GetLastError_fn)dlsym(snpeLib, "Snpe_ErrorCode_GetLastErrorString");

    // Open DLContainer
    Handle_t container = DlContainer_Open(dlcPath.c_str());
    if (!container) {
        std::cerr << "[FATAL] Failed opening DLC container: " << dlcPath << std::endl;
        return 1;
    }

    // Configure SNPE Builder
    Handle_t builder = SNPEBuilder_Create(container);
    Handle_t rList = RuntimeList_Create();
    int runtimeId = RuntimeList_StringToRuntime(runtimeStr.c_str());
    RuntimeList_Add(rList, runtimeId);
    SNPEBuilder_SetRuntimeProcessorOrder(builder, rList);
    RuntimeList_Delete(rList);

    Handle_t pConfig = PlatformConfig_Create ? PlatformConfig_Create() : nullptr;
    if (pConfig && SNPEBuilder_SetPlatformConfig) SNPEBuilder_SetPlatformConfig(builder, pConfig);
    SNPEBuilder_SetPerformanceProfile(builder, 6); // BURST
    SNPEBuilder_SetProfilingLevel(builder, 0);

    // Build Model (Isolate load time)
    auto tBuild0 = std::chrono::high_resolution_clock::now();
    Handle_t snpe = SNPEBuilder_Build(builder);
    auto tBuild1 = std::chrono::high_resolution_clock::now();
    double buildMs = std::chrono::duration<double, std::milli>(tBuild1 - tBuild0).count();

    if (!snpe) {
        std::cerr << "[FATAL] Build failed: " 
                  << (ErrorCode_GetLastErrorString ? ErrorCode_GetLastErrorString() : "unknown") 
                  << std::endl;
        return 1;
    }
    std::cerr << "[INFO] Model built and resident in memory (" << buildMs << " ms)" << std::endl;

    // Allocate Input Tensors
    Handle_t inputNames = SNPE_GetInputTensorNames(snpe);
    size_t numInputs = StringList_Size(inputNames);
    Handle_t inputMap = TensorMap_Create();
    float* imgBuffer = nullptr;
    float* maskBuffer = nullptr;
    size_t imgBufferSize = 0;
    size_t maskBufferSize = 0;

    for (size_t i = 0; i < numInputs; ++i) {
        const char* name = StringList_At(inputNames, i);
        Handle_t dims = SNPE_GetInputDimensions(snpe, name);
        Handle_t tensor = Util_CreateITensor(dims);
        size_t tensorSize = ITensor_GetSize(tensor);
        void* dataPtr = ITensor_GetData(tensor);

        std::string sName(name);
        if (sName == "image") {
            imgBuffer = reinterpret_cast<float*>(dataPtr);
            imgBufferSize = tensorSize;
        } else if (sName == "mask") {
            maskBuffer = reinterpret_cast<float*>(dataPtr);
            maskBufferSize = tensorSize;
        }
        TensorMap_Add(inputMap, name, tensor);
    }

    if (!imgBuffer || !maskBuffer) {
        std::cerr << "[FATAL] DLC model missing required 'image' or 'mask' inputs" << std::endl;
        return 1;
    }

    // Load sample image
    std::string imgPath = "/data/local/tmp/lama/bench_in/raw_image/001.raw";
    std::ifstream fImg(imgPath, std::ios::binary);
    if (!fImg.is_open()) {
        std::cerr << "[FATAL] Image file not found: " << imgPath << std::endl;
        return 1;
    }
    fImg.read(reinterpret_cast<char*>(imgBuffer), imgBufferSize * sizeof(float));

    // Fill centered mask (25% hole)
    std::fill(maskBuffer, maskBuffer + maskBufferSize, 0.0f);
    for (int r = 128; r < 384; ++r) {
        for (int c = 128; c < 384; ++c) {
            maskBuffer[r * 512 + c] = 1.0f;
        }
    }

    // Find compute thermal zones
    std::vector<std::string> computeZones;
    if (runtimeStr == "dsp") {
        computeZones = getThermalZones("nsphvx", "nsphmx");
    } else {
        computeZones = getThermalZones("gpuss");
    }

    // Warm-up inference
    {
        Handle_t outMap = TensorMap_Create();
        int st = SNPE_ExecuteITensors(snpe, inputMap, outMap);
        if (st != 0) {
            std::cerr << "[FATAL] Warm-up inference failed with code " << st << std::endl;
            return 1;
        }
    }

    // Open Output CSV
    std::ofstream csv(csvOutPath);
    if (!csv.is_open()) {
        std::cerr << "[FATAL] Cannot open CSV path for writing: " << csvOutPath << std::endl;
        return 1;
    }
    csv << "step,target_deg_c,sample_idx,junction_temp_c,sys_therm0_temp_c,"
        << "cooling_cdsp_state,cooling_kgsl_state,gpu_clk_mhz,gpu_busy_pct,"
        << "latency_ms,power_mw,energy_mj,ram_rss_mb,transition_time_s\n";

    // Target temperature steps to evaluate (from baseline up to targetMaxC)
    int initTempMilliC = getMaxTemperatureMilliC(computeZones);
    int currentDeg = static_cast<int>(std::floor(initTempMilliC / 1000.0));
    std::vector<int> targetSteps;
    for (int t = currentDeg; t <= targetMaxC; t += 2) {
        targetSteps.push_back(t);
    }
    if (targetSteps.empty() || targetSteps.back() != targetMaxC) {
        targetSteps.push_back(targetMaxC);
    }

    std::cout << "---SWEEP_DATA_START---" << std::endl;
    int globalStep = 0;

    for (int targetT : targetSteps) {
        double transitionS = heatToTarget(targetT, computeZones, 83.5);
        
        int cdspCooling = readSysfsInt("/sys/class/thermal/cooling_device38/cur_state");
        int kgslCooling = readSysfsInt("/sys/class/thermal/cooling_device35/cur_state");
        long gpuClkHz = readSysfsLong("/sys/class/kgsl/kgsl-3d0/gpuclk");
        double gpuClkMhz = (gpuClkHz > 0) ? (gpuClkHz / 1e6) : 0.0;
        int gpuBusy = readSysfsInt("/sys/devices/platform/soc/3d00000.qcom,kgsl-3d0/kgsl/kgsl-3d0/gpu_busy_percentage");

        std::vector<double> latencies;

        for (int s = 0; s < samplesPerStep; ++s) {
            int juncMilliC = getMaxTemperatureMilliC(computeZones);
            int sys0MilliC = getSysTherm0MilliC();
            double juncTempC = juncMilliC / 1000.0;
            double sys0TempC = sys0MilliC / 1000.0;

            // Power readings from battery
            long vNow = readSysfsLong("/sys/class/power_supply/battery/voltage_now");
            long iNow = std::abs(readSysfsLong("/sys/class/power_supply/battery/current_now"));
            double voltV = (vNow > 0) ? (vNow / 1e6) : 0.0;
            double currMa = (iNow > 0) ? (iNow / 1e3) : 0.0;
            double powerMw = voltV * currMa;

            // Execute isolated inference
            Handle_t outMap = TensorMap_Create();
            auto t0 = std::chrono::high_resolution_clock::now();
            int st = SNPE_ExecuteITensors(snpe, inputMap, outMap);
            auto t1 = std::chrono::high_resolution_clock::now();
            double latencyMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

            if (st != 0) {
                std::cerr << "[ERROR] SNPE_ExecuteITensors returned " << st << std::endl;
                continue;
            }

            latencies.push_back(latencyMs);
            double energyMj = powerMw * (latencyMs / 1000.0);
            MemoryStats mem = getProcessMemory();
            double rssMb = mem.vmRssKb / 1024.0;

            // Log to CSV
            csv << globalStep << ","
                << targetT << ","
                << s << ","
                << std::fixed << std::setprecision(2) << juncTempC << ","
                << std::fixed << std::setprecision(2) << sys0TempC << ","
                << cdspCooling << ","
                << kgslCooling << ","
                << std::fixed << std::setprecision(1) << gpuClkMhz << ","
                << gpuBusy << ","
                << std::fixed << std::setprecision(3) << latencyMs << ","
                << std::fixed << std::setprecision(2) << powerMw << ","
                << std::fixed << std::setprecision(2) << energyMj << ","
                << std::fixed << std::setprecision(2) << rssMb << ","
                << std::fixed << std::setprecision(2) << transitionS << "\n";
            csv.flush();

            // Emit JSON for stream monitoring
            std::cout << "{"
                      << "\"step\":" << globalStep << ","
                      << "\"target_t\":" << targetT << ","
                      << "\"sample\":" << s << ","
                      << "\"junc_c\":" << juncTempC << ","
                      << "\"sys0_c\":" << sys0TempC << ","
                      << "\"latency_ms\":" << latencyMs << ","
                      << "\"cooling_cdsp\":" << cdspCooling << ","
                      << "\"cooling_kgsl\":" << kgslCooling << ","
                      << "\"gpu_clk_mhz\":" << gpuClkMhz << ","
                      << "\"power_mw\":" << powerMw << ","
                      << "\"energy_mj\":" << energyMj
                      << "}" << std::endl;
            std::cout.flush();

            globalStep++;
        }

        // Summary of this target step
        double meanLat = 0.0;
        if (!latencies.empty()) {
            meanLat = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
        }
        std::cerr << "[STEP SUMMARY] Target " << targetT << " C -> Mean Latency: " 
                  << meanLat << " ms | CDSP state: " << cdspCooling 
                  << " | KGSL state: " << kgslCooling 
                  << " | GPU clk: " << gpuClkMhz << " MHz" << std::endl;

        // Emergency abort if board reached 83.5 C
        int checkJunc = getMaxTemperatureMilliC(computeZones);
        int checkSys0 = getSysTherm0MilliC();
        if (checkJunc >= 83500 || checkSys0 >= 83500) {
            std::cerr << "[SAFETY CEILING] Silicon die reached 83.5 C. Safely concluding sweep." << std::endl;
            break;
        }
    }

    std::cout << "---SWEEP_DATA_END---" << std::endl;
    csv.close();
    std::cerr << "[COMPLETE] High-temperature stress benchmark finished successfully." << std::endl;
    return 0;
}
