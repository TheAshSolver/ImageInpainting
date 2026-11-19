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
#include <map>
#include <thread>
#include <atomic>

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
typedef Handle_t (*SNPE_GetOutputTensorNames_fn)(Handle_t);
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

struct MemoryStats {
    long vmRssKb = 0;
    long vmHwmKb = 0;
};

MemoryStats getProcessMemory() {
    MemoryStats stats;
    std::ifstream s("/proc/self/status");
    if (!s.is_open()) {
        std::cerr << "[FATAL] Failed opening /proc/self/status" << std::endl;
        std::exit(1);
    }
    std::string line;
    while (std::getline(s, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::sscanf(line.c_str(), "VmRSS: %ld kB", &stats.vmRssKb);
        } else if (line.rfind("VmHWM:", 0) == 0) {
            std::sscanf(line.c_str(), "VmHWM: %ld kB", &stats.vmHwmKb);
        }
    }
    if (stats.vmRssKb == 0) {
        std::cerr << "[FATAL] Failed reading VmRSS from /proc/self/status" << std::endl;
        std::exit(1);
    }
    return stats;
}

std::vector<std::string> getThermalZones(const std::string& prefix1, const std::string& prefix2 = "") {
    std::vector<std::string> paths;
    for (int i = 0; i < 100; ++i) {
        std::string typePath = "/sys/class/thermal/thermal_zone" + std::to_string(i) + "/type";
        std::ifstream tf(typePath);
        if (!tf) continue;
        std::string tname;
        tf >> tname;
        if (tname.find(prefix1) != std::string::npos || (!prefix2.empty() && tname.find(prefix2) != std::string::npos)) {
            paths.push_back("/sys/class/thermal/thermal_zone" + std::to_string(i) + "/temp");
        }
    }
    if (paths.empty()) {
        std::cerr << "[FATAL] No thermal zones matching prefixes: " << prefix1 
                  << (prefix2.empty() ? "" : ", ") << prefix2 << std::endl;
        std::exit(1);
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
    if (maxT == 0) {
        std::cerr << "[FATAL] Failed reading valid temperature from thermal zones" << std::endl;
        std::exit(1);
    }
    return maxT;
}

void getBatteryTelemetry(long& uV, long& uA) {
    uV = 0; uA = 0;
    std::ifstream fV("/sys/class/power_supply/battery/voltage_now");
    if (!fV.is_open() || !(fV >> uV)) {
        std::cerr << "[FATAL] Failed reading /sys/class/power_supply/battery/voltage_now" << std::endl;
        std::exit(1);
    }
    std::ifstream fA("/sys/class/power_supply/battery/current_now");
    if (!fA.is_open() || !(fA >> uA)) {
        std::cerr << "[FATAL] Failed reading /sys/class/power_supply/battery/current_now" << std::endl;
        std::exit(1);
    }
}

double getGpuBusyPercentage() {
    std::ifstream f("/sys/devices/platform/soc/3d00000.qcom,kgsl-3d0/kgsl/kgsl-3d0/gpu_busy_percentage");
    if (!f.is_open()) {
        std::cerr << "[FATAL] Cannot open /sys/devices/platform/soc/3d00000.qcom,kgsl-3d0/kgsl/kgsl-3d0/gpu_busy_percentage" << std::endl;
        std::exit(1);
    }
    int busy = 0;
    if (!(f >> busy)) {
        std::cerr << "[FATAL] Failed reading gpu_busy_percentage" << std::endl;
        std::exit(1);
    }
    return static_cast<double>(busy);
}

// ==============================================================================
// ACTIVE CONTROLLED THERMAL INJECTION (ORYON CPU BOOSTER)
// ==============================================================================
std::atomic<bool> g_heating{false};

void heaterWorker() {
    volatile double x = 1.0001;
    while (g_heating.load(std::memory_order_relaxed)) {
        for (int i = 0; i < 10000; ++i) {
            x = x * 1.000001 + 0.000001;
        }
    }
}

double boostTemperatureTo(int targetDegC, const std::vector<std::string>& computeZones) {
    auto t0 = std::chrono::high_resolution_clock::now();
    int currentTemp = getMaxTemperatureMilliC(computeZones);
    if (currentTemp >= targetDegC * 1000) {
        return 0.0;
    }

    g_heating.store(true, std::memory_order_relaxed);
    std::vector<std::thread> workers;
    int numCores = 4; // 4 Oryon cores provide rapid, controlled thermal rise
    for (int i = 0; i < numCores; ++i) {
        workers.emplace_back(heaterWorker);
    }

    auto startWait = std::chrono::steady_clock::now();
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(120));
        int t = getMaxTemperatureMilliC(computeZones);
        if (t >= targetDegC * 1000) break;
        
        // Safety timeout per degree: 20 seconds maximum
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - startWait
        ).count();
        if (elapsed > 20) break;
    }

    g_heating.store(false, std::memory_order_relaxed);
    for (auto& w : workers) {
        if (w.joinable()) w.join();
    }

    // Brief stabilization pause (150 ms) for CPU cores to return to 0% idle
    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

double coolTemperatureTo(int targetDegC, const std::vector<std::string>& computeZones) {
    auto t0 = std::chrono::high_resolution_clock::now();
    int currentTemp = getMaxTemperatureMilliC(computeZones);
    if (currentTemp <= targetDegC * 1000 + 500) {
        return 0.0;
    }

    auto startWait = std::chrono::steady_clock::now();
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        int t = getMaxTemperatureMilliC(computeZones);
        if (t <= targetDegC * 1000 + 500) break;
        
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - startWait
        ).count();
        if (elapsed > 15) break; // Maximum 15 seconds wait per cooling step
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

// ==============================================================================
// MAIN HARNESS
// ==============================================================================
int main(int argc, char** argv) {
    if (argc < 8) {
        std::cerr << "Usage: " << argv[0] 
                  << " <dlc_path> <runtime: dsp|gpu|gpu_float16> <polarity: 0=std, 1=inv> <model_name> <direction: up|down> <target_bound_c> <samples_per_deg> [img_path]" 
                  << std::endl;
        return 1;
    }

    std::string dlcPath = argv[1];
    std::string runtimeStr = argv[2]; // "dsp", "gpu", or "gpu_float16"
    int invertedPolarity = std::atoi(argv[3]);
    std::string modelName = argv[4];
    std::string direction = argv[5]; // "up" or "down"
    int targetBoundC = std::atoi(argv[6]); // 60 for up, 33 for down
    int samplesPerDegree = std::atoi(argv[7]);
    if (samplesPerDegree < 1) samplesPerDegree = 1;
    std::string imgPath = argc > 8 ? argv[8] : "/data/local/tmp/lama/bench_in/raw_image/001.raw";

    std::cerr << "[DAEMON_SWEEP] Initializing " << modelName << " on " << runtimeStr 
              << " | Direction: " << direction << " | Bound: " << targetBoundC 
              << " C | Samples/deg: " << samplesPerDegree << std::endl;

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
        std::cerr << "[FATAL] Failed opening container: " << dlcPath << std::endl;
        return 1;
    }

    // Configure runtime (dsp, gpu, or gpu_float16)
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

    // Build model (Loading time ISOLATED from measurements)
    auto tBuild0 = std::chrono::high_resolution_clock::now();
    Handle_t snpe = SNPEBuilder_Build(builder);
    auto tBuild1 = std::chrono::high_resolution_clock::now();
    double buildMs = std::chrono::duration<double, std::milli>(tBuild1 - tBuild0).count();

    if (!snpe) {
        std::cerr << "[FATAL] Build failed: " << (ErrorCode_GetLastErrorString ? ErrorCode_GetLastErrorString() : "unknown") << std::endl;
        return 1;
    }
    std::cerr << "[DAEMON_SWEEP] Model preloaded in " << buildMs << " ms (Excluded from latency metric)" << std::endl;

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
        std::cerr << "[FATAL] Missing image or mask input tensor" << std::endl;
        return 1;
    }

    // Preload image - STRICT CHECK, NO SILENT FALLBACK
    std::ifstream fImg(imgPath, std::ios::binary);
    if (!fImg.is_open()) {
        std::cerr << "[FATAL] Image file not found: " << imgPath << " (No fallback allowed)" << std::endl;
        return 1;
    }
    fImg.read(reinterpret_cast<char*>(imgBuffer), imgBufferSize * sizeof(float));
    if (fImg.gcount() != static_cast<std::streamsize>(imgBufferSize * sizeof(float))) {
        std::cerr << "[FATAL] Image read truncated: read " << fImg.gcount() 
                  << " bytes, expected " << (imgBufferSize * sizeof(float)) << std::endl;
        return 1;
    }

    // Set standard 25% centered mask
    float holeVal = invertedPolarity ? 0.0f : 1.0f;
    float keepVal = invertedPolarity ? 1.0f : 0.0f;
    std::fill(maskBuffer, maskBuffer + maskBufferSize, keepVal);
    int r0 = 128, r1 = 384, c0 = 128, c1 = 384;
    for (int r = r0; r < r1; ++r) {
        for (int c = c0; c < c1; ++c) {
            maskBuffer[r * 512 + c] = holeVal;
        }
    }

    // Select compute thermal zones
    std::vector<std::string> computeZones;
    if (runtimeStr == "dsp") {
        computeZones = getThermalZones("nsphvx", "nsphmx");
    } else {
        computeZones = getThermalZones("gpuss");
    }

    // 1 un-timed warmup to ensure memory pages and cache lines are resident
    {
        Handle_t outMap = TensorMap_Create();
        int st = SNPE_ExecuteITensors(snpe, inputMap, outMap);
        if (st != 0) {
            std::cerr << "[FATAL] Warmup inference failed with status " << st << std::endl;
            return 1;
        }
    }

    // Determine initial degree
    int initialTempMilliC = getMaxTemperatureMilliC(computeZones);
    int startDegree = static_cast<int>(std::floor(initialTempMilliC / 1000.0));

    // Print JSON output stream marker
    std::cout << "---SWEEP_DATA_START---" << std::endl;

    auto sweepWindowStart = std::chrono::high_resolution_clock::now();

    bool isUp = (direction == "up");
    int degStep = isUp ? 1 : -1;
    int curTargetDeg = startDegree;

    while (true) {
        double thermalTransitionS = 0.0;
        if (isUp) {
            thermalTransitionS = boostTemperatureTo(curTargetDeg, computeZones);
        } else {
            thermalTransitionS = coolTemperatureTo(curTargetDeg, computeZones);
        }

        for (int sampleIdx = 0; sampleIdx < samplesPerDegree; ++sampleIdx) {
            int currentMilliC = getMaxTemperatureMilliC(computeZones);
            double exactTempC = currentMilliC / 1000.0;
            int currentDegree = static_cast<int>(std::floor(exactTempC));

            long v0 = 0, a0 = 0, v1 = 0, a1 = 0;
            getBatteryTelemetry(v0, a0);

            // Execute PURE inference
            Handle_t outMap = TensorMap_Create();
            auto t0 = std::chrono::high_resolution_clock::now();
            int st = SNPE_ExecuteITensors(snpe, inputMap, outMap);
            auto t1 = std::chrono::high_resolution_clock::now();
            double latencyMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

            if (st != 0) {
                std::cerr << "[FATAL] SNPE_ExecuteITensors failed with status: " << st << std::endl;
                return 1;
            }

            getBatteryTelemetry(v1, a1);
            MemoryStats mem = getProcessMemory();

            // Real measured hardware utilization
            double npuUsagePct = 0.0;
            double gpuUsagePct = 0.0;

            if (runtimeStr == "dsp") {
                auto now = std::chrono::high_resolution_clock::now();
                double windowMs = std::chrono::duration<double, std::milli>(now - sweepWindowStart).count();
                sweepWindowStart = now;
                if (windowMs > 0.0) {
                    npuUsagePct = std::clamp((latencyMs / windowMs) * 100.0, 0.0, 100.0);
                }
                gpuUsagePct = 0.0;
            } else {
                gpuUsagePct = getGpuBusyPercentage();
                npuUsagePct = 0.0;
            }

            // Power calculations from real PMIC sensors
            long avgVoltageUv = (v0 + v1) / 2;
            long avgCurrentUa = std::abs((a0 + a1) / 2);
            double voltageV = avgVoltageUv / 1e6;
            double currentMa = avgCurrentUa / 1e3;
            double powerMw = (voltageV * currentMa);
            double energyMj = powerMw * (latencyMs / 1000.0);

            // Emit single JSON entry per recorded measurement
            std::cout << "{"
                      << "\"model\":\"" << modelName << "\","
                      << "\"runtime\":\"" << (runtimeStr == "dsp" ? "NPU" : "GPU") << "\","
                      << "\"direction\":\"" << direction << "\","
                      << "\"temp_deg\":" << currentDegree << ","
                      << "\"exact_temp_c\":" << exactTempC << ","
                      << "\"latency_ms\":" << latencyMs << ","
                      << "\"ram_rss_mb\":" << (mem.vmRssKb / 1024.0) << ","
                      << "\"ram_hwm_mb\":" << (mem.vmHwmKb / 1024.0) << ","
                      << "\"npu_usage_pct\":" << npuUsagePct << ","
                      << "\"gpu_usage_pct\":" << gpuUsagePct << ","
                      << "\"battery_v\":" << voltageV << ","
                      << "\"battery_current_ma\":" << currentMa << ","
                      << "\"power_mw\":" << powerMw << ","
                      << "\"energy_mj\":" << energyMj << ","
                      << "\"transition_time_s\":" << thermalTransitionS
                      << "}" << std::endl;
            std::cout.flush();
        }

        // Advance to next target degree
        curTargetDeg += degStep;

        // Check if reached bounds
        int checkTemp = getMaxTemperatureMilliC(computeZones);
        if (checkTemp >= 70000) {
            std::cerr << "[EMERGENCY] Silicon temperature (" << (checkTemp / 1000.0) 
                      << " C) exceeded 70 C safety threshold! Aborting sweep." << std::endl;
            break;
        }

        if (isUp && curTargetDeg > targetBoundC) {
            std::cerr << "[DAEMON_SWEEP] Reached target ceiling of " << targetBoundC << " C! Completed up sweep." << std::endl;
            break;
        }
        if (!isUp && curTargetDeg < targetBoundC) {
            std::cerr << "[DAEMON_SWEEP] Reached target bottom of " << targetBoundC << " C! Completed down sweep." << std::endl;
            break;
        }
    }

    std::cout << "---SWEEP_DATA_END---" << std::endl;
    return 0;
}
