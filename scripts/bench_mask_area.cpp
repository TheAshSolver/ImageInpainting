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
#include <dirent.h>

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
typedef Handle_t (*TensorMap_GetTensor_Ref_fn)(Handle_t, const char*);
typedef int (*SNPE_ExecuteITensors_fn)(Handle_t, Handle_t, Handle_t);
typedef const char* (*Util_GetLastError_fn)();

struct MemoryStats {
    long vmRssKb = 0;
    long vmHwmKb = 0;
    long vmSizeKb = 0;
};

MemoryStats getMemoryStats() {
    MemoryStats stats;
    std::ifstream s("/proc/self/status");
    std::string line;
    while (std::getline(s, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::sscanf(line.c_str(), "VmRSS: %ld kB", &stats.vmRssKb);
        } else if (line.rfind("VmHWM:", 0) == 0) {
            std::sscanf(line.c_str(), "VmHWM: %ld kB", &stats.vmHwmKb);
        } else if (line.rfind("VmSize:", 0) == 0) {
            std::sscanf(line.c_str(), "VmSize: %ld kB", &stats.vmSizeKb);
        }
    }
    return stats;
}

std::vector<std::string> getNspThermalZones() {
    std::vector<std::string> paths;
    for (int i = 0; i < 100; ++i) {
        std::string typePath = "/sys/class/thermal/thermal_zone" + std::to_string(i) + "/type";
        std::ifstream tf(typePath);
        if (!tf) continue;
        std::string tname;
        tf >> tname;
        if (tname.find("nsp") != std::string::npos) {
            paths.push_back("/sys/class/thermal/thermal_zone" + std::to_string(i) + "/temp");
        }
    }
    return paths;
}

int getMaxTemperature(const std::vector<std::string>& zones) {
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

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <dlc_path> <polarity: 0=standard, 1=inverted> <model_name> [iterations=10] [img_path]" << std::endl;
        return 1;
    }

    std::string dlcPath = argv[1];
    int invertedPolarity = std::atoi(argv[2]); // 1 if MIGAN, 0 if LaMa / AOT-GAN
    std::string modelName = argv[3];
    int numIterations = argc > 4 ? std::atoi(argv[4]) : 10;
    std::string imgPath = argc > 5 ? argv[5] : "/data/local/tmp/lama/bench_in/raw_image/001.raw";

    std::cout << "=================================================================" << std::endl;
    std::cout << "🔬 HARDWARE MASK-AREA SENSITIVITY PROFILER (HTP v79)" << std::endl;
    std::cout << "Model: " << modelName << " | DLC: " << dlcPath << std::endl;
    std::cout << "Polarity: " << (invertedPolarity ? "INVERTED (0=hole, 1=keep)" : "STANDARD (1=hole, 0=keep)") << std::endl;
    std::cout << "Iterations per test point: " << numIterations << std::endl;
    std::cout << "=================================================================" << std::endl;

    void* snpeLib = dlopen("/data/local/tmp/lama/lib/libSNPE.so", RTLD_NOW | RTLD_GLOBAL);
    if (!snpeLib) {
        std::cerr << "[FATAL] dlopen libSNPE.so failed: " << dlerror() << std::endl;
        return 1;
    }

    auto DlContainer_Open = (DlContainer_Open_fn)dlsym(snpeLib, "Snpe_DlContainer_Open");
    auto DlContainer_Delete = (DlContainer_Delete_fn)dlsym(snpeLib, "Snpe_DlContainer_Delete");
    auto RuntimeList_Create = (RuntimeList_Create_fn)dlsym(snpeLib, "Snpe_RuntimeList_Create");
    auto RuntimeList_StringToRuntime = (RuntimeList_StringToRuntime_fn)dlsym(snpeLib, "Snpe_RuntimeList_StringToRuntime");
    auto RuntimeList_Add = (RuntimeList_Add_fn)dlsym(snpeLib, "Snpe_RuntimeList_Add");
    auto RuntimeList_Delete = (RuntimeList_Delete_fn)dlsym(snpeLib, "Snpe_RuntimeList_Delete");
    auto PlatformConfig_Create = (PlatformConfig_Create_fn)dlsym(snpeLib, "Snpe_PlatformConfig_Create");
    auto PlatformConfig_Delete = (PlatformConfig_Delete_fn)dlsym(snpeLib, "Snpe_PlatformConfig_Delete");
    auto SNPEBuilder_Create = (SNPEBuilder_Create_fn)dlsym(snpeLib, "Snpe_SNPEBuilder_Create");
    auto SNPEBuilder_SetRuntimeProcessorOrder = (SNPEBuilder_SetRuntimeProcessorOrder_fn)dlsym(snpeLib, "Snpe_SNPEBuilder_SetRuntimeProcessorOrder");
    auto SNPEBuilder_SetPerformanceProfile = (SNPEBuilder_SetPerformanceProfile_fn)dlsym(snpeLib, "Snpe_SNPEBuilder_SetPerformanceProfile");
    auto SNPEBuilder_SetProfilingLevel = (SNPEBuilder_SetProfilingLevel_fn)dlsym(snpeLib, "Snpe_SNPEBuilder_SetProfilingLevel");
    auto SNPEBuilder_SetPlatformConfig = (SNPEBuilder_SetPlatformConfig_fn)dlsym(snpeLib, "Snpe_SNPEBuilder_SetPlatformConfig");
    auto SNPEBuilder_Build = (SNPEBuilder_Build_fn)dlsym(snpeLib, "Snpe_SNPEBuilder_Build");
    auto SNPEBuilder_Delete = (SNPEBuilder_Delete_fn)dlsym(snpeLib, "Snpe_SNPEBuilder_Delete");

    auto SNPE_GetInputTensorNames = (SNPE_GetInputTensorNames_fn)dlsym(snpeLib, "Snpe_SNPE_GetInputTensorNames");
    auto SNPE_GetOutputTensorNames = (SNPE_GetOutputTensorNames_fn)dlsym(snpeLib, "Snpe_SNPE_GetOutputTensorNames");
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

    // Build SNPE DSP model
    Handle_t container = DlContainer_Open(dlcPath.c_str());
    if (!container) {
        std::cerr << "[FATAL] Failed opening container: " << dlcPath << std::endl;
        return 1;
    }

    Handle_t builder = SNPEBuilder_Create(container);
    Handle_t rList = RuntimeList_Create();
    int dspRuntime = RuntimeList_StringToRuntime("dsp");
    RuntimeList_Add(rList, dspRuntime);
    SNPEBuilder_SetRuntimeProcessorOrder(builder, rList);
    RuntimeList_Delete(rList);

    Handle_t pConfig = PlatformConfig_Create ? PlatformConfig_Create() : nullptr;
    if (pConfig && SNPEBuilder_SetPlatformConfig) SNPEBuilder_SetPlatformConfig(builder, pConfig);
    SNPEBuilder_SetPerformanceProfile(builder, 6); // BURST
    SNPEBuilder_SetProfilingLevel(builder, 0); // OFF

    auto tBuild0 = std::chrono::high_resolution_clock::now();
    Handle_t snpe = SNPEBuilder_Build(builder);
    auto tBuild1 = std::chrono::high_resolution_clock::now();
    double buildMs = std::chrono::duration<double, std::milli>(tBuild1 - tBuild0).count();

    if (!snpe) {
        std::cerr << "[FATAL] SNPE build failed: " << (ErrorCode_GetLastErrorString ? ErrorCode_GetLastErrorString() : "unknown") << std::endl;
        return 1;
    }
    std::cout << "✅ SNPE built in " << buildMs << " ms! Handle: " << snpe << std::endl;

    // Allocate Input ITensors
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
        std::cerr << "[FATAL] Could not locate 'image' or 'mask' input tensors" << std::endl;
        return 1;
    }

    // Load actual image if available, else standard constant
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
    std::cout << "Loaded image from " << imgPath << " (" << imgBufferSize << " floats)" << std::endl;

    std::vector<std::string> nspZones = getNspThermalZones();

    // Data points to test (Mask Area percentages)
    std::vector<double> maskPcts = {5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 99.0};

    std::cout << "\n{\"test_results\": [" << std::endl;

    for (size_t pIdx = 0; pIdx < maskPcts.size(); ++pIdx) {
        double pct = maskPcts[pIdx];
        double fraction = pct / 100.0;

        // Construct 512x512 mask: centered square with exact coverage
        int side = static_cast<int>(std::round(512.0 * std::sqrt(fraction)));
        side = std::clamp(side, 1, 512);
        int r0 = (512 - side) / 2;
        int r1 = r0 + side;
        int c0 = (512 - side) / 2;
        int c1 = c0 + side;

        float holeVal = invertedPolarity ? 0.0f : 1.0f;
        float keepVal = invertedPolarity ? 1.0f : 0.0f;

        // Initialize background
        std::fill(maskBuffer, maskBuffer + maskBufferSize, keepVal);

        // Fill hole region
        long actualHolePixels = 0;
        for (int r = r0; r < r1; ++r) {
            for (int c = c0; c < c1; ++c) {
                maskBuffer[r * 512 + c] = holeVal;
                actualHolePixels++;
            }
        }
        double actualPct = (static_cast<double>(actualHolePixels) / (512.0 * 512.0)) * 100.0;

        // Cooldown / Pre-read
        int tempStart_mC = getMaxTemperature(nspZones);
        MemoryStats memBefore = getMemoryStats();

        // 3 Warm-up runs
        for (int w = 0; w < 3; ++w) {
            Handle_t outMap = TensorMap_Create();
            SNPE_ExecuteITensors(snpe, inputMap, outMap);
        }

        // Timed measurement iterations
        std::vector<double> latenciesMs;
        latenciesMs.reserve(numIterations);

        for (int it = 0; it < numIterations; ++it) {
            Handle_t outMap = TensorMap_Create();
            auto t0 = std::chrono::high_resolution_clock::now();
            int st = SNPE_ExecuteITensors(snpe, inputMap, outMap);
            auto t1 = std::chrono::high_resolution_clock::now();
            double elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

            if (st == 0) {
                latenciesMs.push_back(elapsedMs);
            }
        }

        int tempEnd_mC = getMaxTemperature(nspZones);
        MemoryStats memAfter = getMemoryStats();

        // Calculate statistics
        double sum = std::accumulate(latenciesMs.begin(), latenciesMs.end(), 0.0);
        double meanMs = latenciesMs.empty() ? 0.0 : sum / latenciesMs.size();

        double sq_sum = 0.0;
        for (double l : latenciesMs) sq_sum += (l - meanMs) * (l - meanMs);
        double stdevMs = latenciesMs.size() > 1 ? std::sqrt(sq_sum / (latenciesMs.size() - 1)) : 0.0;

        std::sort(latenciesMs.begin(), latenciesMs.end());
        double minMs = latenciesMs.empty() ? 0.0 : latenciesMs.front();
        double maxMs = latenciesMs.empty() ? 0.0 : latenciesMs.back();
        double medMs = latenciesMs.empty() ? 0.0 : latenciesMs[latenciesMs.size() / 2];

        // Output JSON entry
        std::cout << "  {"
                  << "\"model\": \"" << modelName << "\", "
                  << "\"target_pct\": " << pct << ", "
                  << "\"actual_pct\": " << actualPct << ", "
                  << "\"hole_pixels\": " << actualHolePixels << ", "
                  << "\"hole_side\": " << side << ", "
                  << "\"iterations\": " << latenciesMs.size() << ", "
                  << "\"mean_latency_ms\": " << meanMs << ", "
                  << "\"stdev_latency_ms\": " << stdevMs << ", "
                  << "\"min_latency_ms\": " << minMs << ", "
                  << "\"max_latency_ms\": " << maxMs << ", "
                  << "\"median_latency_ms\": " << medMs << ", "
                  << "\"vm_rss_kb\": " << memAfter.vmRssKb << ", "
                  << "\"vm_hwm_kb\": " << memAfter.vmHwmKb << ", "
                  << "\"vm_size_kb\": " << memAfter.vmSizeKb << ", "
                  << "\"temp_start_c\": " << (tempStart_mC / 1000.0) << ", "
                  << "\"temp_end_c\": " << (tempEnd_mC / 1000.0)
                  << "}" << (pIdx + 1 < maskPcts.size() ? "," : "")
                  << std::endl;
    }

    std::cout << "]}" << std::endl;

    return 0;
}
