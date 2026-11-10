#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <cstring>
#include <dlfcn.h>
#include <cstdlib>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unordered_map>

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

struct ResidentModel {
    std::string modelName;
    std::string dlcPath;
    Handle_t snpe = nullptr;
    Handle_t inputMap = nullptr;
    Handle_t outputMap = nullptr;
    std::unordered_map<std::string, void*> inputBuffers;
    std::unordered_map<std::string, size_t> inputSizes;
    std::string outputTensorName;
};

// Global function pointers
DlContainer_Open_fn DlContainer_Open;
DlContainer_Delete_fn DlContainer_Delete;
RuntimeList_Create_fn RuntimeList_Create;
RuntimeList_StringToRuntime_fn RuntimeList_StringToRuntime;
RuntimeList_Add_fn RuntimeList_Add;
RuntimeList_Delete_fn RuntimeList_Delete;
PlatformConfig_Create_fn PlatformConfig_Create;
PlatformConfig_Delete_fn PlatformConfig_Delete;
SNPEBuilder_Create_fn SNPEBuilder_Create;
SNPEBuilder_SetRuntimeProcessorOrder_fn SNPEBuilder_SetRuntimeProcessorOrder;
SNPEBuilder_SetPerformanceProfile_fn SNPEBuilder_SetPerformanceProfile;
SNPEBuilder_SetProfilingLevel_fn SNPEBuilder_SetProfilingLevel;
SNPEBuilder_SetPlatformConfig_fn SNPEBuilder_SetPlatformConfig;
SNPEBuilder_Build_fn SNPEBuilder_Build;
SNPEBuilder_Delete_fn SNPEBuilder_Delete;

SNPE_GetInputTensorNames_fn SNPE_GetInputTensorNames;
SNPE_GetOutputTensorNames_fn SNPE_GetOutputTensorNames;
StringList_Size_fn StringList_Size;
StringList_At_fn StringList_At;
SNPE_GetInputDimensions_fn SNPE_GetInputDimensions;
Util_CreateITensor_fn Util_CreateITensor;
ITensor_GetData_fn ITensor_GetData;
ITensor_GetSize_fn ITensor_GetSize;
TensorMap_Create_fn TensorMap_Create;
TensorMap_Add_fn TensorMap_Add;
TensorMap_GetTensor_Ref_fn TensorMap_GetTensor_Ref;
SNPE_ExecuteITensors_fn SNPE_ExecuteITensors;
Util_GetLastError_fn ErrorCode_GetLastErrorString;

std::unordered_map<std::string, ResidentModel> g_models;

bool initSNPEModel(const std::string& key, const std::string& dlcPath) {
    if (g_models.find(key) != g_models.end()) {
        return true; // Already loaded
    }

    std::cout << "[DAEMON] Loading resident model '" << key << "' from " << dlcPath << "..." << std::endl;
    Handle_t container = DlContainer_Open(dlcPath.c_str());
    if (!container) {
        std::cerr << "[DAEMON ERROR] Failed opening container: " << dlcPath << std::endl;
        return false;
    }

    Handle_t builder = SNPEBuilder_Create(container);
    Handle_t rList = RuntimeList_Create();
    int dspRuntime = RuntimeList_StringToRuntime("dsp");
    RuntimeList_Add(rList, dspRuntime);
    SNPEBuilder_SetRuntimeProcessorOrder(builder, rList);
    RuntimeList_Delete(rList);

    SNPEBuilder_SetPerformanceProfile(builder, 6); // BURST
    SNPEBuilder_SetProfilingLevel(builder, 0); // OFF

    auto t0 = std::chrono::high_resolution_clock::now();
    Handle_t snpe = SNPEBuilder_Build(builder);
    auto t1 = std::chrono::high_resolution_clock::now();
    double initMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (!snpe) {
        std::cerr << "[DAEMON ERROR] Build failed: " << (ErrorCode_GetLastErrorString ? ErrorCode_GetLastErrorString() : "unknown") << std::endl;
        return false;
    }

    ResidentModel m;
    m.modelName = key;
    m.dlcPath = dlcPath;
    m.snpe = snpe;
    m.inputMap = TensorMap_Create();
    m.outputMap = TensorMap_Create();

    Handle_t inNames = SNPE_GetInputTensorNames(snpe);
    size_t nIn = StringList_Size(inNames);
    for (size_t i = 0; i < nIn; ++i) {
        const char* name = StringList_At(inNames, i);
        Handle_t dims = SNPE_GetInputDimensions(snpe, name);
        Handle_t tensor = Util_CreateITensor(dims);
        size_t tSize = ITensor_GetSize(tensor);
        void* dataPtr = ITensor_GetData(tensor);
        m.inputBuffers[name] = dataPtr;
        m.inputSizes[name] = tSize;
        TensorMap_Add(m.inputMap, name, tensor);
    }

    Handle_t outNames = SNPE_GetOutputTensorNames(snpe);
    if (StringList_Size(outNames) > 0) {
        m.outputTensorName = StringList_At(outNames, 0);
    }

    g_models[key] = m;
    std::cout << "[DAEMON] ✅ Model '" << key << "' RESIDENT in FastRPC DSP memory in " << initMs << " ms! (Output: " << m.outputTensorName << ")" << std::endl;
    return true;
}

bool executeInference(
    const std::string& key,
    const std::string& imgPath,
    const std::string& maskPath,
    const std::string& outPath,
    double& outLatencyMs,
    std::string& outError
) {
    if (g_models.find(key) == g_models.end()) {
        outError = "Model not found: " + key;
        return false;
    }

    ResidentModel& m = g_models[key];

    // Read image into ITensor buffer
    if (m.inputBuffers.find("image") != m.inputBuffers.end()) {
        std::ifstream fImg(imgPath, std::ios::binary);
        if (!fImg) {
            outError = "Failed opening image file: " + imgPath;
            return false;
        }
        fImg.read(reinterpret_cast<char*>(m.inputBuffers["image"]), m.inputSizes["image"] * sizeof(float));
    }

    // Read mask into ITensor buffer
    if (m.inputBuffers.find("mask") != m.inputBuffers.end()) {
        std::ifstream fMask(maskPath, std::ios::binary);
        if (!fMask) {
            outError = "Failed opening mask file: " + maskPath;
            return false;
        }
        fMask.read(reinterpret_cast<char*>(m.inputBuffers["mask"]), m.inputSizes["mask"] * sizeof(float));
    }

    // Execute resident DSP graph!
    Handle_t outputMap = TensorMap_Create();
    auto t0 = std::chrono::high_resolution_clock::now();
    int status = SNPE_ExecuteITensors(m.snpe, m.inputMap, outputMap);
    auto t1 = std::chrono::high_resolution_clock::now();
    outLatencyMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (status != 0) {
        outError = "SNPE_ExecuteITensors returned " + std::to_string(status);
        return false;
    }

    // Retrieve output tensor and write directly to outPath
    Handle_t outTensor = TensorMap_GetTensor_Ref(outputMap, m.outputTensorName.c_str());
    if (!outTensor) {
        outError = "Output tensor not found in outputMap: " + m.outputTensorName;
        return false;
    }

    void* outData = ITensor_GetData(outTensor);
    size_t outSize = ITensor_GetSize(outTensor);

    std::ofstream fOut(outPath, std::ios::binary);
    if (!fOut) {
        outError = "Failed opening output destination: " + outPath;
        return false;
    }
    fOut.write(reinterpret_cast<const char*>(outData), outSize * sizeof(float));
    fOut.close();
    chmod(outPath.c_str(), 0666);

    return true;
}

int main(int argc, char** argv) {
    int port = 55555;
    if (argc > 1) port = std::atoi(argv[1]);

    std::cout << "========================================================" << std::endl;
    std::cout << "🚀 QUALCOMM HEXAGON HTP v79 RESIDENT SNPE DAEMON" << std::endl;
    std::cout << "========================================================" << std::endl;

    void* snpeLib = dlopen("/data/local/tmp/lama/lib/libSNPE.so", RTLD_NOW | RTLD_GLOBAL);
    if (!snpeLib) {
        std::cerr << "[DAEMON FATAL] dlopen libSNPE.so failed: " << dlerror() << std::endl;
        return 1;
    }

    DlContainer_Open = (DlContainer_Open_fn)dlsym(snpeLib, "Snpe_DlContainer_Open");
    DlContainer_Delete = (DlContainer_Delete_fn)dlsym(snpeLib, "Snpe_DlContainer_Delete");
    RuntimeList_Create = (RuntimeList_Create_fn)dlsym(snpeLib, "Snpe_RuntimeList_Create");
    RuntimeList_StringToRuntime = (RuntimeList_StringToRuntime_fn)dlsym(snpeLib, "Snpe_RuntimeList_StringToRuntime");
    RuntimeList_Add = (RuntimeList_Add_fn)dlsym(snpeLib, "Snpe_RuntimeList_Add");
    RuntimeList_Delete = (RuntimeList_Delete_fn)dlsym(snpeLib, "Snpe_RuntimeList_Delete");
    PlatformConfig_Create = (PlatformConfig_Create_fn)dlsym(snpeLib, "Snpe_PlatformConfig_Create");
    PlatformConfig_Delete = (PlatformConfig_Delete_fn)dlsym(snpeLib, "Snpe_PlatformConfig_Delete");
    SNPEBuilder_Create = (SNPEBuilder_Create_fn)dlsym(snpeLib, "Snpe_SNPEBuilder_Create");
    SNPEBuilder_SetRuntimeProcessorOrder = (SNPEBuilder_SetRuntimeProcessorOrder_fn)dlsym(snpeLib, "Snpe_SNPEBuilder_SetRuntimeProcessorOrder");
    SNPEBuilder_SetPerformanceProfile = (SNPEBuilder_SetPerformanceProfile_fn)dlsym(snpeLib, "Snpe_SNPEBuilder_SetPerformanceProfile");
    SNPEBuilder_SetProfilingLevel = (SNPEBuilder_SetProfilingLevel_fn)dlsym(snpeLib, "Snpe_SNPEBuilder_SetProfilingLevel");
    SNPEBuilder_SetPlatformConfig = (SNPEBuilder_SetPlatformConfig_fn)dlsym(snpeLib, "Snpe_SNPEBuilder_SetPlatformConfig");
    SNPEBuilder_Build = (SNPEBuilder_Build_fn)dlsym(snpeLib, "Snpe_SNPEBuilder_Build");
    SNPEBuilder_Delete = (SNPEBuilder_Delete_fn)dlsym(snpeLib, "Snpe_SNPEBuilder_Delete");

    SNPE_GetInputTensorNames = (SNPE_GetInputTensorNames_fn)dlsym(snpeLib, "Snpe_SNPE_GetInputTensorNames");
    SNPE_GetOutputTensorNames = (SNPE_GetOutputTensorNames_fn)dlsym(snpeLib, "Snpe_SNPE_GetOutputTensorNames");
    StringList_Size = (StringList_Size_fn)dlsym(snpeLib, "Snpe_StringList_Size");
    StringList_At = (StringList_At_fn)dlsym(snpeLib, "Snpe_StringList_At");
    SNPE_GetInputDimensions = (SNPE_GetInputDimensions_fn)dlsym(snpeLib, "Snpe_SNPE_GetInputDimensions");
    Util_CreateITensor = (Util_CreateITensor_fn)dlsym(snpeLib, "Snpe_Util_CreateITensor");
    ITensor_GetData = (ITensor_GetData_fn)dlsym(snpeLib, "Snpe_ITensor_GetData");
    ITensor_GetSize = (ITensor_GetSize_fn)dlsym(snpeLib, "Snpe_ITensor_GetSize");
    TensorMap_Create = (TensorMap_Create_fn)dlsym(snpeLib, "Snpe_TensorMap_Create");
    TensorMap_Add = (TensorMap_Add_fn)dlsym(snpeLib, "Snpe_TensorMap_Add");
    TensorMap_GetTensor_Ref = (TensorMap_GetTensor_Ref_fn)dlsym(snpeLib, "Snpe_TensorMap_GetTensor_Ref");
    SNPE_ExecuteITensors = (SNPE_ExecuteITensors_fn)dlsym(snpeLib, "Snpe_SNPE_ExecuteITensors");
    ErrorCode_GetLastErrorString = (Util_GetLastError_fn)dlsym(snpeLib, "Snpe_ErrorCode_GetLastErrorString");

    // Pre-warm and resident load MIGAN
    initSNPEModel("migan", "/data/local/tmp/lama/migan_htp_v79.dlc");
    // Pre-warm and resident load LaMa Dilated
    initSNPEModel("lama", "/data/local/tmp/lama/bench_cache_lama_dilated.dlc");
    // Pre-warm and resident load AOT-GAN
    initSNPEModel("aotgan", "/data/local/tmp/lama/bench_cache_aotgan.dlc");

    // Setup TCP server on 127.0.0.1:port
    int serverFd = socket(AF_INET, SOCK_STREAM, 0);
    if (serverFd < 0) {
        std::cerr << "[DAEMON FATAL] Failed creating socket" << std::endl;
        return 1;
    }

    int opt = 1;
    setsockopt(serverFd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    addr.sin_port = htons(port);

    if (bind(serverFd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        std::cerr << "[DAEMON FATAL] Failed binding to 127.0.0.1:" << port << std::endl;
        close(serverFd);
        return 1;
    }

    if (listen(serverFd, 10) < 0) {
        std::cerr << "[DAEMON FATAL] Failed listen on socket" << std::endl;
        close(serverFd);
        return 1;
    }

    std::cout << "[DAEMON] ⚡ Listening on 127.0.0.1:" << port << " — Ready for sub-60ms inpainting!" << std::endl;

    while (true) {
        sockaddr_in clientAddr{};
        socklen_t clientLen = sizeof(clientAddr);
        int clientFd = accept(serverFd, (struct sockaddr*)&clientAddr, &clientLen);
        if (clientFd < 0) continue;

        char buffer[2048];
        ssize_t bytesRead = read(clientFd, buffer, sizeof(buffer) - 1);
        if (bytesRead > 0) {
            buffer[bytesRead] = '\0';
            std::string req(buffer);
            // Trim whitespace
            while (!req.empty() && (req.back() == '\n' || req.back() == '\r')) req.pop_back();

            std::stringstream ss(req);
            std::string cmd;
            ss >> cmd;

            if (cmd == "PING") {
                const char* reply = "PONG\n";
                write(clientFd, reply, std::strlen(reply));
            } else if (cmd == "INFER") {
                std::string modelKey, imgPath, maskPath, outPath;
                ss >> modelKey >> imgPath >> maskPath >> outPath;

                double latency = 0.0;
                std::string err;
                bool ok = executeInference(modelKey, imgPath, maskPath, outPath, latency, err);
                if (ok) {
                    std::string reply = "OK " + std::to_string(latency) + "\n";
                    write(clientFd, reply.c_str(), reply.size());
                    std::cout << "[DAEMON] ⚡ INFER '" << modelKey << "' SUCCESS in " << latency << " ms!" << std::endl;
                } else {
                    std::string reply = "ERROR " + err + "\n";
                    write(clientFd, reply.c_str(), reply.size());
                    std::cerr << "[DAEMON] ❌ INFER '" << modelKey << "' FAILED: " << err << std::endl;
                }
            } else if (cmd == "STOP") {
                const char* reply = "BYE\n";
                write(clientFd, reply, std::strlen(reply));
                close(clientFd);
                break;
            } else {
                const char* reply = "ERROR Unknown command\n";
                write(clientFd, reply, std::strlen(reply));
            }
        }
        close(clientFd);
    }

    close(serverFd);
    std::cout << "[DAEMON] Daemon exited cleanly." << std::endl;
    return 0;
}
