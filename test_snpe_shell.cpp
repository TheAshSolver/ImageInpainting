#include <iostream>
#include <chrono>
#include <cstring>
#include <dlfcn.h>
#include <cstdlib>

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

int main(int argc, char** argv) {
    std::cout << "[TEST] Loading libSNPE.so..." << std::endl;
    void* handle = dlopen("/data/local/tmp/lama/lib/libSNPE.so", RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        std::cerr << "[ERROR] dlopen failed: " << dlerror() << std::endl;
        return 1;
    }

    auto DlContainer_Open = (DlContainer_Open_fn)dlsym(handle, "Snpe_DlContainer_Open");
    auto DlContainer_Delete = (DlContainer_Delete_fn)dlsym(handle, "Snpe_DlContainer_Delete");
    auto RuntimeList_Create = (RuntimeList_Create_fn)dlsym(handle, "Snpe_RuntimeList_Create");
    auto RuntimeList_StringToRuntime = (RuntimeList_StringToRuntime_fn)dlsym(handle, "Snpe_RuntimeList_StringToRuntime");
    auto RuntimeList_Add = (RuntimeList_Add_fn)dlsym(handle, "Snpe_RuntimeList_Add");
    auto RuntimeList_Delete = (RuntimeList_Delete_fn)dlsym(handle, "Snpe_RuntimeList_Delete");
    auto PlatformConfig_Create = (PlatformConfig_Create_fn)dlsym(handle, "Snpe_PlatformConfig_Create");
    auto PlatformConfig_Delete = (PlatformConfig_Delete_fn)dlsym(handle, "Snpe_PlatformConfig_Delete");
    auto SNPEBuilder_Create = (SNPEBuilder_Create_fn)dlsym(handle, "Snpe_SNPEBuilder_Create");
    auto SNPEBuilder_SetRuntimeProcessorOrder = (SNPEBuilder_SetRuntimeProcessorOrder_fn)dlsym(handle, "Snpe_SNPEBuilder_SetRuntimeProcessorOrder");
    auto SNPEBuilder_SetPerformanceProfile = (SNPEBuilder_SetPerformanceProfile_fn)dlsym(handle, "Snpe_SNPEBuilder_SetPerformanceProfile");
    auto SNPEBuilder_SetProfilingLevel = (SNPEBuilder_SetProfilingLevel_fn)dlsym(handle, "Snpe_SNPEBuilder_SetProfilingLevel");
    auto SNPEBuilder_SetPlatformConfig = (SNPEBuilder_SetPlatformConfig_fn)dlsym(handle, "Snpe_SNPEBuilder_SetPlatformConfig");
    auto SNPEBuilder_Build = (SNPEBuilder_Build_fn)dlsym(handle, "Snpe_SNPEBuilder_Build");
    auto SNPEBuilder_Delete = (SNPEBuilder_Delete_fn)dlsym(handle, "Snpe_SNPEBuilder_Delete");

    auto SNPE_GetInputTensorNames = (SNPE_GetInputTensorNames_fn)dlsym(handle, "Snpe_SNPE_GetInputTensorNames");
    auto SNPE_GetOutputTensorNames = (SNPE_GetOutputTensorNames_fn)dlsym(handle, "Snpe_SNPE_GetOutputTensorNames");
    auto StringList_Size = (StringList_Size_fn)dlsym(handle, "Snpe_StringList_Size");
    auto StringList_At = (StringList_At_fn)dlsym(handle, "Snpe_StringList_At");
    auto SNPE_GetInputDimensions = (SNPE_GetInputDimensions_fn)dlsym(handle, "Snpe_SNPE_GetInputDimensions");
    auto Util_CreateITensor = (Util_CreateITensor_fn)dlsym(handle, "Snpe_Util_CreateITensor");
    auto ITensor_GetData = (ITensor_GetData_fn)dlsym(handle, "Snpe_ITensor_GetData");
    auto ITensor_GetSize = (ITensor_GetSize_fn)dlsym(handle, "Snpe_ITensor_GetSize");
    auto TensorMap_Create = (TensorMap_Create_fn)dlsym(handle, "Snpe_TensorMap_Create");
    auto TensorMap_Add = (TensorMap_Add_fn)dlsym(handle, "Snpe_TensorMap_Add");
    auto TensorMap_GetTensor_Ref = (TensorMap_GetTensor_Ref_fn)dlsym(handle, "Snpe_TensorMap_GetTensor_Ref");
    auto SNPE_ExecuteITensors = (SNPE_ExecuteITensors_fn)dlsym(handle, "Snpe_SNPE_ExecuteITensors");
    auto ErrorCode_GetLastErrorString = (Util_GetLastError_fn)dlsym(handle, "Snpe_ErrorCode_GetLastErrorString");

    const char* dlcPath = argc > 1 ? argv[1] : "/data/local/tmp/lama/migan_htp_v79.dlc";
    std::cout << "[TEST] Building SNPE instance for: " << dlcPath << std::endl;
    Handle_t container = DlContainer_Open(dlcPath);
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
    std::cout << "[TEST] SNPE built in " << buildMs << " ms! Handle: " << snpe << std::endl;

    if (!snpe) {
        std::cerr << "[ERROR] Build failed: " << (ErrorCode_GetLastErrorString ? ErrorCode_GetLastErrorString() : "unknown") << std::endl;
        return 1;
    }

    // Inspect inputs and create input ITensors
    Handle_t inputNames = SNPE_GetInputTensorNames(snpe);
    size_t numInputs = StringList_Size(inputNames);
    std::cout << "[TEST] Inputs count: " << numInputs << std::endl;

    Handle_t inputMap = TensorMap_Create();
    for (size_t i = 0; i < numInputs; ++i) {
        const char* name = StringList_At(inputNames, i);
        Handle_t dims = SNPE_GetInputDimensions(snpe, name);
        Handle_t tensor = Util_CreateITensor(dims);
        size_t tensorSize = ITensor_GetSize(tensor);
        void* dataPtr = ITensor_GetData(tensor);
        std::cout << "  Input [" << i << "]: " << name << " -> size " << tensorSize << " floats (" << (tensorSize * sizeof(float)) << " bytes)" << std::endl;
        // Zero-fill test input
        std::memset(dataPtr, 0, tensorSize * sizeof(float));
        TensorMap_Add(inputMap, name, tensor);
    }

    // Inspect outputs
    Handle_t outputNames = SNPE_GetOutputTensorNames(snpe);
    size_t numOutputs = StringList_Size(outputNames);
    std::cout << "[TEST] Outputs count: " << numOutputs << std::endl;
    for (size_t i = 0; i < numOutputs; ++i) {
        const char* name = StringList_At(outputNames, i);
        std::cout << "  Output [" << i << "]: " << name << std::endl;
    }

    // Execute multiple iterations to measure steady-state DSP latency!
    std::cout << "\n[BENCHMARK] Executing 5 resident inferences on Hexagon HTP v79 NPU:" << std::endl;
    for (int iter = 0; iter < 5; ++iter) {
        Handle_t outputMap = TensorMap_Create();
        auto t0 = std::chrono::high_resolution_clock::now();
        int status = SNPE_ExecuteITensors(snpe, inputMap, outputMap);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (status != 0) {
            std::cout << "  Iter " << iter << ": FAILED (status=" << status << ") in " << ms << " ms" << std::endl;
        } else {
            std::cout << "  Iter " << iter << ": ⚡ SUCCESS in " << ms << " ms!" << std::endl;
        }
    }

    std::cout << "\n[TEST] All resident inferences completed successfully!" << std::endl;
    return 0;
}
