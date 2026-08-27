#include <iostream>
#include <fstream>
#include <vector>
#include <dlfcn.h>

#include "System/QnnSystemContext.h"
#include "System/QnnSystemInterface.h"

void printTensor(const char* tag, uint32_t idx, const Qnn_Tensor_t& t) {
    if (t.version == QNN_TENSOR_VERSION_1) {
        std::cout << "  " << tag << " [" << idx << "] ID: " << t.v1.id
                  << " | Name: \"" << (t.v1.name ? t.v1.name : "NULL") << "\""
                  << " | DataType: 0x" << std::hex << t.v1.dataType << std::dec
                  << " | Shape: [ ";
        for (uint32_t d = 0; d < t.v1.rank; ++d) {
            std::cout << t.v1.dimensions[d] << " ";
        }
        std::cout << "]" << std::endl;
    } else if (t.version == QNN_TENSOR_VERSION_2) {
        std::cout << "  " << tag << " [" << idx << "] ID: " << t.v2.id
                  << " | Name: \"" << (t.v2.name ? t.v2.name : "NULL") << "\""
                  << " | DataType: 0x" << std::hex << t.v2.dataType << std::dec
                  << " | Shape: [ ";
        for (uint32_t d = 0; d < t.v2.rank; ++d) {
            std::cout << t.v2.dimensions[d] << " ";
        }
        std::cout << "]" << std::endl;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./dump_metadata <path_to_model.serialized.bin> [libQnnSystem.so]" << std::endl;
        return -1;
    }

    std::string binPath = argv[1];
    std::string sysLibPath = (argc > 2) ? argv[2] : "libQnnSystem.so";

    void* handle = dlopen(sysLibPath.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        std::cerr << "Failed to dlopen " << sysLibPath << ": " << dlerror() << std::endl;
        return -1;
    }

    typedef Qnn_ErrorHandle_t (*GetProvidersFn)(const QnnSystemInterface_t***, uint32_t*);
    auto getProviders = (GetProvidersFn)dlsym(handle, "QnnSystemInterface_getProviders");
    if (!getProviders) {
        std::cerr << "Failed to resolve QnnSystemInterface_getProviders" << std::endl;
        return -1;
    }

    const QnnSystemInterface_t** providers = nullptr;
    uint32_t numProviders = 0;
    if (getProviders(&providers, &numProviders) != QNN_SUCCESS || numProviders == 0) {
        std::cerr << "Failed to get system providers" << std::endl;
        return -1;
    }

    auto& sysInterface = providers[0]->QNN_SYSTEM_INTERFACE_VER_NAME;

    std::ifstream file(binPath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << binPath << std::endl;
        return -1;
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);

    QnnSystemContext_Handle_t sysCtx = nullptr;
    sysInterface.systemContextCreate(&sysCtx);

    const QnnSystemContext_BinaryInfo_t* binaryInfo = nullptr;
    Qnn_ContextBinarySize_t infoSize = 0;
    Qnn_ErrorHandle_t status = sysInterface.systemContextGetBinaryInfo(
        sysCtx, buffer.data(), size, &binaryInfo, &infoSize);

    if (status != QNN_SUCCESS || !binaryInfo) {
        std::cerr << "Failed to extract binary info! Status: " << status << std::endl;
        return -1;
    }

    std::cout << "\n================ BINARY METADATA ================" << std::endl;
    std::cout << "Binary Info Struct Version: " << binaryInfo->version << std::endl;

    if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
        auto& binV3 = binaryInfo->contextBinaryInfoV3;
        std::cout << "Number of Graphs: " << binV3.numGraphs << std::endl;

        for (uint32_t g = 0; g < binV3.numGraphs; ++g) {
            auto& gContainer = binV3.graphs[g];
            std::cout << "\n[Graph " << g << "] Container Version: " << gContainer.version << std::endl;

            if (gContainer.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
                auto& graph = gContainer.graphInfoV3;
                std::cout << "Graph Name: " << graph.graphName << std::endl;

                std::cout << "--- INPUT TENSORS (" << graph.numGraphInputs << ") ---" << std::endl;
                for (uint32_t i = 0; i < graph.numGraphInputs; ++i) {
                    printTensor("Input", i, graph.graphInputs[i]);
                }

                std::cout << "--- OUTPUT TENSORS (" << graph.numGraphOutputs << ") ---" << std::endl;
                for (uint32_t o = 0; o < graph.numGraphOutputs; ++o) {
                    printTensor("Output", o, graph.graphOutputs[o]);
                }
            } else if (gContainer.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) {
                auto& graph = gContainer.graphInfoV2;
                std::cout << "Graph Name: " << graph.graphName << std::endl;

                std::cout << "--- INPUT TENSORS (" << graph.numGraphInputs << ") ---" << std::endl;
                for (uint32_t i = 0; i < graph.numGraphInputs; ++i) {
                    printTensor("Input", i, graph.graphInputs[i]);
                }

                std::cout << "--- OUTPUT TENSORS (" << graph.numGraphOutputs << ") ---" << std::endl;
                for (uint32_t o = 0; o < graph.numGraphOutputs; ++o) {
                    printTensor("Output", o, graph.graphOutputs[o]);
                }
            } else if (gContainer.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
                auto& graph = gContainer.graphInfoV1;
                std::cout << "Graph Name: " << graph.graphName << std::endl;

                std::cout << "--- INPUT TENSORS (" << graph.numGraphInputs << ") ---" << std::endl;
                for (uint32_t i = 0; i < graph.numGraphInputs; ++i) {
                    printTensor("Input", i, graph.graphInputs[i]);
                }

                std::cout << "--- OUTPUT TENSORS (" << graph.numGraphOutputs << ") ---" << std::endl;
                for (uint32_t o = 0; o < graph.numGraphOutputs; ++o) {
                    printTensor("Output", o, graph.graphOutputs[o]);
                }
            }
        }
    }

    std::cout << "=================================================\n" << std::endl;

    sysInterface.systemContextFree(sysCtx);
    dlclose(handle);
    return 0;
}