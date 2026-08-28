#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdint>
#include <memory>
#include <algorithm>
#include <cmath>
#include <dlfcn.h>

#include "QnnInterface.h"
#include "QnnCommon.h"
#include "QnnTypes.h"
#include "QnnContext.h"
#include "QnnBackend.h"
#include "QnnGraph.h"
#include "QnnTensor.h"

#define CHECK_QNN_STATUS(status, msg) \
    if (status != QNN_SUCCESS) { \
        std::cerr << "[QNN ERROR] " << msg << " | Error Code: " << status << std::endl; \
        return false; \
    }

inline uint16_t float32ToFloat16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));

    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFFu;

    if (exponent <= 0) {
        if (exponent < -10) return static_cast<uint16_t>(sign);
        mantissa |= 0x800000u;
        uint32_t shift = static_cast<uint32_t>(14 - exponent);
        uint16_t half = static_cast<uint16_t>(sign | (mantissa >> shift));
        if ((mantissa >> (shift - 1)) & 1u) half += 1u;
        return half;
    } else if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }

    uint16_t result = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
    if ((mantissa >> 12) & 1u) result += 1u;
    return result;
}

inline float dequantizeFromUFix16(uint16_t val, float scale, int32_t zero_point) {
    return (static_cast<float>(val) - static_cast<float>(zero_point)) * scale;
}

int main() {
    std::string binaryPath = "models/vae_encoder_htp.serialized.bin";
    std::string graphName = "vae_encoder";
    std::string htpBackendPath = "libQnnHtp.so";

    // 1. Read serialized binary
    std::ifstream file(binaryPath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Cannot open binary: " << binaryPath << std::endl;
        return -1;
    }
    std::streamsize binarySize = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> binaryBuffer(binarySize);
    file.read(binaryBuffer.data(), binarySize);

    // 2. Load backend
    void* backendLibHandle = dlopen(htpBackendPath.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!backendLibHandle) {
        std::cerr << "Failed to dlopen " << htpBackendPath << ": " << dlerror() << std::endl;
        return -1;
    }

    typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(const QnnInterface_t***, uint32_t*);
    auto qnnGetProviders = (QnnInterfaceGetProvidersFn_t)dlsym(backendLibHandle, "QnnInterface_getProviders");
    const QnnInterface_t** providers = nullptr;
    uint32_t numProviders = 0;
    qnnGetProviders(&providers, &numProviders);
    auto qnnInterface = providers[0]->QNN_INTERFACE_VER_NAME;

    Qnn_BackendHandle_t backendHandle = nullptr;
    Qnn_DeviceHandle_t deviceHandle = nullptr;
    Qnn_ContextHandle_t contextHandle = nullptr;
    Qnn_GraphHandle_t graphHandle = nullptr;

    CHECK_QNN_STATUS(qnnInterface.backendCreate(nullptr, nullptr, &backendHandle), "backendCreate");
    if (qnnInterface.deviceCreate) {
        qnnInterface.deviceCreate(nullptr, nullptr, &deviceHandle);
    }

    CHECK_QNN_STATUS(qnnInterface.contextCreateFromBinary(
        backendHandle, deviceHandle, nullptr,
        binaryBuffer.data(), binarySize,
        &contextHandle, nullptr), "contextCreateFromBinary");

    CHECK_QNN_STATUS(qnnInterface.graphRetrieve(contextHandle, graphName.c_str(), &graphHandle), "graphRetrieve");

    // 3. Prepare Input Tensor (FP16: shape [1, 512, 3, 512])
    uint32_t inDims[4] = {1, 512, 3, 512};
    std::vector<uint16_t> inData(1 * 512 * 3 * 512, 0); // test zeroes

    Qnn_Tensor_t inTensor;
    std::memset(&inTensor, 0, sizeof(Qnn_Tensor_t));
    inTensor.version = QNN_TENSOR_VERSION_1;
    inTensor.v1.id = 1;
    inTensor.v1.name = "image_input_nhwc";
    inTensor.v1.type = QNN_TENSOR_TYPE_APP_WRITE;
    inTensor.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    inTensor.v1.dataType = QNN_DATATYPE_FLOAT_16;
    inTensor.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
    inTensor.v1.rank = 4;
    inTensor.v1.dimensions = inDims;
    inTensor.v1.memType = QNN_TENSORMEMTYPE_RAW;
    inTensor.v1.clientBuf.data = inData.data();
    inTensor.v1.clientBuf.dataSize = inData.size() * sizeof(uint16_t);

    // 4. Prepare Output Tensor (UFIX16: shape [1, 64, 64, 4])
    uint32_t outDims[4] = {1, 64, 64, 4};
    std::vector<uint16_t> outData(1 * 64 * 64 * 4, 0);

    Qnn_Tensor_t outTensor;
    std::memset(&outTensor, 0, sizeof(Qnn_Tensor_t));
    outTensor.version = QNN_TENSOR_VERSION_1;
    outTensor.v1.id = 819;
    outTensor.v1.name = "latent_output_nhwc";
    outTensor.v1.type = QNN_TENSOR_TYPE_APP_READ;
    outTensor.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    outTensor.v1.dataType = QNN_DATATYPE_UFIXED_POINT_16;
    outTensor.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
    outTensor.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
    outTensor.v1.quantizeParams.scaleOffsetEncoding.scale = 0.00010596314677968621f;
    outTensor.v1.quantizeParams.scaleOffsetEncoding.offset = 42348;
    outTensor.v1.rank = 4;
    outTensor.v1.dimensions = outDims;
    outTensor.v1.memType = QNN_TENSORMEMTYPE_RAW;
    outTensor.v1.clientBuf.data = outData.data();
    outTensor.v1.clientBuf.dataSize = outData.size() * sizeof(uint16_t);

    std::cout << "Executing VAE Encoder Graph on HTP..." << std::endl;
    Qnn_Tensor_t inputs[1] = { inTensor };
    Qnn_Tensor_t outputs[1] = { outTensor };

    Qnn_ErrorHandle_t status = qnnInterface.graphExecute(
        graphHandle,
        inputs, 1,
        outputs, 1,
        nullptr, nullptr
    );

    CHECK_QNN_STATUS(status, "QnnGraph_execute");
    std::cout << "VAE Encoder Graph Executed Successfully!" << std::endl;

    return 0;
}