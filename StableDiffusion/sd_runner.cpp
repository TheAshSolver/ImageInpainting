#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <memory>
#include <algorithm>
#include <cmath>
#include <random>
#include <dlfcn.h>

// Qualcomm QNN Core Headers
#include "QnnInterface.h"
#include "QnnCommon.h"
#include "QnnTypes.h"
#include "QnnContext.h"
#include "QnnBackend.h"
#include "QnnGraph.h"
#include "QnnTensor.h"
#include "clip_tokenizer.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CHECK_QNN_STATUS(status, msg) \
    if (status != QNN_SUCCESS) { \
        std::cerr << "[QNN ERROR] " << msg << " | Error Code: " << status << std::endl; \
        return false; \
    }

// Quantization helpers for QNN_DATATYPE_UFIXED_POINT_16 (Asymmetric affine quantization)
inline uint16_t quantizeToUFix16(float val, float scale, int32_t zero_point) {
    if (scale <= 0.0f) scale = 1.0f;
    float q = std::round(val / scale) + static_cast<float>(zero_point);
    return static_cast<uint16_t>(std::clamp(q, 0.0f, 65535.0f));
}

inline float dequantizeFromUFix16(uint16_t val, float scale, int32_t zero_point) {
    return (static_cast<float>(val) - static_cast<float>(zero_point)) * scale;
}

// Universal Model Engine for QNN Context Execution
class QnnModelEngine {
private:
    void* m_backendLibHandle = nullptr;
    QNN_INTERFACE_VER_TYPE m_qnnInterface;

    Qnn_BackendHandle_t m_backendHandle = nullptr;
    Qnn_DeviceHandle_t  m_deviceHandle  = nullptr;
    Qnn_ContextHandle_t m_contextHandle = nullptr;
    Qnn_GraphHandle_t   m_graphHandle   = nullptr;

    std::vector<Qnn_Tensor_t> m_inputTensors;
    std::vector<Qnn_Tensor_t> m_outputTensors;
    std::vector<std::unique_ptr<std::vector<uint32_t>>> m_inputDims;
    std::vector<std::vector<uint32_t>> m_outputDims;

public:
    QnnModelEngine() = default;

    ~QnnModelEngine() {
        if (m_contextHandle && m_qnnInterface.contextFree) {
            m_qnnInterface.contextFree(m_contextHandle, nullptr);
        }
        if (m_deviceHandle && m_qnnInterface.deviceFree) {
            m_qnnInterface.deviceFree(m_deviceHandle);
        }
        if (m_backendHandle && m_qnnInterface.backendFree) {
            m_qnnInterface.backendFree(m_backendHandle);
        }
        if (m_backendLibHandle) dlclose(m_backendLibHandle);
    }

    void addInputTensor(uint32_t id, const char* name, Qnn_DataType_t dataType, const std::vector<uint32_t>& dims,
                        float scale = 1.0f, int32_t offset = 0) {
        m_inputDims.push_back(std::make_unique<std::vector<uint32_t>>(dims));
        Qnn_Tensor_t tensor = QNN_TENSOR_INIT;
        tensor.version = QNN_TENSOR_VERSION_1;
        tensor.v1.id = id;
        tensor.v1.name = name;
        tensor.v1.type = QNN_TENSOR_TYPE_APP_WRITE;
        tensor.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
        tensor.v1.dataType = dataType;
        
        if (dataType == QNN_DATATYPE_UFIXED_POINT_16 || dataType == QNN_DATATYPE_UFIXED_POINT_8) {
            tensor.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
            tensor.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
            tensor.v1.quantizeParams.scaleOffsetEncoding.scale = scale;
            tensor.v1.quantizeParams.scaleOffsetEncoding.offset = offset;
        } else {
            tensor.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
        }

        tensor.v1.rank = static_cast<uint32_t>(m_inputDims.back()->size());
        tensor.v1.dimensions = m_inputDims.back()->data();
        tensor.v1.memType = QNN_TENSORMEMTYPE_RAW;
        m_inputTensors.push_back(tensor);
    }

    void addOutputTensor(uint32_t id, const char* name, Qnn_DataType_t dataType, const std::vector<uint32_t>& dims,
                         float scale = 1.0f, int32_t offset = 0) {
        m_outputDims.push_back(dims);
        Qnn_Tensor_t tensor = QNN_TENSOR_INIT;
        tensor.version = QNN_TENSOR_VERSION_1;
        tensor.v1.id = id;
        tensor.v1.name = name;
        tensor.v1.type = QNN_TENSOR_TYPE_APP_READ;
        tensor.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
        tensor.v1.dataType = dataType;

        if (dataType == QNN_DATATYPE_UFIXED_POINT_16 || dataType == QNN_DATATYPE_UFIXED_POINT_8) {
            tensor.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
            tensor.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
            tensor.v1.quantizeParams.scaleOffsetEncoding.scale = scale;
            tensor.v1.quantizeParams.scaleOffsetEncoding.offset = offset;
        } else {
            tensor.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
        }

        tensor.v1.rank = static_cast<uint32_t>(m_outputDims.back().size());
        tensor.v1.dimensions = m_outputDims.back().data();
        tensor.v1.memType = QNN_TENSORMEMTYPE_RAW;
        m_outputTensors.push_back(tensor);
    }

    bool init(const std::string& binaryPath, const std::string& graphName, const std::string& htpBackendPath = "libQnnHtp.so") {
        std::ifstream file(binaryPath, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            std::cerr << "Cannot open binary: " << binaryPath << std::endl;
            return false;
        }
        std::streamsize binarySize = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> binaryBuffer(binarySize);
        if (!file.read(binaryBuffer.data(), binarySize)) {
            std::cerr << "Failed reading binary buffer" << std::endl;
            return false;
        }

        m_backendLibHandle = dlopen(htpBackendPath.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!m_backendLibHandle) {
            std::cerr << "Failed to open " << htpBackendPath << ": " << dlerror() << std::endl;
            return false;
        }

        typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(const QnnInterface_t***, uint32_t*);
        auto qnnGetProviders = (QnnInterfaceGetProvidersFn_t)dlsym(m_backendLibHandle, "QnnInterface_getProviders");
        if (!qnnGetProviders) return false;

        const QnnInterface_t** providers = nullptr;
        uint32_t numProviders = 0;
        CHECK_QNN_STATUS(qnnGetProviders(&providers, &numProviders), "Qnn get providers");
        m_qnnInterface = providers[0]->QNN_INTERFACE_VER_NAME;

        CHECK_QNN_STATUS(m_qnnInterface.backendCreate(nullptr, (const QnnBackend_Config_t**)nullptr, &m_backendHandle), "Backend Create");

        if (m_qnnInterface.deviceCreate) {
            CHECK_QNN_STATUS(m_qnnInterface.deviceCreate(nullptr, (const QnnDevice_Config_t**)nullptr, &m_deviceHandle), "Device Create");
        }

        CHECK_QNN_STATUS(m_qnnInterface.contextCreateFromBinary(
            m_backendHandle, m_deviceHandle, (const QnnContext_Config_t**)nullptr,
            static_cast<const void*>(binaryBuffer.data()), binarySize,
            &m_contextHandle, nullptr), "contextCreateFromBinary");

        CHECK_QNN_STATUS(m_qnnInterface.graphRetrieve(m_contextHandle, graphName.c_str(), &m_graphHandle), "graphRetrieve");
        return true;
    }

    bool execute(const std::vector<void*>& inputPtrs, const std::vector<size_t>& inputByteSizes,
                 const std::vector<void*>& outputPtrs, const std::vector<size_t>& outputByteSizes) {

        for (size_t i = 0; i < m_inputTensors.size(); ++i) {
            m_inputTensors[i].v1.clientBuf.data = inputPtrs[i];
            m_inputTensors[i].v1.clientBuf.dataSize = inputByteSizes[i];
        }

        for (size_t i = 0; i < m_outputTensors.size(); ++i) {
            m_outputTensors[i].v1.clientBuf.data = outputPtrs[i];
            m_outputTensors[i].v1.clientBuf.dataSize = outputByteSizes[i];
        }

        Qnn_ErrorHandle_t status = m_qnnInterface.graphExecute(
            m_graphHandle,
            m_inputTensors.data(), static_cast<uint32_t>(m_inputTensors.size()),
            m_outputTensors.data(), static_cast<uint32_t>(m_outputTensors.size()),
            nullptr, nullptr
        );

        CHECK_QNN_STATUS(status, "QnnGraph_execute");
        return true;
    }
};

// Euler noise schedule matching diffusers' EulerDiscreteScheduler
struct EulerSchedule {
    std::vector<float> timesteps;
    std::vector<float> sigmas;

    EulerSchedule(int numInferenceSteps, int trainTimesteps = 1000) {
        float beta_start = 0.00085f, beta_end = 0.0120f;
        float sqrt_start = std::sqrt(beta_start), sqrt_end = std::sqrt(beta_end);

        std::vector<double> full_sigmas(trainTimesteps);
        double prod = 1.0;
        for (int i = 0; i < trainTimesteps; ++i) {
            float sqrt_beta = sqrt_start + (sqrt_end - sqrt_start) * (float(i) / (trainTimesteps - 1));
            double beta = double(sqrt_beta) * double(sqrt_beta);
            prod *= (1.0 - beta);
            full_sigmas[i] = std::sqrt((1.0 - prod) / prod);
        }

        for (int i = 0; i < numInferenceSteps; ++i)
            timesteps.push_back(float(i) * float(trainTimesteps - 1) / float(numInferenceSteps - 1));
        std::reverse(timesteps.begin(), timesteps.end());

        for (float t : timesteps) {
            int lo = (int)std::floor(t);
            int hi = std::min(lo + 1, trainTimesteps - 1);
            float frac = t - lo;
            sigmas.push_back(float(full_sigmas[lo]) * (1 - frac) + float(full_sigmas[hi]) * frac);
        }
        sigmas.push_back(0.0f);
    }
};

int main(int argc, char** argv) {
    int numSteps = 20;
    float guidanceScale = 7.5f;
    std::string outputPath = "sd_output.png";

    std::cout << "=================================================" << std::endl;
    std::cout << "Starting SD 1.5 Pipeline on Snapdragon 8 Elite (HTP v79)" << std::endl;
    std::cout << "Inference Steps: " << numSteps << " | Guidance Scale: " << guidanceScale << std::endl;
    std::cout << "=================================================" << std::endl;

    // -------------------------------------------------------------
    // Quantization Parameters from metadata.json
    // -------------------------------------------------------------
    // text_encoder.bin
    const float   TE_OUT_SCALE         = 0.0009303585393354297f;
    const int32_t TE_OUT_ZERO_POINT    = 30063;

    // unet.bin
    const float   UNET_TIMESTEP_SCALE  = 0.014770733192563057f;
    const int32_t UNET_TIMESTEP_ZP     = 0;
    const float   UNET_LATENT_IN_SCALE = 0.00024176308943424374f;
    const int32_t UNET_LATENT_IN_ZP    = 33983;
    const float   UNET_TEXT_EMB_SCALE  = 0.0009331560577265918f;
    const int32_t UNET_TEXT_EMB_ZP     = 30103;
    const float   UNET_OUT_SCALE       = 0.0001881735515780747f;
    const int32_t UNET_OUT_ZP          = 32340;

    // vae.bin
    const float   VAE_LATENT_IN_SCALE  = 0.00034003707696683705f;
    const int32_t VAE_LATENT_IN_ZP     = 34382;
    const float   VAE_IMAGE_OUT_SCALE  = 0.000015259021893143654f;
    const int32_t VAE_IMAGE_OUT_ZP     = 0;

    // -------------------------------------------------------------
    // 1. Initializing Models with Quantized Types & Scales
    // -------------------------------------------------------------
    QnnModelEngine textEncoder, unet, vaeDecoder;

    // Text Encoder Setup
    textEncoder.addInputTensor(2, "tokens", QNN_DATATYPE_INT_32, {1, 77});
    textEncoder.addOutputTensor(707, "text_embedding", QNN_DATATYPE_UFIXED_POINT_16, {1, 77, 768}, TE_OUT_SCALE, TE_OUT_ZERO_POINT);
    if (!textEncoder.init("models/text_encoder.serialized.bin", "stable_diffusion_v1_5_text_encoder")) {
        std::cerr << "Failed to init Text Encoder." << std::endl;
        return -1;
    }

    // UNet Setup
    unet.addInputTensor(1, "timestep", QNN_DATATYPE_UFIXED_POINT_16, {1, 1}, UNET_TIMESTEP_SCALE, UNET_TIMESTEP_ZP);
    unet.addInputTensor(20, "latent", QNN_DATATYPE_UFIXED_POINT_16, {1, 64, 64, 4}, UNET_LATENT_IN_SCALE, UNET_LATENT_IN_ZP);
    unet.addInputTensor(315, "text_emb", QNN_DATATYPE_UFIXED_POINT_16, {1, 77, 768}, UNET_TEXT_EMB_SCALE, UNET_TEXT_EMB_ZP);
    unet.addOutputTensor(9009, "output_latent", QNN_DATATYPE_UFIXED_POINT_16, {1, 64, 64, 4}, UNET_OUT_SCALE, UNET_OUT_ZP);
    if (!unet.init("models/unet.serialized.bin", "stable_diffusion_v1_5_unet")) {
        std::cerr << "Failed to init UNet." << std::endl;
        return -1;
    }

    // VAE Decoder Setup
    vaeDecoder.addInputTensor(1, "latent", QNN_DATATYPE_UFIXED_POINT_16, {1, 64, 64, 4}, VAE_LATENT_IN_SCALE, VAE_LATENT_IN_ZP);
    vaeDecoder.addOutputTensor(426, "image", QNN_DATATYPE_UFIXED_POINT_16, {1, 512, 512, 3}, VAE_IMAGE_OUT_SCALE, VAE_IMAGE_OUT_ZP);
    if (!vaeDecoder.init("models/vae.serialized.bin", "stable_diffusion_v1_5_vae")) {
        std::cerr << "Failed to init VAE Decoder." << std::endl;
        return -1;
    }

    // -------------------------------------------------------------
    // 2. Text Conditioning
    // -------------------------------------------------------------
    CLIPTokenizer tokenizer;
    if (!tokenizer.load("vocab.json", "merges.txt")) {
        std::cerr << "Failed to load CLIP Tokenizer assets." << std::endl;
        return -1;
    }

    std::string userPrompt = (argc > 1) ? argv[1] : "a photo of an astronaut riding a horse";
    std::string negPrompt  = "";

    std::vector<int32_t> promptTokens = tokenizer.encode(userPrompt);
    std::vector<int32_t> uncondTokens = tokenizer.encode(negPrompt);

    const size_t textEmbElements = 1 * 77 * 768;
    std::vector<uint16_t> condTextEmbRaw(textEmbElements);
    std::vector<uint16_t> uncondTextEmbRaw(textEmbElements);

    // Conditional forward pass
    std::vector<void*> condInPtrs = { promptTokens.data() };
    std::vector<size_t> condInSizes = { promptTokens.size() * sizeof(int32_t) };
    std::vector<void*> condOutPtrs = { condTextEmbRaw.data() };
    std::vector<size_t> condOutSizes = { condTextEmbRaw.size() * sizeof(uint16_t) };
    if (!textEncoder.execute(condInPtrs, condInSizes, condOutPtrs, condOutSizes)) {
        std::cerr << "Text encoder (conditional) execution failed." << std::endl;
        return -1;
    }

    // Unconditional forward pass
    std::vector<void*> uncondInPtrs = { uncondTokens.data() };
    std::vector<size_t> uncondInSizes = { uncondTokens.size() * sizeof(int32_t) };
    std::vector<void*> uncondOutPtrs = { uncondTextEmbRaw.data() };
    std::vector<size_t> uncondOutSizes = { uncondTextEmbRaw.size() * sizeof(uint16_t) };
    if (!textEncoder.execute(uncondInPtrs, uncondInSizes, uncondOutPtrs, uncondOutSizes)) {
        std::cerr << "Text encoder (unconditional) execution failed." << std::endl;
        return -1;
    }

    // Requantize Text Encoder output -> UNet text_emb input
    std::vector<uint16_t> unetCondTextEmb(textEmbElements);
    std::vector<uint16_t> unetUncondTextEmb(textEmbElements);

    for (size_t i = 0; i < textEmbElements; ++i) {
        float condVal = dequantizeFromUFix16(condTextEmbRaw[i], TE_OUT_SCALE, TE_OUT_ZERO_POINT);
        float uncondVal = dequantizeFromUFix16(uncondTextEmbRaw[i], TE_OUT_SCALE, TE_OUT_ZERO_POINT);
        unetCondTextEmb[i] = quantizeToUFix16(condVal, UNET_TEXT_EMB_SCALE, UNET_TEXT_EMB_ZP);
        unetUncondTextEmb[i] = quantizeToUFix16(uncondVal, UNET_TEXT_EMB_SCALE, UNET_TEXT_EMB_ZP);
    }

    // -------------------------------------------------------------
    // 3. Euler Denoising Loop
    // -------------------------------------------------------------
    std::cout << "[Pipeline 2/3] Denoising Latents on HTP..." << std::endl;

    const size_t latentSize = 1 * 64 * 64 * 4;
    const size_t imagePixelCount = 512 * 512 * 3;

    std::vector<float> latents(latentSize);
    std::mt19937 rng(1337);
    std::normal_distribution<float> gaussian(0.0f, 1.0f);

    EulerSchedule schedule(numSteps);

    float init_sigma = schedule.sigmas[0];
    for (size_t i = 0; i < latentSize; ++i) {
        latents[i] = gaussian(rng) * init_sigma;
    }

    std::vector<uint16_t> ufixTimestep(1);
    std::vector<uint16_t> ufixLatentInput(latentSize);
    std::vector<uint16_t> ufixVaeInput(latentSize);
    std::vector<uint16_t> ufixNoisePredCond(latentSize);
    std::vector<uint16_t> ufixNoisePredUncond(latentSize);
    std::vector<uint16_t> ufixRgbOutput(imagePixelCount, 0);
    std::vector<uint8_t>  rgb888(imagePixelCount);

    for (int step = 0; step < numSteps; ++step) {
        float t = schedule.timesteps[step];
        float sigma = schedule.sigmas[step];
        float sigma_next = schedule.sigmas[step + 1];

        // Quantize Timestep
        ufixTimestep[0] = quantizeToUFix16(t, UNET_TIMESTEP_SCALE, UNET_TIMESTEP_ZP);

        // Scale and quantize input latents for UNet
        float inv_scale = 1.0f / std::sqrt(sigma * sigma + 1.0f);
        for (size_t i = 0; i < latentSize; ++i) {
            ufixLatentInput[i] = quantizeToUFix16(latents[i] * inv_scale, UNET_LATENT_IN_SCALE, UNET_LATENT_IN_ZP);
        }

        // UNet conditional execution
        if (!unet.execute({ufixTimestep.data(), ufixLatentInput.data(), unetCondTextEmb.data()},
                   {sizeof(uint16_t), latentSize * sizeof(uint16_t), unetCondTextEmb.size() * sizeof(uint16_t)},
                   {ufixNoisePredCond.data()}, {latentSize * sizeof(uint16_t)})) {
            std::cerr << "UNet conditional execution failed at step " << step << std::endl;
            return -1;
        }

        // UNet unconditional execution
        if (!unet.execute({ufixTimestep.data(), ufixLatentInput.data(), unetUncondTextEmb.data()},
                   {sizeof(uint16_t), latentSize * sizeof(uint16_t), unetUncondTextEmb.size() * sizeof(uint16_t)},
                   {ufixNoisePredUncond.data()}, {latentSize * sizeof(uint16_t)})) {
            std::cerr << "UNet unconditional execution failed at step " << step << std::endl;
            return -1;
        }

        // Dequantize UNet output latents and apply Classifier-Free Guidance
        for (size_t i = 0; i < latentSize; ++i) {
            float noise_cond = dequantizeFromUFix16(ufixNoisePredCond[i], UNET_OUT_SCALE, UNET_OUT_ZP);
            float noise_uncond = dequantizeFromUFix16(ufixNoisePredUncond[i], UNET_OUT_SCALE, UNET_OUT_ZP);

            float noise_pred = noise_uncond + guidanceScale * (noise_cond - noise_uncond);

            // Compute intermediate clean latent preview (x0)
            float pred_x0 = latents[i] - sigma * noise_pred;
            ufixVaeInput[i] = quantizeToUFix16(pred_x0, VAE_LATENT_IN_SCALE, VAE_LATENT_IN_ZP);

            // Update ODE trajectory for next step
            latents[i] = latents[i] + noise_pred * (sigma_next - sigma);
        }

        // Decode preview intermediate latent
        if (!vaeDecoder.execute({ufixVaeInput.data()}, {latentSize * sizeof(uint16_t)},
                                {ufixRgbOutput.data()}, {imagePixelCount * sizeof(uint16_t)})) {
            std::cerr << "VAE decoder execution failed at step " << step << std::endl;
            return -1;
        }

        for (size_t i = 0; i < imagePixelCount; ++i) {
            float val = dequantizeFromUFix16(ufixRgbOutput[i], VAE_IMAGE_OUT_SCALE, VAE_IMAGE_OUT_ZP);
            float clamped = std::clamp(val * 255.0f, 0.0f, 255.0f);
            rgb888[i] = static_cast<uint8_t>(clamped);
        }

        std::string stepOutputPath = "step_" + std::to_string(step) + "_" + outputPath;
        stbi_write_png(stepOutputPath.c_str(), 512, 512, 3, rgb888.data(), 512 * 3);

        std::cout << "\rProgress: [" << (step + 1) << "/" << numSteps << "] steps completed." << std::flush;
    }
    std::cout << "\n";

    // -------------------------------------------------------------
    // 4. Final VAE Latent Decoding
    // -------------------------------------------------------------
    std::cout << "[Pipeline 3/3] Decoding Final Latent to Image via VAE..." << std::endl;

    for (size_t i = 0; i < latentSize; ++i) {
        ufixLatentInput[i] = quantizeToUFix16(latents[i], VAE_LATENT_IN_SCALE, VAE_LATENT_IN_ZP);
    }

    if (!vaeDecoder.execute({ufixLatentInput.data()}, {latentSize * sizeof(uint16_t)},
                            {ufixRgbOutput.data()}, {imagePixelCount * sizeof(uint16_t)})) {
        std::cerr << "VAE decoder execution failed for final latent." << std::endl;
        return -1;
    }

    for (size_t i = 0; i < imagePixelCount; ++i) {
        float val = dequantizeFromUFix16(ufixRgbOutput[i], VAE_IMAGE_OUT_SCALE, VAE_IMAGE_OUT_ZP);
        float clamped = std::clamp(val * 255.0f, 0.0f, 255.0f);
        rgb888[i] = static_cast<uint8_t>(clamped);
    }

    if (stbi_write_png(outputPath.c_str(), 512, 512, 3, rgb888.data(), 512 * 3)) {
        std::cout << ">> Pipeline Complete! Final image saved to: " << outputPath << std::endl;
    } else {
        std::cerr << "Failed to save final output image file." << std::endl;
    }

    return 0;
}