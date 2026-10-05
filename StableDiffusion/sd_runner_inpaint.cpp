#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdint>
#include <memory>
#include <algorithm>
#include <cmath>
#include <random>
#include <chrono>
#include <dlfcn.h>

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

#define VAE_018215_BAKED_INTO_QUANT_SCALE 1

inline float applyVaeEncScale(float dequantizedLatent) {
#if VAE_018215_BAKED_INTO_QUANT_SCALE
    return dequantizedLatent;
#else
    return dequantizedLatent * 0.18215f;
#endif
}

inline float applyVaeDecScale(float latent) {
#if VAE_018215_BAKED_INTO_QUANT_SCALE
    return latent;
#else
    return latent * (1.0f / 0.18215f);
#endif
}

inline uint16_t quantizeToUFix16(float val, float scale, int32_t zero_point) {
    if (scale <= 0.0f) scale = 1.0f;
    float q = std::round(val / scale) + static_cast<float>(zero_point);
    return static_cast<uint16_t>(std::clamp(q, 0.0f, 65535.0f));
}

inline float dequantizeFromUFix16(uint16_t val, float scale, int32_t zero_point) {
    return (static_cast<float>(val) - static_cast<float>(zero_point)) * scale;
}

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
    std::vector<std::unique_ptr<std::vector<uint32_t>>> m_outputDims;

public:
    QnnModelEngine() = default;

    ~QnnModelEngine() {
        if (m_contextHandle && m_qnnInterface.contextFree) m_qnnInterface.contextFree(m_contextHandle, nullptr);
        if (m_deviceHandle && m_qnnInterface.deviceFree) m_qnnInterface.deviceFree(m_deviceHandle);
        if (m_backendHandle && m_qnnInterface.backendFree) m_qnnInterface.backendFree(m_backendHandle);
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

        if (dataType == QNN_DATATYPE_UFIXED_POINT_16 || dataType == QNN_DATATYPE_UFIXED_POINT_8 ||
            dataType == QNN_DATATYPE_SFIXED_POINT_16 || dataType == QNN_DATATYPE_SFIXED_POINT_8) {
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
        m_outputDims.push_back(std::make_unique<std::vector<uint32_t>>(dims));
        Qnn_Tensor_t tensor = QNN_TENSOR_INIT;
        tensor.version = QNN_TENSOR_VERSION_1;
        tensor.v1.id = id;
        tensor.v1.name = name;
        tensor.v1.type = QNN_TENSOR_TYPE_APP_READ;
        tensor.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
        tensor.v1.dataType = dataType;

        if (dataType == QNN_DATATYPE_UFIXED_POINT_16 || dataType == QNN_DATATYPE_UFIXED_POINT_8 ||
            dataType == QNN_DATATYPE_SFIXED_POINT_16 || dataType == QNN_DATATYPE_SFIXED_POINT_8) {
            tensor.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
            tensor.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
            tensor.v1.quantizeParams.scaleOffsetEncoding.scale = scale;
            tensor.v1.quantizeParams.scaleOffsetEncoding.offset = offset;
        } else {
            tensor.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
        }

        tensor.v1.rank = static_cast<uint32_t>(m_outputDims.back()->size());
        tensor.v1.dimensions = m_outputDims.back()->data();
        tensor.v1.memType = QNN_TENSORMEMTYPE_RAW;
        m_outputTensors.push_back(tensor);
    }

    bool init(const std::string& binaryPath, const std::string& graphName, const std::string& htpBackendPath = "libQnnHtp.so") {
        std::ifstream file(binaryPath, std::ios::binary | std::ios::ate);
        if (!file.is_open()) return false;
        std::streamsize binarySize = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> binaryBuffer(binarySize);
        if (!file.read(binaryBuffer.data(), binarySize)) return false;

        m_backendLibHandle = dlopen(htpBackendPath.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!m_backendLibHandle) return false;

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
        return m_qnnInterface.graphExecute(
            m_graphHandle,
            m_inputTensors.data(), static_cast<uint32_t>(m_inputTensors.size()),
            m_outputTensors.data(), static_cast<uint32_t>(m_outputTensors.size()),
            nullptr, nullptr
        ) == QNN_SUCCESS;
    }
};

// =====================================================================
// DPM-Solver++ (2M) with Karras Sigmas (Diffusers-Aligned for 8 Steps)
// =====================================================================
struct DPMSolverSchedule {
    int numSteps;
    std::vector<float> timesteps;
    std::vector<float> sigmas;
    std::vector<float> alpha_t;
    std::vector<float> sigma_t;
    std::vector<float> lambda_t;
    const float init_noise_sigma = 1.0f;

    DPMSolverSchedule(int numInferenceSteps, int trainTimesteps = 1000)
        : numSteps(numInferenceSteps) {
        float beta_start = 0.00085f, beta_end = 0.0120f;
        float sqrt_start = std::sqrt(beta_start), sqrt_end = std::sqrt(beta_end);

        std::vector<double> train_sigmas(trainTimesteps);
        std::vector<double> train_log_sigmas(trainTimesteps);

        double prod = 1.0;
        for (int i = 0; i < trainTimesteps; ++i) {
            float sqrt_beta = sqrt_start + (sqrt_end - sqrt_start) * (float(i) / (trainTimesteps - 1));
            double beta = double(sqrt_beta) * double(sqrt_beta);
            prod *= (1.0 - beta);
            train_sigmas[i] = std::sqrt((1.0 - prod) / prod);
            train_log_sigmas[i] = std::log(train_sigmas[i]);
        }

        double sigma_min = train_sigmas[0];
        double sigma_max = train_sigmas[trainTimesteps - 1];
        double rho = 7.0;
        double min_inv_rho = std::pow(sigma_min, 1.0 / rho);
        double max_inv_rho = std::pow(sigma_max, 1.0 / rho);

        sigmas.resize(numSteps + 1);
        timesteps.resize(numSteps);

        for (int i = 0; i < numSteps; ++i) {
            double ramp = double(i) / double(numSteps - 1);
            sigmas[i] = static_cast<float>(std::pow(max_inv_rho + ramp * (min_inv_rho - max_inv_rho), rho));

            double target_log_sigma = std::log(double(sigmas[i]));
            auto it = std::lower_bound(train_log_sigmas.begin(), train_log_sigmas.end(), target_log_sigma);
            int idx = static_cast<int>(it - train_log_sigmas.begin());
            if (idx == 0) {
                timesteps[i] = 0.0f;
            } else if (idx >= trainTimesteps) {
                timesteps[i] = static_cast<float>(trainTimesteps - 1);
            } else {
                double low_log = train_log_sigmas[idx - 1];
                double high_log = train_log_sigmas[idx];
                double frac = (target_log_sigma - low_log) / (high_log - low_log);
                // Continuous float timestep for smooth sinusoidal positional embeddings
                timesteps[i] = static_cast<float>((idx - 1) + frac);
            }
        }
        sigmas[numSteps] = 0.0f;

        alpha_t.resize(numSteps + 1);
        sigma_t.resize(numSteps + 1);
        lambda_t.resize(numSteps + 1);

        for (int i = 0; i < numSteps; ++i) {
            float s = sigmas[i];
            alpha_t[i] = 1.0f / std::sqrt(s * s + 1.0f);
            sigma_t[i] = s * alpha_t[i];
            lambda_t[i] = -std::log(s);
        }
        alpha_t[numSteps] = 1.0f;
        sigma_t[numSteps] = 0.0f;
        lambda_t[numSteps] = 1e9f;
    }
};

bool readRawFloats(const std::string& path, std::vector<float>& buf, size_t expectedElements) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    buf.resize(expectedElements);
    f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(expectedElements * sizeof(float)));
    return f.gcount() == static_cast<std::streamsize>(expectedElements * sizeof(float));
}

void downsampleMaskNearest(const std::vector<float>& mask512, std::vector<float>& mask64) {
    mask64.resize(64 * 64);
    for (int h = 0; h < 64; ++h) {
        for (int w = 0; w < 64; ++w) {
            mask64[h * 64 + w] = (mask512[(h * 8) * 512 + (w * 8)] > 0.5f) ? 1.0f : 0.0f;
        }
    }
}

int main(int argc, char** argv) {
    int numSteps = (argc > 2) ? std::atoi(argv[2]) : 12;
    float guidanceScale = 2.0f;
    std::string outputPath = "sd_output.png";

    std::cout << "==========================================================" << std::endl;
    std::cout << "SD 1.5 Inpainting on Qualcomm Snapdragon 8 Elite (QIDK)" << std::endl;
    std::cout << "Scheduler: DPM-Solver++ (2M) with Karras Sigmas" << std::endl;
    std::cout << "Target Steps: " << numSteps << " | Guidance Scale: " << guidanceScale << std::endl;
    std::cout << "==========================================================" << std::endl;

    const float TE_OUT_SCALE         = 0.0009303585393354297f;
    const int32_t TE_OUT_ZP          = 30063;
    const float VAE_LATENT_IN_SCALE  = 0.00034003707696683705f;
    const int32_t VAE_LATENT_IN_ZP   = 34382;
    const float VAE_IMAGE_OUT_SCALE  = 0.000015259021893143654f;
    const int32_t VAE_IMAGE_OUT_ZP   = 0;
    const float VAE_ENC_IN_SCALE     = 0.00003051804378628731f;
    const int32_t VAE_ENC_IN_ZP      = 32768;
    const float VAE_ENC_OUT_SCALE    = 0.00010813291009981185f;
    const int32_t VAE_ENC_OUT_ZP     = 42195;

    const size_t HW = 64 * 64;
    const size_t latent4Size = 4 * HW;
    const size_t sample16Size = 16 * HW;
    const size_t imagePixelCount = 512 * 512 * 3;
    const size_t maskPixelCount = 512 * 512;
    const size_t textEmbElements = 77 * 768;

    QnnModelEngine textEncoder, unet, vaeDecoder, vaeEncoder;

    std::cout << "[Init] Loading QNN engines from models/ directory..." << std::endl;
    textEncoder.addInputTensor(2, "tokens", QNN_DATATYPE_INT_32, {1, 77});
    textEncoder.addOutputTensor(707, "text_embedding", QNN_DATATYPE_UFIXED_POINT_16, {1, 77, 768}, TE_OUT_SCALE, TE_OUT_ZP);
    if (!textEncoder.init("models/text_encoder.serialized.bin", "stable_diffusion_v1_5_text_encoder")) {
        std::cerr << "Failed to init Text Encoder." << std::endl;
        return -1;
    }

    unet.addInputTensor(1, "timestep", QNN_DATATYPE_FLOAT_32, {1});
    unet.addInputTensor(22, "sample", QNN_DATATYPE_FLOAT_32, {1, 16, 64, 64});
    unet.addInputTensor(305, "encoder_hidden_states", QNN_DATATYPE_FLOAT_32, {1, 77, 768});
    unet.addOutputTensor(8669, "out_sample", QNN_DATATYPE_FLOAT_32, {1, 4, 64, 64});
    if (!unet.init("models/unet.bin", "graph_wlqbe2kd")) {
        std::cerr << "Failed to init UNet." << std::endl;
        return -1;
    }

    vaeDecoder.addInputTensor(1, "latent", QNN_DATATYPE_UFIXED_POINT_16, {1, 64, 64, 4}, VAE_LATENT_IN_SCALE, VAE_LATENT_IN_ZP);
    vaeDecoder.addOutputTensor(426, "image", QNN_DATATYPE_UFIXED_POINT_16, {1, 512, 512, 3}, VAE_IMAGE_OUT_SCALE, VAE_IMAGE_OUT_ZP);
    if (!vaeDecoder.init("models/vae.serialized.bin", "stable_diffusion_v1_5_vae")) {
        std::cerr << "Failed to init VAE Decoder." << std::endl;
        return -1;
    }

    vaeEncoder.addInputTensor(1, "image_input", QNN_DATATYPE_UFIXED_POINT_16, {1, 512, 512, 3}, VAE_ENC_IN_SCALE, VAE_ENC_IN_ZP);
    vaeEncoder.addOutputTensor(1285, "latent_output", QNN_DATATYPE_UFIXED_POINT_16, {1, 64, 64, 4}, VAE_ENC_OUT_SCALE, VAE_ENC_OUT_ZP);
    if (!vaeEncoder.init("models/vae_encoder_htp.serialized.bin", "vae_encoder")) {
        std::cerr << "Failed to init VAE Encoder." << std::endl;
        return -1;
    }

    std::vector<float> imageBuf, maskBuf512, maskBuf64;
    if (!readRawFloats("image.raw", imageBuf, imagePixelCount)) {
        std::cerr << "Failed reading image.raw! Verify file size is 3145728 bytes." << std::endl;
        return -1;
    }
    if (!readRawFloats("mask.raw", maskBuf512, maskPixelCount)) {
        std::cerr << "Failed reading mask.raw! Verify file size is 1048576 bytes." << std::endl;
        return -1;
    }
    downsampleMaskNearest(maskBuf512, maskBuf64);

    auto totalStart = std::chrono::high_resolution_clock::now();

    // 1. VAE Encode Masked Image
    auto vaeEncStart = std::chrono::high_resolution_clock::now();
    std::vector<uint16_t> ufixMaskedImageInput(imagePixelCount);
    for (size_t i = 0; i < maskPixelCount; ++i) {
        float m = (maskBuf512[i] > 0.5f) ? 1.0f : 0.0f;
        for (int c = 0; c < 3; ++c) {
            size_t idx = i * 3 + c;
            float normPixel = (imageBuf[idx] * 2.0f) - 1.0f;
            float maskedPixel = (m > 0.5f) ? 0.0f : normPixel;
            ufixMaskedImageInput[idx] = quantizeToUFix16(maskedPixel, VAE_ENC_IN_SCALE, VAE_ENC_IN_ZP);
        }
    }

    std::vector<float> masked_latents_nchw(latent4Size);
    {
        std::vector<uint16_t> ufixMaskedLatentRaw(latent4Size);
        if (!vaeEncoder.execute({ufixMaskedImageInput.data()}, {ufixMaskedImageInput.size() * sizeof(uint16_t)},
                                {ufixMaskedLatentRaw.data()}, {latent4Size * sizeof(uint16_t)})) {
            std::cerr << "VAE Encoder failed execution." << std::endl;
            return -1;
        }
        for (size_t p = 0; p < HW; ++p) {
            for (size_t c = 0; c < 4; ++c) {
                float deq = dequantizeFromUFix16(ufixMaskedLatentRaw[p * 4 + c], VAE_ENC_OUT_SCALE, VAE_ENC_OUT_ZP);
                masked_latents_nchw[c * HW + p] = applyVaeEncScale(deq);
            }
        }
    }
    auto vaeEncEnd = std::chrono::high_resolution_clock::now();
    std::cout << "[Profile] VAE Encoder: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(vaeEncEnd - vaeEncStart).count() << " ms" << std::endl;

    // 2. Tokenize & Text Encode
    auto teStart = std::chrono::high_resolution_clock::now();
    CLIPTokenizer tokenizer;
    if (!tokenizer.load("vocab.json", "merges.txt")) {
        std::cerr << "Failed loading tokenizer vocab.json / merges.txt" << std::endl;
        return -1;
    }

    std::string userPrompt = (argc > 1) ? argv[1] : "photograph of a beautiful empty scene, highest quality settings";
    std::string negPrompt  = "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured";

    std::vector<int32_t> promptTokens = tokenizer.encode(userPrompt);
    std::vector<int32_t> uncondTokens = tokenizer.encode(negPrompt);

    std::vector<uint16_t> condTextEmbRaw(textEmbElements), uncondTextEmbRaw(textEmbElements);
    textEncoder.execute({promptTokens.data()}, {promptTokens.size() * sizeof(int32_t)},
                        {condTextEmbRaw.data()}, {condTextEmbRaw.size() * sizeof(uint16_t)});
    textEncoder.execute({uncondTokens.data()}, {uncondTokens.size() * sizeof(int32_t)},
                        {uncondTextEmbRaw.data()}, {uncondTextEmbRaw.size() * sizeof(uint16_t)});

    std::vector<float> unetCondTextEmb(textEmbElements), unetUncondTextEmb(textEmbElements);
    for (size_t i = 0; i < textEmbElements; ++i) {
        unetCondTextEmb[i]   = dequantizeFromUFix16(condTextEmbRaw[i], TE_OUT_SCALE, TE_OUT_ZP);
        unetUncondTextEmb[i] = dequantizeFromUFix16(uncondTextEmbRaw[i], TE_OUT_SCALE, TE_OUT_ZP);
    }
    auto teEnd = std::chrono::high_resolution_clock::now();
    std::cout << "[Profile] Text Encoder: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(teEnd - teStart).count() << " ms" << std::endl;

    // 3. Build Schedule & Sample Initial Gaussian Latents
    DPMSolverSchedule schedule(numSteps);

    std::vector<float> latents(latent4Size);
    std::mt19937 rng(42);
    std::normal_distribution<float> gaussian(0.0f, 1.0f);
    for (size_t i = 0; i < latent4Size; ++i) {
        latents[i] = gaussian(rng) * schedule.init_noise_sigma;
    }

    std::vector<float> sampleInput(sample16Size, 0.0f);
    std::vector<float> noisePredCond(latent4Size), noisePredUncond(latent4Size);
    std::vector<float> prev_d(latent4Size, 0.0f), curr_d(latent4Size, 0.0f);

    auto denoiseStart = std::chrono::high_resolution_clock::now();

    // 4. DPM-Solver++ (2M) Inpainting Denoising Loop
    for (int step = 0; step < numSteps; ++step) {
        auto stepStart = std::chrono::high_resolution_clock::now();
        float timestepVal = schedule.timesteps[step];

        // Pack 16-channel sample: [latents (4), mask (1), masked_latents (4), padding (7)]
        for (size_t p = 0; p < HW; ++p) {
            for (size_t c = 0; c < 4; ++c) {
                sampleInput[c * HW + p] = latents[c * HW + p];
            }
            sampleInput[4 * HW + p] = maskBuf64[p];
            for (size_t c = 0; c < 4; ++c) {
                sampleInput[(5 + c) * HW + p] = masked_latents_nchw[c * HW + p];
            }
            for (size_t c = 9; c < 16; ++c) {
                sampleInput[c * HW + p] = 0.0f;
            }
        }

        // Conditional & Unconditional UNet executions
        unet.execute({&timestepVal, sampleInput.data(), unetCondTextEmb.data()},
                     {sizeof(float), sample16Size * sizeof(float), textEmbElements * sizeof(float)},
                     {noisePredCond.data()}, {latent4Size * sizeof(float)});

        unet.execute({&timestepVal, sampleInput.data(), unetUncondTextEmb.data()},
                     {sizeof(float), sample16Size * sizeof(float), textEmbElements * sizeof(float)},
                     {noisePredUncond.data()}, {latent4Size * sizeof(float)});

        float alpha_curr = schedule.alpha_t[step];
        float sigma_curr = schedule.sigma_t[step];

        // Predict x0 (data prediction)
        for (size_t i = 0; i < latent4Size; ++i) {
            float noise_pred = noisePredUncond[i] + guidanceScale * (noisePredCond[i] - noisePredUncond[i]);
            curr_d[i] = (latents[i] - sigma_curr * noise_pred) / alpha_curr;
        }

        int step_next = step + 1;
        float sigma_t_curr = schedule.sigma_t[step];
        float sigma_t_next = schedule.sigma_t[step_next];
        float alpha_next   = schedule.alpha_t[step_next];

        // Solver Update Rule
        if (step == numSteps - 1 || sigma_t_next <= 1e-4f) {
            // Diffusers lower_order_final: directly project to final predicted x0
            for (size_t i = 0; i < latent4Size; ++i) {
                latents[i] = curr_d[i];
            }
        } else if (step == 0) {
            // First-order update for step 0
            float lambda_curr = schedule.lambda_t[step];
            float lambda_next = schedule.lambda_t[step_next];
            float h = lambda_next - lambda_curr;
            float factor1 = sigma_t_next / sigma_t_curr;
            float factor2 = alpha_next * std::expm1(-h);

            for (size_t i = 0; i < latent4Size; ++i) {
                latents[i] = factor1 * latents[i] - factor2 * curr_d[i];
            }
        } else {
            // Second-order multistep update for intermediate steps
            float lambda_prev = schedule.lambda_t[step - 1];
            float lambda_curr = schedule.lambda_t[step];
            float lambda_next = schedule.lambda_t[step_next];
            float h_0 = lambda_curr - lambda_prev;
            float h   = lambda_next - lambda_curr;
            float r0  = h_0 / h;

            float factor1 = sigma_t_next / sigma_t_curr;
            float factor2 = alpha_next * std::expm1(-h);

            for (size_t i = 0; i < latent4Size; ++i) {
                float d_hat = (1.0f + 0.5f / r0) * curr_d[i] - (0.5f / r0) * prev_d[i];
                latents[i] = factor1 * latents[i] - factor2 * d_hat;
            }
        }

        prev_d = curr_d;
        auto stepEnd = std::chrono::high_resolution_clock::now();
        std::cout << "[Step " << (step + 1) << "/" << numSteps << "] Timestep: " << timestepVal
                  << " | Latency: " << std::chrono::duration_cast<std::chrono::milliseconds>(stepEnd - stepStart).count()
                  << " ms" << std::endl;
    }

    auto denoiseEnd = std::chrono::high_resolution_clock::now();
    std::cout << "[Profile] Total Denoising Loop: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(denoiseEnd - denoiseStart).count() << " ms" << std::endl;

    // 5. VAE Decode & Full-Res Compositing
    auto vaeDecStart = std::chrono::high_resolution_clock::now();
    std::vector<uint16_t> ufixLatentDecInput(latent4Size);
    std::vector<uint16_t> ufixRgbOutput(imagePixelCount);

    for (size_t p = 0; p < HW; ++p) {
        for (size_t c = 0; c < 4; ++c) {
            float val = latents[c * HW + p];
            ufixLatentDecInput[p * 4 + c] = quantizeToUFix16(applyVaeDecScale(val), VAE_LATENT_IN_SCALE, VAE_LATENT_IN_ZP);
        }
    }

    if (!vaeDecoder.execute({ufixLatentDecInput.data()}, {latent4Size * sizeof(uint16_t)},
                            {ufixRgbOutput.data()}, {imagePixelCount * sizeof(uint16_t)})) {
        std::cerr << "VAE Decoder execution failed." << std::endl;
        return -1;
    }

    std::vector<uint8_t> finalRgb888(imagePixelCount);
    for (size_t i = 0; i < maskPixelCount; ++i) {
        float m = (maskBuf512[i] > 0.5f) ? 1.0f : 0.0f;
        for (int c = 0; c < 3; ++c) {
            size_t idx = i * 3 + c;
            float decodedVal = dequantizeFromUFix16(ufixRgbOutput[idx], VAE_IMAGE_OUT_SCALE, VAE_IMAGE_OUT_ZP) * 255.0f;
            float originalVal = imageBuf[idx] * 255.0f;
            finalRgb888[idx] = static_cast<uint8_t>(std::clamp(m * decodedVal + (1.0f - m) * originalVal, 0.0f, 255.0f));
        }
    }
    auto vaeDecEnd = std::chrono::high_resolution_clock::now();
    std::cout << "[Profile] VAE Decoder & Composite: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(vaeDecEnd - vaeDecStart).count() << " ms" << std::endl;

    stbi_write_png(outputPath.c_str(), 512, 512, 3, finalRgb888.data(), 512 * 3);
    auto totalEnd = std::chrono::high_resolution_clock::now();

    std::cout << "==========================================================" << std::endl;
    std::cout << "Output successfully saved to: " << outputPath << std::endl;
    std::cout << "End-to-End Pipeline Latency: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd - totalStart).count() << " ms" << std::endl;
    std::cout << "==========================================================" << std::endl;

    return 0;
}