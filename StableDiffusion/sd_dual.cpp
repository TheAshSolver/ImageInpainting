// sd_inpaint.cpp
// ---------------------------------------------------------------------------------------------
// Production Stable-Diffusion-1.5 inpainting runner for Hexagon HTP v79 (Snapdragon 8 Elite / QIDK).
//
// Pipeline : CLIP text encoder (NPU) -> VAE encoder (NPU) -> DPM-Solver++(2M, Karras) denoising with
//            classifier-free guidance -> TAESD/VAE decoder (NPU) -> Adreno Poisson seam blend (GPU).
// Speed    : dual-graph feature caching (full U-Net every N steps, shallow graph + Taylor forecast in
//            between), zero-copy dma-buf buffers, HTP burst power configuration, fused NEON solver step,
//            models stay loaded in --serve mode, and the VAE encoder is skipped while the image/mask
//            are unchanged (about 300 ms per request on the benchmarked device).
// Prompts  : --prompt / --negative, --steps, --cfg (cfg <= 1 drops the negative branch: ~2x faster).
//
// Build (Android NDK r27+, aarch64, QAIRT 2.50 headers):
//   $NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++20 -O3
//       -march=armv8.2-a+fp16+dotprod -I$QNN_SDK_ROOT/include/QNN -I. sd_inpaint.cpp -ldl -llog -o sd_inpaint
//   Needs clip_tokenizer.hpp and stb_image_write.h next to this file. stb_image.h is optional: with it
//   the image and mask can be PNG/JPG; without it only .raw inputs are accepted.
//   optional: -DSD_USE_AHB (AHardwareBuffer allocator, add -landroid), -DSD_QNN_MEM_DMABUF, -DUNET_BATCH=2
//
// Run (device shell, same LD_LIBRARY_PATH / ADSP_LIBRARY_PATH as for the benchmark runner):
//   ./sd_inpaint -i photo.png -m mask.png -p "a red brick wall" -n "blurry, text" -s 12 -o out.png
//   ./sd_inpaint --serve -m mask.raw -i image.raw        # then feed tab-separated requests on stdin
//   ./sd_inpaint --help
//
// Models/binaries are those produced by dual_dlc_export_aimet.py + submit_dual_dlc_to_cloud.py and
// the text encoder / VAE binaries of the benchmark runner. The core pipeline (QNN wrapper, kernels,
// zero-copy pool, GPU blender) is the verified code of sd_runner_dual_dlc_zerocopy.cpp.
// ---------------------------------------------------------------------------------------------
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <algorithm>
#include <cmath>
#include <random>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <type_traits>
#include <utility>
#include <dlfcn.h>
#include <cerrno>
#include <sched.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/eventfd.h>
#include <linux/types.h>
#include <csignal>
#include <cctype>
#include <cstdio>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#if defined(SD_USE_AHB)
#include <android/hardware_buffer.h>
#endif

#include "QnnInterface.h"
#include "QnnCommon.h"
#include "QnnTypes.h"
#include "QnnContext.h"
#include "QnnBackend.h"
#include "QnnDevice.h"
#include "QnnGraph.h"
#include "QnnTensor.h"
#include "QnnMem.h"
#include "QnnProfile.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpPerfInfrastructure.h"
#include "System/QnnSystemContext.h"
#include "System/QnnSystemInterface.h"
#include "clip_tokenizer.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#if defined(__has_include)
#  if __has_include("stb_image.h")
#    define SD_HAVE_STB_IMAGE 1
#    define STB_IMAGE_IMPLEMENTATION
#    include "stb_image.h"
#  endif
#endif

#ifndef UNET_BATCH
#define UNET_BATCH 1
#endif

using Clock = std::chrono::steady_clock;
static inline double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
static bool g_verbose = false;                      // set by -v / --verbose
#define STAGE(msg) do { if (g_verbose) std::cout << "[Stage] " << msg << std::endl; } while (0)
#define CHECK_QNN_STATUS(status, msg) \
    do { Qnn_ErrorHandle_t _s = (status); if (_s != QNN_SUCCESS) { \
        std::cerr << "[QNN ERROR] " << msg << " | Error Code: " << _s << std::endl; return false; } } while (0)

#define VAE_018215_BAKED_INTO_QUANT_SCALE 1
inline float applyVaeEncScale(float v) { return VAE_018215_BAKED_INTO_QUANT_SCALE ? v : v * 0.18215f; }
inline float applyVaeDecScale(float v) { return VAE_018215_BAKED_INTO_QUANT_SCALE ? v : v * (1.0f / 0.18215f); }

inline uint16_t quantizeToUFix16(float val, float scale, int32_t zp) {
    if (scale <= 0.0f) scale = 1.0f;
    float q = std::round(val / scale) + static_cast<float>(zp);
    return static_cast<uint16_t>(std::clamp(q, 0.0f, 65535.0f));
}
inline float dequantizeFromUFix16(uint16_t v, float scale, int32_t zp) {
    return (static_cast<float>(v) - static_cast<float>(zp)) * scale;
}

static void pinToCore(int cpu) {
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0)
        std::cerr << "[Warn] sched_setaffinity failed: " << std::strerror(errno) << std::endl;
}
static bool envOn(const char* n) { const char* v = std::getenv(n); return v && *v && std::strcmp(v, "0") != 0; }
static std::string envStr(const char* n, const char* d) { const char* v = std::getenv(n); return (v && *v) ? v : d; }

// =====================================================================
// Memory-Mapped File RAII Wrapper (Zero-Copy Model Loading)
// =====================================================================
struct MappedFile {
    void* addr = MAP_FAILED;
    size_t size = 0;
    int fd = -1;

    bool map(const std::string& path) {
        fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            std::cerr << "[MMap] Cannot open '" << path << "': " << std::strerror(errno) << std::endl;
            return false;
        }
        struct stat st{};
        if (fstat(fd, &st) != 0 || st.st_size <= 0) {
            std::cerr << "[MMap] Cannot stat or empty file '" << path << "': " << std::strerror(errno) << std::endl;
            close(fd);
            fd = -1;
            return false;
        }
        size = static_cast<size_t>(st.st_size);
        addr = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        if (addr == MAP_FAILED) {
            std::cerr << "[MMap] Failed mmap on '" << path << "' (" << size << " bytes): " << std::strerror(errno) << std::endl;
            close(fd);
            fd = -1;
            return false;
        }
        // Prefetch pages into cache immediately to minimize page-fault latency during deserialization
        madvise(addr, size, MADV_WILLNEED);
        return true;
    }

    void unmap() {
        if (addr != MAP_FAILED) {
            munmap(addr, size);
            addr = MAP_FAILED;
        }
        if (fd >= 0) {
            close(fd);
            fd = -1;
        }
        size = 0;
    }

    ~MappedFile() {
        unmap();
    }
};

// =====================================================================
// rpcmem shared buffers
// =====================================================================
struct SharedBuf {
    void* ptr = nullptr; int fd = -1; size_t bytes = 0;
    float* f32() const { return static_cast<float*>(ptr); }
};

class RpcMem {
public:
    bool load() {
        lib = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
        if (!lib) { std::cerr << "[rpcmem] dlopen libcdsprpc.so failed: " << dlerror() << std::endl; return false; }
        pAlloc = (AllocFn)dlsym(lib, "rpcmem_alloc");
        pFree  = (FreeFn)dlsym(lib, "rpcmem_free");
        pFd    = (FdFn)dlsym(lib, "rpcmem_to_fd");
        auto init = (void (*)())dlsym(lib, "rpcmem_init");
        if (init) init();
        deinit = (void (*)())dlsym(lib, "rpcmem_deinit");
        if (!pAlloc || !pFree || !pFd) { std::cerr << "[rpcmem] missing symbols" << std::endl; return false; }
        return true;
    }
    bool alloc(SharedBuf& b, size_t bytes) {
        constexpr int RPCMEM_DEFAULT_HEAP = -1;
        constexpr uint32_t RPCMEM_DEFAULT_FLAGS = 1;
        b.ptr = pAlloc(RPCMEM_DEFAULT_HEAP, RPCMEM_DEFAULT_FLAGS, static_cast<int>(bytes));
        if (!b.ptr) {
            constexpr int RPCMEM_HEAP_ID_SYSTEM = 25;
            b.ptr = pAlloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, static_cast<int>(bytes));
        }
        if (!b.ptr) { std::cerr << "[rpcmem] alloc of " << bytes << " bytes FAILED" << std::endl; return false; }
        b.fd = pFd(b.ptr); b.bytes = bytes;
        std::memset(b.ptr, 0, bytes);
        bufs.push_back(b.ptr);
        if (b.fd < 0) { std::cerr << "[rpcmem] rpcmem_to_fd returned " << b.fd << std::endl; return false; }
        return true;
    }
    ~RpcMem() {
        for (void* p : bufs) if (pFree) pFree(p);
        if (deinit) deinit();
        if (lib) dlclose(lib);
    }
private:
    typedef void* (*AllocFn)(int, uint32_t, int);
    typedef void  (*FreeFn)(void*);
    typedef int   (*FdFn)(void*);
    void* lib = nullptr; AllocFn pAlloc = nullptr; FreeFn pFree = nullptr; FdFn pFd = nullptr;
    void (*deinit)() = nullptr;
    std::vector<void*> bufs;
};

// =====================================================================
// Tensor storage + metadata extraction from context binary
// =====================================================================
struct TensorCopy {
    std::string name;
    std::vector<uint32_t> dims;
    Qnn_Tensor_t t = QNN_TENSOR_INIT;
};
using TensorPtr = std::shared_ptr<TensorCopy>;

static bool isQuantType(Qnn_DataType_t t) {
    return t == QNN_DATATYPE_UFIXED_POINT_16 || t == QNN_DATATYPE_UFIXED_POINT_8 ||
           t == QNN_DATATYPE_SFIXED_POINT_16 || t == QNN_DATATYPE_SFIXED_POINT_8;
}

static TensorPtr makeTensor(uint32_t id, const std::string& name, Qnn_TensorType_t type, Qnn_DataType_t dt,
                            const std::vector<uint32_t>& dims, bool quantDefined, float scale, int32_t offset) {
    auto c = std::make_shared<TensorCopy>();
    c->name = name; c->dims = dims;
    c->t = QNN_TENSOR_INIT;
    c->t.version = QNN_TENSOR_VERSION_1;
    c->t.v1.id = id;
    c->t.v1.name = c->name.c_str();
    c->t.v1.type = type;
    c->t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    c->t.v1.dataType = dt;
    if (quantDefined) {
        c->t.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
        c->t.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
        c->t.v1.quantizeParams.scaleOffsetEncoding.scale = scale;
        c->t.v1.quantizeParams.scaleOffsetEncoding.offset = offset;
    } else {
        c->t.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
    }
    c->t.v1.rank = static_cast<uint32_t>(c->dims.size());
    c->t.v1.dimensions = c->dims.data();
    c->t.v1.memType = QNN_TENSORMEMTYPE_RAW;
    return c;
}

static TensorPtr copyFromMeta(const Qnn_Tensor_t& s, bool isInput) {
    uint32_t id, rank; const char* nm; const uint32_t* dims; Qnn_DataType_t dt; Qnn_QuantizeParams_t q;
    if (s.version == QNN_TENSOR_VERSION_2) {
        id = s.v2.id; nm = s.v2.name; rank = s.v2.rank; dims = s.v2.dimensions; dt = s.v2.dataType; q = s.v2.quantizeParams;
    } else {
        id = s.v1.id; nm = s.v1.name; rank = s.v1.rank; dims = s.v1.dimensions; dt = s.v1.dataType; q = s.v1.quantizeParams;
    }
    std::vector<uint32_t> d(dims, dims + rank);
    bool qd = (q.encodingDefinition == QNN_DEFINITION_DEFINED &&
               q.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET && isQuantType(dt));
    return makeTensor(id, nm ? nm : "", isInput ? QNN_TENSOR_TYPE_APP_WRITE : QNN_TENSOR_TYPE_APP_READ, dt, d,
                      qd, qd ? q.scaleOffsetEncoding.scale : 1.0f, qd ? q.scaleOffsetEncoding.offset : 0);
}

struct GraphMeta {
    std::string name;
    std::vector<TensorPtr> in, out;
};

template <typename G>
static void takeGraph(const G& g, std::vector<GraphMeta>& metas) {
    GraphMeta m;
    m.name = g.graphName ? g.graphName : "";
    for (uint32_t i = 0; i < g.numGraphInputs; ++i)  m.in.push_back(copyFromMeta(g.graphInputs[i], true));
    for (uint32_t i = 0; i < g.numGraphOutputs; ++i) m.out.push_back(copyFromMeta(g.graphOutputs[i], false));
    metas.push_back(std::move(m));
}

static void takeGraphs(uint32_t n, const QnnSystemContext_GraphInfo_t* graphs, std::vector<GraphMeta>& metas) {
    for (uint32_t g = 0; g < n; ++g) {
        const auto& c = graphs[g];
        if (c.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3)      takeGraph(c.graphInfoV3, metas);
        else if (c.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) takeGraph(c.graphInfoV2, metas);
        else if (c.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) takeGraph(c.graphInfoV1, metas);
    }
}

// =====================================================================
// Shared QNN runtime
// =====================================================================
class QnnRuntime {
    using SysIface = std::remove_cv_t<std::remove_reference_t<
        decltype(std::declval<QnnSystemInterface_t>().QNN_SYSTEM_INTERFACE_VER_NAME)>>;
public:
    void* lib = nullptr;
    QNN_INTERFACE_VER_TYPE qnn{};
    Qnn_BackendHandle_t backend = nullptr;
    Qnn_DeviceHandle_t  device  = nullptr;
    QnnHtpDevice_PerfInfrastructure_t* perf = nullptr;
    uint32_t powerCfgId = 0;
    bool perfOn = false;

    void* sysLib = nullptr;
    SysIface sys{};
    bool sysOk = false;

    bool init(const std::string& backendPath = "libQnnHtp.so") {
        lib = dlopen(backendPath.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!lib) { std::cerr << "[Init] dlopen(" << backendPath << ") failed: " << dlerror() << std::endl; return false; }
        typedef Qnn_ErrorHandle_t (*GetProvidersFn)(const QnnInterface_t***, uint32_t*);
        auto getProviders = (GetProvidersFn)dlsym(lib, "QnnInterface_getProviders");
        if (!getProviders) { std::cerr << "[Init] QnnInterface_getProviders missing" << std::endl; return false; }
        const QnnInterface_t** providers = nullptr; uint32_t n = 0;
        CHECK_QNN_STATUS(getProviders(&providers, &n), "get providers");
        if (n == 0 || !providers) return false;
        qnn = providers[0]->QNN_INTERFACE_VER_NAME;
        CHECK_QNN_STATUS(qnn.backendCreate(nullptr, nullptr, &backend), "backendCreate");
        if (qnn.deviceCreate) CHECK_QNN_STATUS(qnn.deviceCreate(nullptr, nullptr, &device), "deviceCreate");

        sysLib = dlopen("libQnnSystem.so", RTLD_NOW | RTLD_LOCAL);
        if (sysLib) {
            typedef Qnn_ErrorHandle_t (*SysGetFn)(const QnnSystemInterface_t***, uint32_t*);
            auto g = (SysGetFn)dlsym(sysLib, "QnnSystemInterface_getProviders");
            const QnnSystemInterface_t** sp = nullptr; uint32_t sn = 0;
            if (g && g(&sp, &sn) == QNN_SUCCESS && sn > 0 && sp) { sys = sp[0]->QNN_SYSTEM_INTERFACE_VER_NAME; sysOk = true; }
        }
        if (!sysOk) std::cerr << "[Warn] libQnnSystem.so unavailable: falling back to hard-coded tensor ids" << std::endl;
        return true;
    }

    bool readMeta(const void* binData, size_t binSize, std::vector<GraphMeta>& metas) {
        if (!sysOk || !binData || binSize == 0) return false;
        QnnSystemContext_Handle_t h = nullptr;
        if (sys.systemContextCreate(&h) != QNN_SUCCESS || !h) return false;
        const QnnSystemContext_BinaryInfo_t* bi = nullptr; Qnn_ContextBinarySize_t bsz = 0;
        bool ok = false;
        if (sys.systemContextGetBinaryInfo(h, const_cast<void*>(binData), binSize, &bi, &bsz) == QNN_SUCCESS && bi) {
            if (bi->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3)
                takeGraphs(bi->contextBinaryInfoV3.numGraphs, bi->contextBinaryInfoV3.graphs, metas);
            else if (bi->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2)
                takeGraphs(bi->contextBinaryInfoV2.numGraphs, bi->contextBinaryInfoV2.graphs, metas);
            else if (bi->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1)
                takeGraphs(bi->contextBinaryInfoV1.numGraphs, bi->contextBinaryInfoV1.graphs, metas);
            ok = !metas.empty();
        }
        sys.systemContextFree(h);
        return ok;
    }

    bool setupPerf() {
        QnnDevice_Infrastructure_t infra = nullptr;
        if (!qnn.deviceGetInfrastructure || qnn.deviceGetInfrastructure(&infra) != QNN_SUCCESS || !infra) {
            std::cerr << "[Perf] deviceGetInfrastructure unavailable" << std::endl; return false;
        }
        auto* htp = reinterpret_cast<QnnHtpDevice_Infrastructure_t*>(infra);
        perf = &htp->perfInfra;
        if (!perf->createPowerConfigId || !perf->setPowerConfig) { std::cerr << "[Perf] perf infra incomplete" << std::endl; perf = nullptr; return false; }
        CHECK_QNN_STATUS(perf->createPowerConfigId(0, 0, &powerCfgId), "createPowerConfigId");

        QnnHtpPerfInfrastructure_PowerConfig_t dcvs{};
        dcvs.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
        dcvs.dcvsV3Config.contextId        = powerCfgId;
        dcvs.dcvsV3Config.setDcvsEnable    = 1;
        dcvs.dcvsV3Config.dcvsEnable       = 0;
        dcvs.dcvsV3Config.powerMode        = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
        dcvs.dcvsV3Config.setSleepLatency  = 1;
        dcvs.dcvsV3Config.sleepLatency     = 40;
        dcvs.dcvsV3Config.setSleepDisable  = 1;
        dcvs.dcvsV3Config.sleepDisable     = 1;
        dcvs.dcvsV3Config.setBusParams     = 1;
        dcvs.dcvsV3Config.busVoltageCornerMin    = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        dcvs.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        dcvs.dcvsV3Config.busVoltageCornerMax    = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        dcvs.dcvsV3Config.setCoreParams    = 1;
        dcvs.dcvsV3Config.coreVoltageCornerMin    = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        dcvs.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        dcvs.dcvsV3Config.coreVoltageCornerMax    = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;

        QnnHtpPerfInfrastructure_PowerConfig_t hmx{};
        hmx.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_HMX_V2;
        hmx.hmxV2Config.hmxPickDefault         = 0;
        hmx.hmxV2Config.hmxVoltageCornerMin    = DCVS_EXP_VCORNER_MAX;
        hmx.hmxV2Config.hmxVoltageCornerTarget = DCVS_EXP_VCORNER_MAX;
        hmx.hmxV2Config.hmxVoltageCornerMax    = DCVS_EXP_VCORNER_MAX;
        hmx.hmxV2Config.hmxPerfMode            = QNN_HTP_PERF_INFRASTRUCTURE_CLK_PERF_HIGH;

        QnnHtpPerfInfrastructure_PowerConfig_t rpcLat{};
        rpcLat.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
        rpcLat.rpcControlLatencyConfig = 100;
        QnnHtpPerfInfrastructure_PowerConfig_t rpcPoll{};
        rpcPoll.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
        rpcPoll.rpcPollingTimeConfig = 9999;

        const QnnHtpPerfInfrastructure_PowerConfig_t* cfgs[] = {&dcvs, &hmx, &rpcLat, &rpcPoll, nullptr};
        CHECK_QNN_STATUS(perf->setPowerConfig(powerCfgId, cfgs), "setPowerConfig");
        perfOn = true;
        if (g_verbose) std::cout << "[Perf] HTP burst / DCVS-off / HMX-max / RPC polling enabled" << std::endl;
        return true;
    }

    ~QnnRuntime() {
        if (perfOn && perf && perf->destroyPowerConfigId) perf->destroyPowerConfigId(powerCfgId);
        if (device  && qnn.deviceFree)  qnn.deviceFree(device);
        if (backend && qnn.backendFree) qnn.backendFree(backend);
        if (lib) dlclose(lib);
        if (sysLib) dlclose(sysLib);
    }
};

struct IoRef {
    Qnn_MemHandle_t h = nullptr;
    void* ptr = nullptr;
    size_t bytes = 0;
};

// =====================================================================
// QNN graph wrapper
// =====================================================================
class QnnModelEngine {
    QnnRuntime* m_rt = nullptr;
    Qnn_ContextHandle_t m_ctx = nullptr;
    Qnn_GraphHandle_t   m_graph = nullptr;
    Qnn_ProfileHandle_t m_prof = nullptr;
    std::vector<TensorPtr> m_inKeep, m_outKeep;
    std::vector<Qnn_Tensor_t> m_in, m_out;
    std::vector<Qnn_MemHandle_t> m_handles;
    std::string m_tag;
    int m_regFail = 0;

    void rebuild() {
        m_in.clear(); m_out.clear();
        for (auto& p : m_inKeep)  m_in.push_back(p->t);
        for (auto& p : m_outKeep) m_out.push_back(p->t);
    }
    static uint64_t elems(const Qnn_Tensor_t& t) {
        uint64_t n = 1; for (uint32_t i = 0; i < t.v1.rank; ++i) n *= t.v1.dimensions[i]; return n;
    }
    void dumpIo() const {
        auto pr = [&](const char* k, size_t i, const Qnn_Tensor_t& t) {
            std::cout << "   " << k << "[" << i << "] id=" << t.v1.id << " name=\"" << (t.v1.name ? t.v1.name : "") << "\" dtype=0x"
                      << std::hex << t.v1.dataType << std::dec << " shape=[";
            for (uint32_t d = 0; d < t.v1.rank; ++d) std::cout << t.v1.dimensions[d] << (d + 1 < t.v1.rank ? "," : "");
            std::cout << "]";
            if (t.v1.quantizeParams.encodingDefinition == QNN_DEFINITION_DEFINED)
                std::cout << " scale=" << t.v1.quantizeParams.scaleOffsetEncoding.scale
                          << " offset=" << t.v1.quantizeParams.scaleOffsetEncoding.offset;
            std::cout << std::endl;
        };
        for (size_t i = 0; i < m_in.size(); ++i)  pr("in ", i, m_in[i]);
        for (size_t i = 0; i < m_out.size(); ++i) pr("out", i, m_out[i]);
    }

public:
    ~QnnModelEngine() {
        if (m_rt) {
            if (!m_handles.empty() && m_rt->qnn.memDeRegister)
                m_rt->qnn.memDeRegister(m_handles.data(), static_cast<uint32_t>(m_handles.size()));
            if (m_prof && m_rt->qnn.profileFree) m_rt->qnn.profileFree(m_prof);
            if (m_ctx && m_rt->qnn.contextFree) m_rt->qnn.contextFree(m_ctx, nullptr);
        }
    }

    void addInputTensor(uint32_t id, const char* name, Qnn_DataType_t dt, std::vector<uint32_t> dims,
                        float scale = 1.0f, int32_t offset = 0) {
        m_inKeep.push_back(makeTensor(id, name, QNN_TENSOR_TYPE_APP_WRITE, dt, dims, isQuantType(dt), scale, offset));
    }
    void addOutputTensor(uint32_t id, const char* name, Qnn_DataType_t dt, std::vector<uint32_t> dims,
                        float scale = 1.0f, int32_t offset = 0) {
        m_outKeep.push_back(makeTensor(id, name, QNN_TENSOR_TYPE_APP_READ, dt, dims, isQuantType(dt), scale, offset));
    }

    // Zero-Copy initialization using mmap()
    bool init(QnnRuntime& rt, const std::string& tag, const std::string& binPath, const std::string& graphHint) {
        m_rt = &rt; m_tag = tag;
        MappedFile mappedBin;
        if (!mappedBin.map(binPath)) {
            return false;
        }
        if (g_verbose) std::cout << "[Init] " << tag << ": " << binPath << " (" << mappedBin.size << " bytes, zero-copy mmap)" << std::endl;

        std::string graphName = graphHint;
        std::vector<GraphMeta> metas;
        if (rt.readMeta(mappedBin.addr, mappedBin.size, metas)) {
            const GraphMeta* pick = nullptr;
            for (auto& m : metas) if (m.name == graphHint) pick = &m;
            if (!pick) {
                pick = &metas[0];
                std::cerr << "[Warn] " << tag << ": graph '" << graphHint << "' not in binary; using '" << pick->name
                          << "' (" << metas.size() << " graph(s) present)" << std::endl;
            }
            graphName = pick->name;
            m_inKeep = pick->in; m_outKeep = pick->out;
            if (g_verbose) std::cout << "[Meta] " << tag << " graph '" << graphName << "' (from binary):" << std::endl;
        } else {
            if (g_verbose) std::cout << "[Meta] " << tag << " using hard-coded tensor defs for graph '" << graphName << "':" << std::endl;
        }
        rebuild();
        if (g_verbose) dumpIo();

        CHECK_QNN_STATUS(rt.qnn.contextCreateFromBinary(rt.backend, rt.device, nullptr, mappedBin.addr, mappedBin.size, &m_ctx, nullptr),
                         tag + " contextCreateFromBinary");
        CHECK_QNN_STATUS(rt.qnn.graphRetrieve(m_ctx, graphName.c_str(), &m_graph), tag + " graphRetrieve('" + graphName + "')");
        return true;
    }

    size_t numIn() const  { return m_in.size(); }
    size_t numOut() const { return m_out.size(); }
    uint64_t inElems(size_t i) const  { return elems(m_in[i]); }
    uint64_t outElems(size_t i) const { return elems(m_out[i]); }
    Qnn_DataType_t inType(size_t i) const  { return m_in[i].v1.dataType; }
    Qnn_DataType_t outType(size_t i) const { return m_out[i].v1.dataType; }
    // QNN convention: real = (q + offset) * scale
    bool inQuant(size_t i, float& sc, int32_t& off) const { return quantOf(m_in[i], sc, off); }
    bool outQuant(size_t i, float& sc, int32_t& off) const { return quantOf(m_out[i], sc, off); }
    int findIn(const std::string& n) const  { for (size_t i = 0; i < m_in.size(); ++i)  if (m_in[i].v1.name  && n == m_in[i].v1.name)  return (int)i; return -1; }
    int findOut(const std::string& n) const { for (size_t i = 0; i < m_out.size(); ++i) if (m_out[i].v1.name && n == m_out[i].v1.name) return (int)i; return -1; }
    const std::string& tag() const { return m_tag; }
    int regFailures() const { return m_regFail; }
    static bool quantOf(const Qnn_Tensor_t& t, float& sc, int32_t& off) {
        if (t.v1.quantizeParams.encodingDefinition != QNN_DEFINITION_DEFINED) return false;
        sc = t.v1.quantizeParams.scaleOffsetEncoding.scale; off = t.v1.quantizeParams.scaleOffsetEncoding.offset; return true;
    }

    IoRef registerBuf(bool isInput, size_t idx, const SharedBuf& b, bool allowZeroCopy) {
        IoRef r; r.ptr = b.ptr; r.bytes = b.bytes;
        if (!allowZeroCopy || !m_rt->qnn.memRegister || b.fd < 0) return r;
        Qnn_Tensor_t& t = isInput ? m_in[idx] : m_out[idx];

        Qnn_MemDescriptor_t d = QNN_MEM_DESCRIPTOR_INIT;
        d.memShape = {t.v1.rank, t.v1.dimensions, nullptr};
        d.dataType = t.v1.dataType;
#if defined(SD_QNN_MEM_DMABUF)
        d.memType = QNN_MEM_TYPE_DMA_BUF; d.dmaBufInfo.fd = b.fd; d.dmaBufInfo.data = b.ptr;
#else
        d.memType  = QNN_MEM_TYPE_ION;       // dma-buf fds are accepted as ION fds by the HTP backend
        d.ionInfo.fd = b.fd;
#endif

        Qnn_MemHandle_t h = nullptr;
        Qnn_ErrorHandle_t s = m_rt->qnn.memRegister(m_ctx, &d, 1, &h);
        if (s != QNN_SUCCESS || !h) {
            std::cerr << "[Mem] " << m_tag << " memRegister failed (" << (isInput ? "in " : "out ") << idx << ", err " << s
                      << ") -> using RAW pointer fallback" << std::endl; ++m_regFail;
            return r;
        }
        m_handles.push_back(h);
        r.h = h;
        return r;
    }

    void select(bool isInput, size_t idx, const IoRef& r) {
        Qnn_Tensor_t& t = isInput ? m_in[idx] : m_out[idx];
        if (r.h) {
            t.v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
            t.v1.memHandle = r.h;
        } else {
            t.v1.memType = QNN_TENSORMEMTYPE_RAW;
            t.v1.clientBuf.data = r.ptr;
            t.v1.clientBuf.dataSize = static_cast<uint32_t>(r.bytes);
        }
    }

    bool executeBound() {
        Qnn_ErrorHandle_t s = m_rt->qnn.graphExecute(m_graph, m_in.data(), (uint32_t)m_in.size(),
                                                     m_out.data(), (uint32_t)m_out.size(), m_prof, nullptr);
        if (s != QNN_SUCCESS) std::cerr << "[QNN] " << m_tag << " graphExecute failed: " << s << std::endl;
        return s == QNN_SUCCESS;
    }

    bool execute(const std::vector<void*>& ip, const std::vector<size_t>& is,
                 const std::vector<void*>& op, const std::vector<size_t>& os) {
        if (ip.size() != m_in.size() || op.size() != m_out.size()) {
            std::cerr << "[QNN] " << m_tag << " I/O count mismatch: graph has " << m_in.size() << " in / " << m_out.size()
                      << " out, caller gave " << ip.size() << " / " << op.size() << std::endl;
            return false;
        }
        for (size_t i = 0; i < m_in.size(); ++i)  { m_in[i].v1.memType = QNN_TENSORMEMTYPE_RAW;  m_in[i].v1.clientBuf.data = ip[i];  m_in[i].v1.clientBuf.dataSize = (uint32_t)is[i]; }
        for (size_t i = 0; i < m_out.size(); ++i) { m_out[i].v1.memType = QNN_TENSORMEMTYPE_RAW; m_out[i].v1.clientBuf.data = op[i]; m_out[i].v1.clientBuf.dataSize = (uint32_t)os[i]; }
        return executeBound();
    }

    void enableProfiling() {
        if (m_rt->qnn.profileCreate) m_rt->qnn.profileCreate(m_rt->backend, QNN_PROFILE_LEVEL_BASIC, &m_prof);
    }
    void dumpProfile(const char* tag) {
        if (!m_prof || !m_rt->qnn.profileGetEvents) return;
        const QnnProfile_EventId_t* ev = nullptr; uint32_t n = 0;
        if (m_rt->qnn.profileGetEvents(m_prof, &ev, &n) != QNN_SUCCESS) return;
        std::cout << "[QNN Profile] " << tag << std::endl;
        dumpEvents(ev, n, 1);
    }
private:
    void dumpEvents(const QnnProfile_EventId_t* ev, uint32_t n, int depth) {
        for (uint32_t i = 0; i < n; ++i) {
            QnnProfile_EventData_t d{};
            if (m_rt->qnn.profileGetEventData(ev[i], &d) != QNN_SUCCESS) continue;
            std::cout << std::string(depth * 2, ' ') << (d.identifier ? d.identifier : "?")
                      << " type=" << d.type << " unit=" << d.unit << " value=" << d.value << std::endl;
            const QnnProfile_EventId_t* sub = nullptr; uint32_t ns = 0;
            if (depth < 3 && m_rt->qnn.profileGetSubEvents(ev[i], &sub, &ns) == QNN_SUCCESS && ns)
                dumpEvents(sub, ns, depth + 1);
        }
    }
};

// =====================================================================
// TaylorSeer order-1 forecasting kernels + 3-slot cache ring
// (self-contained: unit-tested off-device with the scalar path)
// =====================================================================
// ==== BEGIN KERNELS ====
// Forecast on step axis x:  F(x_k) = F(x_s) + (F(x_s) - F(x_p)) / (x_s - x_p) * (x_k - x_s)
//                                  = a * F_s + b * F_p ,   a = 1 + r, b = -r, r = (x_k-x_s)/(x_s-x_p)
// With x = step index and uniform interval N this is exactly the task formula
// F(t) + (F(t) - F(t+N))/N * (-k) (their t decreases with denoising; our index increases).
struct TaylorCoef { float a = 1.f, b = 0.f; bool useHistory = false; };

static TaylorCoef makeTaylorCoef(int order, bool haveHistory, double xk, double xs, double xp) {
    TaylorCoef c;
    if (order < 1 || !haveHistory) return c;                  // order-0: reuse newest feature
    const double den = xs - xp;
    if (std::fabs(den) < 1e-12) return c;
    double r = (xk - xs) / den;
    r = std::min(std::max(r, 0.0), 1.5);                      // never extrapolate wildly
    c.a = float(1.0 + r); c.b = float(-r); c.useHistory = true;
    return c;
}

static void forecastF32(float* __restrict dst, const float* __restrict t, const float* __restrict p,
                        size_t n, float a, float b) {
    size_t i = 0;
#if defined(__ARM_NEON)
    const float32x4_t va = vdupq_n_f32(a), vb = vdupq_n_f32(b);
    for (; i + 16 <= n; i += 16) {                            // 4 independent FMA chains / iteration
        float32x4_t t0 = vld1q_f32(t + i),      t1 = vld1q_f32(t + i + 4);
        float32x4_t t2 = vld1q_f32(t + i + 8),  t3 = vld1q_f32(t + i + 12);
        float32x4_t p0 = vld1q_f32(p + i),      p1 = vld1q_f32(p + i + 4);
        float32x4_t p2 = vld1q_f32(p + i + 8),  p3 = vld1q_f32(p + i + 12);
        vst1q_f32(dst + i,      vfmaq_f32(vmulq_f32(p0, vb), t0, va));
        vst1q_f32(dst + i + 4,  vfmaq_f32(vmulq_f32(p1, vb), t1, va));
        vst1q_f32(dst + i + 8,  vfmaq_f32(vmulq_f32(p2, vb), t2, va));
        vst1q_f32(dst + i + 12, vfmaq_f32(vmulq_f32(p3, vb), t3, va));
    }
    for (; i + 4 <= n; i += 4)
        vst1q_f32(dst + i, vfmaq_f32(vmulq_f32(vld1q_f32(p + i), vb), vld1q_f32(t + i), va));
#endif
    for (; i < n; ++i) dst[i] = a * t[i] + b * p[i];
}

// 16-bit affine-quantised cache (QNN: real = (q + off) * scale).  Source grid (sF,oF) is the Full
// graph's cache OUTPUT, destination grid (sI,oI) is the Shallow graph's cache INPUT.  The exporter
// forces them equal; the general ratio form keeps the kernel correct if they ever differ.
//   q_out = ca*q_t + cb*q_p + C,  ca = a*sF/sI, cb = b*sF/sI, C = (ca+cb)*oF - oI
struct U16Forecast { float ca, cb, c; };
static U16Forecast makeU16Forecast(const TaylorCoef& k, float sF, int32_t oF, float sI, int32_t oI) {
    const float ratio = sF / sI;
    U16Forecast u; u.ca = k.a * ratio; u.cb = k.b * ratio;
    u.c = (u.ca + u.cb) * float(oF) - float(oI);
    return u;
}

static inline uint16_t satU16(float x) {
    long v = std::lrintf(x);
    return uint16_t(v < 0 ? 0 : (v > 65535 ? 65535 : v));
}

static void forecastU16(uint16_t* __restrict dst, const uint16_t* __restrict t, const uint16_t* __restrict p,
                        size_t n, const U16Forecast& u) {
    size_t i = 0;
#if defined(__ARM_NEON)
    const float32x4_t vca = vdupq_n_f32(u.ca), vcb = vdupq_n_f32(u.cb), vc = vdupq_n_f32(u.c);
    for (; i + 8 <= n; i += 8) {
        uint16x8_t qt = vld1q_u16(t + i), qp = vld1q_u16(p + i);
        float32x4_t tl = vcvtq_f32_u32(vmovl_u16(vget_low_u16(qt))),  th = vcvtq_f32_u32(vmovl_u16(vget_high_u16(qt)));
        float32x4_t pl = vcvtq_f32_u32(vmovl_u16(vget_low_u16(qp))),  ph = vcvtq_f32_u32(vmovl_u16(vget_high_u16(qp)));
        float32x4_t rl = vfmaq_f32(vfmaq_f32(vc, pl, vcb), tl, vca);
        float32x4_t rh = vfmaq_f32(vfmaq_f32(vc, ph, vcb), th, vca);
        // vcvtn: round-to-nearest-even, negatives clamp to 0; vqmovn saturates to 65535
        uint16x4_t ol = vqmovn_u32(vcvtnq_u32_f32(rl)), oh = vqmovn_u32(vcvtnq_u32_f32(rh));
        vst1q_u16(dst + i, vcombine_u16(ol, oh));
    }
#endif
    for (; i < n; ++i) dst[i] = satU16(u.ca * float(t[i]) + u.cb * float(p[i]) + u.c);
}

// Three buffers rotate: {t = newest full feature, prev = full feature before it, pred = scratch}.
// The Full graph always writes into the scratch slot, so no copy is ever needed on commit.
struct CacheRing {
    int t = 0, prev = 1, pred = 2;
    int nHist = 0;                  // number of valid full-step features (0,1,2+)
    double xT = 0, xPrev = 0;       // forecast-axis coordinate of the two newest features
    int fullTarget() const { return pred; }
    void commitFull(double x) {
        const int newT = pred, newPrev = t, newPred = prev;
        t = newT; prev = newPrev; pred = newPred;
        xPrev = xT; xT = x;
        if (nHist < 2) ++nHist;
    }
    bool haveHistory() const { return nHist >= 2; }
};
// ==== END KERNELS ====
// =====================================================================
// DPM-Solver++ (2M) Karras schedule + fused step
// =====================================================================
struct DPMSolverSchedule {
    int numSteps;
    std::vector<float> timesteps, sigmas, alpha_t, sigma_t, lambda_t;
    const float init_noise_sigma = 1.0f;
    DPMSolverSchedule(int steps, int T = 1000) : numSteps(steps) {
        float bs = 0.00085f, be = 0.0120f, s0 = std::sqrt(bs), s1 = std::sqrt(be);
        std::vector<double> ts(T), tls(T);
        double prod = 1.0;
        for (int i = 0; i < T; ++i) {
            float sb = s0 + (s1 - s0) * (float(i) / (T - 1));
            prod *= (1.0 - double(sb) * double(sb));
            ts[i] = std::sqrt((1.0 - prod) / prod);
            tls[i] = std::log(ts[i]);
        }
        double rho = 7.0, mn = std::pow(ts[0], 1.0 / rho), mx = std::pow(ts[T - 1], 1.0 / rho);
        sigmas.resize(numSteps + 1); timesteps.resize(numSteps);
        for (int i = 0; i < numSteps; ++i) {
            double ramp = double(i) / double(numSteps - 1);
            sigmas[i] = static_cast<float>(std::pow(mx + ramp * (mn - mx), rho));
            double tl = std::log(double(sigmas[i]));
            int idx = int(std::lower_bound(tls.begin(), tls.end(), tl) - tls.begin());
            if (idx == 0) timesteps[i] = 0.0f;
            else if (idx >= T) timesteps[i] = float(T - 1);
            else timesteps[i] = float((idx - 1) + (tl - tls[idx - 1]) / (tls[idx] - tls[idx - 1]));
        }
        sigmas[numSteps] = 0.0f;
        alpha_t.resize(numSteps + 1); sigma_t.resize(numSteps + 1); lambda_t.resize(numSteps + 1);
        for (int i = 0; i < numSteps; ++i) {
            float s = sigmas[i];
            alpha_t[i] = 1.0f / std::sqrt(s * s + 1.0f);
            sigma_t[i] = s * alpha_t[i];
            lambda_t[i] = -std::log(s);
        }
        alpha_t[numSteps] = 1.0f; sigma_t[numSteps] = 0.0f; lambda_t[numSteps] = 1e9f;
    }
};

struct StepCoef { float a, b, k0, k1, f1, f2; };

static StepCoef makeCoef(const DPMSolverSchedule& s, int step) {
    StepCoef k{};
    k.a = 1.0f / s.alpha_t[step];
    k.b = -s.sigma_t[step] / s.alpha_t[step];
    int nx = step + 1;
    float sn = s.sigma_t[nx], sc = s.sigma_t[step], an = s.alpha_t[nx];
    if (step == s.numSteps - 1 || sn <= 1e-4f) {
        k.k0 = 1.f; k.k1 = 0.f; k.f1 = 0.f; k.f2 = -1.f;
    } else if (step == 0) {
        float h = s.lambda_t[nx] - s.lambda_t[step];
        k.k0 = 1.f; k.k1 = 0.f; k.f1 = sn / sc; k.f2 = an * std::expm1(-h);
    } else {
        float h0 = s.lambda_t[step] - s.lambda_t[step - 1];
        float h  = s.lambda_t[nx]   - s.lambda_t[step];
        float r0 = h0 / h;
        k.k0 = 1.f + 0.5f / r0; k.k1 = 0.5f / r0;
        k.f1 = sn / sc; k.f2 = an * std::expm1(-h);
    }
    return k;
}

static void fusedStep(float* __restrict lat, float* __restrict lat2, const float* __restrict u,
                      const float* __restrict c, float* __restrict prev, size_t n, float g, const StepCoef& k) {
    size_t i = 0;
#if defined(__ARM_NEON)
    const float32x4_t vg = vdupq_n_f32(g), va = vdupq_n_f32(k.a), vb = vdupq_n_f32(k.b),
                      vk0 = vdupq_n_f32(k.k0), vk1 = vdupq_n_f32(k.k1),
                      vf1 = vdupq_n_f32(k.f1), vf2 = vdupq_n_f32(k.f2);
    for (; i + 4 <= n; i += 4) {
        float32x4_t L = vld1q_f32(lat + i), U = vld1q_f32(u + i), C = vld1q_f32(c + i), P = vld1q_f32(prev + i);
        float32x4_t N  = vfmaq_f32(U, vsubq_f32(C, U), vg);
        float32x4_t D  = vfmaq_f32(vmulq_f32(L, va), N, vb);
        float32x4_t DH = vfmsq_f32(vmulq_f32(D, vk0), P, vk1);
        float32x4_t R  = vfmsq_f32(vmulq_f32(L, vf1), DH, vf2);
        vst1q_f32(prev + i, D);
        vst1q_f32(lat + i, R);
        if (lat2) vst1q_f32(lat2 + i, R);
    }
#endif
    for (; i < n; ++i) {
        float nn = u[i] + g * (c[i] - u[i]);
        float d  = k.a * lat[i] + k.b * nn;
        float r  = k.f1 * lat[i] - k.f2 * (k.k0 * d - k.k1 * prev[i]);
        prev[i] = d; lat[i] = r; if (lat2) lat2[i] = r;
    }
}

static bool readRawFloats(const std::string& path, std::vector<float>& buf, size_t n) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    buf.resize(n);
    f.read(reinterpret_cast<char*>(buf.data()), std::streamsize(n * sizeof(float)));
    return f.gcount() == std::streamsize(n * sizeof(float));
}


// =====================================================================
// dma-buf CPU-access fencing (cache maintenance for CPU <-> NPU/GPU sharing)
// =====================================================================
struct sd_dma_buf_sync { uint64_t flags; };
static constexpr uint64_t SD_DMA_BUF_SYNC_READ = 1u << 0, SD_DMA_BUF_SYNC_WRITE = 2u << 0,
                          SD_DMA_BUF_SYNC_RW = SD_DMA_BUF_SYNC_READ | SD_DMA_BUF_SYNC_WRITE,
                          SD_DMA_BUF_SYNC_START = 0u << 2, SD_DMA_BUF_SYNC_END = 1u << 2;
#define SD_DMA_BUF_IOCTL_SYNC _IOW('b', 0, sd_dma_buf_sync)

static void dmabufSync(int fd, uint64_t flags) {
    static bool unsupported = false;            // rpcmem fds on some kernels reject the ioctl: stop trying
    if (fd < 0 || unsupported) return;
    sd_dma_buf_sync s{flags};
    if (ioctl(fd, SD_DMA_BUF_IOCTL_SYNC, &s) != 0 && (errno == ENOTTY || errno == EINVAL || errno == EBADF))
        unsupported = true;
}
// RAII: bracket every CPU touch of a buffer the NPU/GPU also uses.
struct CpuAccess {
    int fd; uint64_t rw;
    CpuAccess(const SharedBuf& b, bool write) : fd(b.fd), rw(write ? SD_DMA_BUF_SYNC_RW : SD_DMA_BUF_SYNC_READ) {
        dmabufSync(fd, SD_DMA_BUF_SYNC_START | rw);
    }
    ~CpuAccess() { dmabufSync(fd, SD_DMA_BUF_SYNC_END | rw); }
};

static size_t alignUp(size_t v, size_t a) { return (v + a - 1) / a * a; }

// =====================================================================
// Buffer pool: rpcmem (ION/system heap) | Linux dma-heap | AHardwareBuffer(BLOB)
// Every buffer is page aligned and padded to a multiple of 4 KiB (HTP maps whole pages; the
// 128-byte HVX vector width is a divisor, so any tensor start/stride stays vector aligned).
// =====================================================================
class BufferPool {
public:
    enum Mode { RPCMEM, DMAHEAP, AHB };
    Mode mode = RPCMEM;
    static constexpr size_t kPage = 4096;

    bool init(Mode m) {
        mode = m;
        if (mode == RPCMEM) return rpc.load();
        if (mode == DMAHEAP) {
            heapFd = open("/dev/dma_heap/system", O_RDONLY | O_CLOEXEC);
            if (heapFd < 0) { std::cerr << "[Alloc] /dev/dma_heap/system: " << std::strerror(errno) << std::endl; return false; }
            return true;
        }
#if defined(SD_USE_AHB)
        return true;
#else
        std::cerr << "[Alloc] built without -DSD_USE_AHB" << std::endl; return false;
#endif
    }

    bool alloc(SharedBuf& b, size_t bytes) {
        const size_t padded = alignUp(bytes, kPage);
        bool ok = false;
        if (mode == RPCMEM) ok = rpc.alloc(b, padded);
        else if (mode == DMAHEAP) ok = allocHeap(b, padded);
        else ok = allocAhb(b, padded);
        if (!ok) return false;
        b.bytes = bytes;
        if (reinterpret_cast<uintptr_t>(b.ptr) % 128 != 0) {
            std::cerr << "[Alloc] buffer not 128B aligned" << std::endl; return false;
        }
        return true;
    }

    ~BufferPool() {
        for (auto it = cleanups.rbegin(); it != cleanups.rend(); ++it) (*it)();
        if (heapFd >= 0) close(heapFd);
    }
    RpcMem rpc;

private:
    int heapFd = -1;
    std::vector<std::function<void()>> cleanups;

    struct sd_heap_alloc { uint64_t len; uint32_t fd; uint32_t fd_flags; uint64_t heap_flags; };
    bool allocHeap(SharedBuf& b, size_t padded) {
        sd_heap_alloc a{}; a.len = padded; a.fd_flags = O_RDWR | O_CLOEXEC;
        if (ioctl(heapFd, _IOWR('H', 0, sd_heap_alloc), &a) != 0) {
            std::cerr << "[Alloc] DMA_HEAP_IOCTL_ALLOC(" << padded << "): " << std::strerror(errno) << std::endl; return false;
        }
        const int fd = static_cast<int>(a.fd);
        void* p = mmap(nullptr, padded, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) { std::cerr << "[Alloc] mmap(dma-buf): " << std::strerror(errno) << std::endl; close(fd); return false; }
        { dmabufSync(fd, SD_DMA_BUF_SYNC_START | SD_DMA_BUF_SYNC_RW); std::memset(p, 0, padded); dmabufSync(fd, SD_DMA_BUF_SYNC_END | SD_DMA_BUF_SYNC_RW); }
        b.ptr = p; b.fd = fd;
        cleanups.push_back([=] { munmap(p, padded); close(fd); });
        return true;
    }

#if defined(SD_USE_AHB)
    struct sd_native_handle { int version, numFds, numInts; int data[1]; };
    bool allocAhb(SharedBuf& b, size_t padded) {
        AHardwareBuffer_Desc d{};
        d.width = static_cast<uint32_t>(padded); d.height = 1; d.layers = 1;
        d.format = AHARDWAREBUFFER_FORMAT_BLOB;
        d.usage = AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN | AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN |
                  AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER;
        AHardwareBuffer* hb = nullptr;
        if (AHardwareBuffer_allocate(&d, &hb) != 0 || !hb) { std::cerr << "[Alloc] AHardwareBuffer_allocate failed" << std::endl; return false; }
        void* p = nullptr;
        if (AHardwareBuffer_lock(hb, d.usage & (AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN | AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN), -1, nullptr, &p) != 0 || !p) {
            AHardwareBuffer_release(hb); std::cerr << "[Alloc] AHardwareBuffer_lock failed" << std::endl; return false;
        }
        const auto* nh = reinterpret_cast<const sd_native_handle*>(AHardwareBuffer_getNativeHandle(hb));
        if (!nh || nh->numFds < 1 || nh->data[0] < 0) {
            AHardwareBuffer_unlock(hb, nullptr); AHardwareBuffer_release(hb);
            std::cerr << "[Alloc] AHardwareBuffer has no dma-buf fd" << std::endl; return false;
        }
        std::memset(p, 0, padded);
        b.ptr = p; b.fd = nh->data[0];                      // fd stays owned by the AHardwareBuffer
        cleanups.push_back([=] { AHardwareBuffer_unlock(hb, nullptr); AHardwareBuffer_release(hb); });
        return true;
    }
#else
    bool allocAhb(SharedBuf&, size_t) { return false; }
#endif
};

// =====================================================================
// Completion fence between "NPU finished writing" and "GPU may read".
// Preferred: a real Linux sync_file created on a /dev/sw_sync timeline (poll()-able, mergeable,
// importable into Vulkan/EGL as SYNC_FD).  QNN-HTP does not export a sync_file for graphExecute,
// so the runner signals the timeline right after the (blocking) graphExecute returns.
// Fallback when sw_sync is not accessible (production Android): eventfd with identical poll() semantics.
// =====================================================================
class Fence {
public:
    bool init() {
        timeline = open("/dev/sw_sync", O_RDWR | O_CLOEXEC);
        real_ = timeline >= 0;
        if (g_verbose) std::cout << "[Fence] " << (real_ ? "sw_sync sync_file timeline" : "eventfd fallback (no /dev/sw_sync)") << std::endl;
        return true;
    }
    bool isSyncFile() const { return real_; }

    // Returns an fd (caller closes) that becomes readable when signal() is called. -1 on error.
    int arm() {
        if (real_) {
            struct { uint32_t value; char name[32]; int32_t fence; } d{};
            d.value = ++point; std::snprintf(d.name, sizeof(d.name), "npu_done_%u", d.value);
            if (ioctl(timeline, _IOWR('W', 0, d), &d) != 0) { std::cerr << "[Fence] CREATE_FENCE: " << std::strerror(errno) << std::endl; return -1; }
            return d.fence;
        }
        if (ev >= 0) { close(ev); ev = -1; }
        ev = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
        if (ev < 0) { std::cerr << "[Fence] eventfd: " << std::strerror(errno) << std::endl; return -1; }
        return dup(ev);
    }
    bool signal() {
        if (real_) { uint32_t one = 1; return ioctl(timeline, _IOW('W', 1, uint32_t), &one) == 0; }
        if (ev < 0) return false;
        uint64_t one = 1; const bool ok = write(ev, &one, sizeof(one)) == sizeof(one);
        close(ev); ev = -1; return ok;
    }
    static bool wait(int fd, int timeoutMs) {
        pollfd p{fd, POLLIN, 0};
        int r;
        do { r = poll(&p, 1, timeoutMs); } while (r < 0 && errno == EINTR);
        return r > 0 && (p.revents & POLLIN);
    }
    ~Fence() { if (timeline >= 0) close(timeline); if (ev >= 0) close(ev); }
private:
    int timeline = -1, ev = -1; uint32_t point = 0; bool real_ = false;
};

// =====================================================================
// Adreno OpenCL blender (libOpenCL.so via dlopen; own minimal CL type decls, no CL headers needed).
// Imports the NPU's VAE-output dma-buf and the final-image dma-buf with cl_qcom_ion_host_ptr, then
// runs gradient-domain (Poisson) boundary blending entirely on the GPU:
//   unknown d = f - g (f = blended, g = decoded) solves  laplace(d) = 0 in the mask,
//   d = original - g outside (Dirichlet).   coarse (1/4 res) Jacobi -> upsample -> fine Jacobi.
// =====================================================================
namespace sdcl {
using cl_int = int32_t; using cl_uint = uint32_t; using cl_ulong = uint64_t;
typedef struct _p* cl_platform_id; typedef struct _d* cl_device_id; typedef struct _c* cl_context;
typedef struct _q* cl_command_queue; typedef struct _m* cl_mem; typedef struct _pr* cl_program;
typedef struct _k* cl_kernel; typedef struct _e* cl_event;
constexpr cl_ulong DEV_GPU = 1u << 2, MEM_RW = 1u << 0, MEM_RO = 1u << 2;
constexpr cl_uint PROG_BUILD_LOG = 0x1183, DEV_EXT = 0x1030, DEV_NAME = 0x102B;
constexpr cl_ulong MEM_EXT_HOST_PTR_QCOM = 1ull << 29;   // verify against device cl_ext_qcom.h
constexpr cl_uint ION_HOST_PTR_QCOM = 0x40A8, HOST_WRITEBACK_QCOM = 0x40AA;
struct mem_ext_host_ptr { cl_uint allocation_type; cl_uint host_cache_policy; };
struct mem_ion_host_ptr { mem_ext_host_ptr ext_host_ptr; int ion_filedesc; void* ion_hostptr; };
}  // namespace sdcl

static const char* kPoissonCL = R"CLC(
#define W 512
#define H 512
__kernel void k_prep(__global const ushort* q, float scale, float off, __global const float* orig,
                     __global const float* mask, __global float* g, __global float* d, __global uchar* fx) {
  int x = get_global_id(0), y = get_global_id(1); int id = y * W + x; bool om = mask[id] > 0.5f;
  fx[id] = om ? 0 : 1;
  for (int c = 0; c < 3; ++c) { int i = id * 3 + c; float gv = ((float)q[i] + off) * scale;
    g[i] = gv; d[i] = om ? 0.f : orig[i] - gv; } }
__kernel void k_down4(__global const float* d, __global const uchar* fx, __global float* dc, __global uchar* fxc) {
  int x = get_global_id(0), y = get_global_id(1); int cnt = 0; float s0 = 0, s1 = 0, s2 = 0;
  for (int j = 0; j < 4; ++j) for (int i = 0; i < 4; ++i) { int id = (y * 4 + j) * W + x * 4 + i;
    if (fx[id]) { ++cnt; s0 += d[id * 3]; s1 += d[id * 3 + 1]; s2 += d[id * 3 + 2]; } }
  int o = y * (W / 4) + x; bool f = cnt >= 8; fxc[o] = f ? 1 : 0;
  float r = f ? 1.f / cnt : 0.f; dc[o * 3] = s0 * r; dc[o * 3 + 1] = s1 * r; dc[o * 3 + 2] = s2 * r; }
__kernel void k_jacobi(int w, int h, __global const float* src, __global float* dst, __global const uchar* fx) {
  int x = get_global_id(0), y = get_global_id(1); int id = y * w + x;
  int xl = max(x - 1, 0), xr = min(x + 1, w - 1), yu = max(y - 1, 0), yd = min(y + 1, h - 1);
  for (int c = 0; c < 3; ++c) dst[id * 3 + c] = fx[id] ? src[id * 3 + c] :
    0.25f * (src[(y * w + xl) * 3 + c] + src[(y * w + xr) * 3 + c] + src[(yu * w + x) * 3 + c] + src[(yd * w + x) * 3 + c]); }
__kernel void k_up4(__global const float* dc, __global const uchar* fx, __global float* d) {
  int x = get_global_id(0), y = get_global_id(1); int id = y * W + x; if (fx[id]) return;
  const int cw = W / 4, ch = H / 4; float fx_ = (x + 0.5f) / 4.f - 0.5f, fy_ = (y + 0.5f) / 4.f - 0.5f;
  int x0 = (int)floor(fx_), y0 = (int)floor(fy_); float tx = fx_ - x0, ty = fy_ - y0;
  int xa = clamp(x0, 0, cw - 1), xb = clamp(x0 + 1, 0, cw - 1), ya = clamp(y0, 0, ch - 1), yb = clamp(y0 + 1, 0, ch - 1);
  for (int c = 0; c < 3; ++c) d[id * 3 + c] =
    (1 - ty) * ((1 - tx) * dc[(ya * cw + xa) * 3 + c] + tx * dc[(ya * cw + xb) * 3 + c]) +
    ty * ((1 - tx) * dc[(yb * cw + xa) * 3 + c] + tx * dc[(yb * cw + xb) * 3 + c]); }
__kernel void k_final(__global const float* g, __global const float* d, __global uchar* out) {
  int x = get_global_id(0), y = get_global_id(1); int id = y * W + x;
  for (int c = 0; c < 3; ++c) { int i = id * 3 + c; out[i] = (uchar)clamp((g[i] + d[i]) * 255.f + 0.5f, 0.f, 255.f); } }
)CLC";

class GpuBlender {
public:
    bool ok = false, importedIn = false, importedOut = false;
    bool init() {
        lib = dlopen("libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
        if (!lib) lib = dlopen("/vendor/lib64/libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
        if (!lib) { std::cerr << "[GPU] libOpenCL.so not found" << std::endl; return false; }
#define SD_CL(n) n = (decltype(n))dlsym(lib, #n); if (!n) { std::cerr << "[GPU] missing " #n << std::endl; return false; }
        SD_CL(clGetPlatformIDs) SD_CL(clGetDeviceIDs) SD_CL(clGetDeviceInfo) SD_CL(clCreateContext)
        SD_CL(clCreateCommandQueue) SD_CL(clCreateProgramWithSource) SD_CL(clBuildProgram)
        SD_CL(clGetProgramBuildInfo) SD_CL(clCreateKernel) SD_CL(clSetKernelArg) SD_CL(clEnqueueNDRangeKernel)
        SD_CL(clCreateBuffer) SD_CL(clEnqueueWriteBuffer) SD_CL(clEnqueueReadBuffer) SD_CL(clFinish)
        SD_CL(clReleaseMemObject)
#undef SD_CL
        sdcl::cl_platform_id plats[8]; sdcl::cl_uint np = 0;
        if (clGetPlatformIDs(8, plats, &np) != 0 || np == 0) { std::cerr << "[GPU] no CL platform" << std::endl; return false; }
        for (sdcl::cl_uint i = 0; i < np && !dev; ++i) { sdcl::cl_uint nd = 0; sdcl::cl_device_id d;
            if (clGetDeviceIDs(plats[i], sdcl::DEV_GPU, 1, &d, &nd) == 0 && nd) dev = d; }
        if (!dev) { std::cerr << "[GPU] no GPU device" << std::endl; return false; }
        char name[128] = {}, ext[8192] = {};
        clGetDeviceInfo(dev, sdcl::DEV_NAME, sizeof(name), name, nullptr);
        clGetDeviceInfo(dev, sdcl::DEV_EXT, sizeof(ext) - 1, ext, nullptr);
        hasIon = std::strstr(ext, "cl_qcom_ion_host_ptr") != nullptr;
        if (g_verbose) std::cout << "[GPU] " << name << " | cl_qcom_ion_host_ptr=" << (hasIon ? "yes" : "no") << std::endl;
        sdcl::cl_int e = 0;
        ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &e); if (e) return fail("clCreateContext", e);
        q = clCreateCommandQueue(ctx, dev, 0, &e); if (e) return fail("clCreateCommandQueue", e);
        const char* src = kPoissonCL; size_t len = std::strlen(src);
        prog = clCreateProgramWithSource(ctx, 1, &src, &len, &e); if (e) return fail("clCreateProgramWithSource", e);
        if ((e = clBuildProgram(prog, 1, &dev, "-cl-fast-relaxed-math", nullptr, nullptr)) != 0) {
            char log[8192] = {}; clGetProgramBuildInfo(prog, dev, sdcl::PROG_BUILD_LOG, sizeof(log) - 1, log, nullptr);
            std::cerr << "[GPU] build log:\n" << log << std::endl; return fail("clBuildProgram", e); }
        const char* names[] = {"k_prep", "k_down4", "k_jacobi", "k_up4", "k_final"};
        sdcl::cl_kernel* ks[] = {&kPrep, &kDown, &kJac, &kUp, &kFin};
        for (int i = 0; i < 5; ++i) { *ks[i] = clCreateKernel(prog, names[i], &e); if (e) return fail(names[i], e); }
        const size_t F = 512 * 512 * 3 * sizeof(float);
        origB = mk(sdcl::MEM_RO, F); maskB = mk(sdcl::MEM_RO, 512 * 512 * 4);
        gB = mk(sdcl::MEM_RW, F); dA = mk(sdcl::MEM_RW, F); dB = mk(sdcl::MEM_RW, F); fxB = mk(sdcl::MEM_RW, 512 * 512);
        const size_t C = 128 * 128 * 3 * sizeof(float);
        cA = mk(sdcl::MEM_RW, C); cB = mk(sdcl::MEM_RW, C); fxcB = mk(sdcl::MEM_RW, 128 * 128);
        ok = origB && maskB && gB && dA && dB && fxB && cA && cB && fxcB;
        return ok;
    }
    // Zero-copy import of a dma-buf; on failure falls back to a device buffer + explicit copies.
    bool importBuf(const SharedBuf& b, bool readOnly, sdcl::cl_mem& out) {
        sdcl::cl_int e = 0;
        if (hasIon && b.fd >= 0) {
            sdcl::mem_ion_host_ptr h{}; h.ext_host_ptr.allocation_type = sdcl::ION_HOST_PTR_QCOM;
            h.ext_host_ptr.host_cache_policy = sdcl::HOST_WRITEBACK_QCOM; h.ion_filedesc = b.fd; h.ion_hostptr = b.ptr;
            out = clCreateBuffer(ctx, (readOnly ? sdcl::MEM_RO : sdcl::MEM_RW) | sdcl::MEM_EXT_HOST_PTR_QCOM,
                                 alignUp(b.bytes, 4096), &h, &e);
            if (!e && out) return true;
            std::cerr << "[GPU] ion import failed (" << e << "), copy fallback" << std::endl;
        }
        out = clCreateBuffer(ctx, readOnly ? sdcl::MEM_RO : sdcl::MEM_RW, b.bytes, nullptr, &e);
        return false;
    }
    bool bind(const SharedBuf& vaeOut, const SharedBuf& outImg) {
        importedIn = importBuf(vaeOut, true, inB); importedOut = importBuf(outImg, false, outB);
        return inB && outB;
    }
    bool setScene(const std::vector<float>& img, const std::vector<float>& mask) {
        return clEnqueueWriteBuffer(q, origB, 1, 0, img.size() * 4, img.data(), 0, nullptr, nullptr) == 0 &&
               clEnqueueWriteBuffer(q, maskB, 1, 0, mask.size() * 4, mask.data(), 0, nullptr, nullptr) == 0;
    }
    // Runs after the NPU fence fired.  Returns GPU wall ms (kernels + finish); <0 on error.
    double blend(const SharedBuf& vaeOut, const SharedBuf& outImg, float scale, float off, int coarseIt, int fineIt) {
        auto t0 = std::chrono::steady_clock::now();
        if (!importedIn && clEnqueueWriteBuffer(q, inB, 1, 0, vaeOut.bytes, vaeOut.ptr, 0, nullptr, nullptr)) return -1;
        sdcl::cl_int s = 0;
        const size_t g512[2] = {512, 512}, g128[2] = {128, 128};
        auto A = [&](sdcl::cl_kernel k, int i, size_t sz, const void* p) { s |= clSetKernelArg(k, i, sz, p); };
        const int c512 = 512, c128 = 128;
        A(kPrep, 0, sizeof(inB), &inB); A(kPrep, 1, 4, &scale); A(kPrep, 2, 4, &off); A(kPrep, 3, sizeof(origB), &origB);
        A(kPrep, 4, sizeof(maskB), &maskB); A(kPrep, 5, sizeof(gB), &gB); A(kPrep, 6, sizeof(dA), &dA); A(kPrep, 7, sizeof(fxB), &fxB);
        s |= clEnqueueNDRangeKernel(q, kPrep, 2, nullptr, g512, nullptr, 0, nullptr, nullptr);
        A(kDown, 0, sizeof(dA), &dA); A(kDown, 1, sizeof(fxB), &fxB); A(kDown, 2, sizeof(cA), &cA); A(kDown, 3, sizeof(fxcB), &fxcB);
        s |= clEnqueueNDRangeKernel(q, kDown, 2, nullptr, g128, nullptr, 0, nullptr, nullptr);
        sdcl::cl_mem src = cA, dst = cB;
        for (int i = 0; i < coarseIt; ++i) {
            A(kJac, 0, 4, &c128); A(kJac, 1, 4, &c128); A(kJac, 2, sizeof(src), &src); A(kJac, 3, sizeof(dst), &dst); A(kJac, 4, sizeof(fxcB), &fxcB);
            s |= clEnqueueNDRangeKernel(q, kJac, 2, nullptr, g128, nullptr, 0, nullptr, nullptr); std::swap(src, dst); }
        A(kUp, 0, sizeof(src), &src); A(kUp, 1, sizeof(fxB), &fxB); A(kUp, 2, sizeof(dA), &dA);
        s |= clEnqueueNDRangeKernel(q, kUp, 2, nullptr, g512, nullptr, 0, nullptr, nullptr);
        sdcl::cl_mem fs = dA, fd = dB;
        for (int i = 0; i < fineIt; ++i) {
            A(kJac, 0, 4, &c512); A(kJac, 1, 4, &c512); A(kJac, 2, sizeof(fs), &fs); A(kJac, 3, sizeof(fd), &fd); A(kJac, 4, sizeof(fxB), &fxB);
            s |= clEnqueueNDRangeKernel(q, kJac, 2, nullptr, g512, nullptr, 0, nullptr, nullptr); std::swap(fs, fd); }
        A(kFin, 0, sizeof(gB), &gB); A(kFin, 1, sizeof(fs), &fs); A(kFin, 2, sizeof(outB), &outB);
        s |= clEnqueueNDRangeKernel(q, kFin, 2, nullptr, g512, nullptr, 0, nullptr, nullptr);
        if (!importedOut) s |= clEnqueueReadBuffer(q, outB, 0, 0, outImg.bytes, outImg.ptr, 0, nullptr, nullptr);
        s |= clFinish(q);
        if (s) { std::cerr << "[GPU] kernel pipeline error " << s << std::endl; return -1; }
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
private:
    void* lib = nullptr;
    sdcl::cl_device_id dev = nullptr; sdcl::cl_context ctx = nullptr; sdcl::cl_command_queue q = nullptr;
    sdcl::cl_program prog = nullptr; sdcl::cl_kernel kPrep{}, kDown{}, kJac{}, kUp{}, kFin{};
    sdcl::cl_mem inB{}, outB{}, origB{}, maskB{}, gB{}, dA{}, dB{}, fxB{}, cA{}, cB{}, fxcB{};
    bool hasIon = false;
    sdcl::cl_int (*clGetPlatformIDs)(sdcl::cl_uint, sdcl::cl_platform_id*, sdcl::cl_uint*);
    sdcl::cl_int (*clGetDeviceIDs)(sdcl::cl_platform_id, sdcl::cl_ulong, sdcl::cl_uint, sdcl::cl_device_id*, sdcl::cl_uint*);
    sdcl::cl_int (*clGetDeviceInfo)(sdcl::cl_device_id, sdcl::cl_uint, size_t, void*, size_t*);
    sdcl::cl_context (*clCreateContext)(const intptr_t*, sdcl::cl_uint, const sdcl::cl_device_id*, void (*)(const char*, const void*, size_t, void*), void*, sdcl::cl_int*);
    sdcl::cl_command_queue (*clCreateCommandQueue)(sdcl::cl_context, sdcl::cl_device_id, sdcl::cl_ulong, sdcl::cl_int*);
    sdcl::cl_program (*clCreateProgramWithSource)(sdcl::cl_context, sdcl::cl_uint, const char**, const size_t*, sdcl::cl_int*);
    sdcl::cl_int (*clBuildProgram)(sdcl::cl_program, sdcl::cl_uint, const sdcl::cl_device_id*, const char*, void (*)(sdcl::cl_program, void*), void*);
    sdcl::cl_int (*clGetProgramBuildInfo)(sdcl::cl_program, sdcl::cl_device_id, sdcl::cl_uint, size_t, void*, size_t*);
    sdcl::cl_kernel (*clCreateKernel)(sdcl::cl_program, const char*, sdcl::cl_int*);
    sdcl::cl_int (*clSetKernelArg)(sdcl::cl_kernel, sdcl::cl_uint, size_t, const void*);
    sdcl::cl_int (*clEnqueueNDRangeKernel)(sdcl::cl_command_queue, sdcl::cl_kernel, sdcl::cl_uint, const size_t*, const size_t*, const size_t*, sdcl::cl_uint, const sdcl::cl_event*, sdcl::cl_event*);
    sdcl::cl_mem (*clCreateBuffer)(sdcl::cl_context, sdcl::cl_ulong, size_t, void*, sdcl::cl_int*);
    sdcl::cl_int (*clEnqueueWriteBuffer)(sdcl::cl_command_queue, sdcl::cl_mem, sdcl::cl_uint, size_t, size_t, const void*, sdcl::cl_uint, const sdcl::cl_event*, sdcl::cl_event*);
    sdcl::cl_int (*clEnqueueReadBuffer)(sdcl::cl_command_queue, sdcl::cl_mem, sdcl::cl_uint, size_t, size_t, void*, sdcl::cl_uint, const sdcl::cl_event*, sdcl::cl_event*);
    sdcl::cl_int (*clFinish)(sdcl::cl_command_queue);
    sdcl::cl_int (*clReleaseMemObject)(sdcl::cl_mem);
    bool fail(const char* w, sdcl::cl_int e) { std::cerr << "[GPU] " << w << " failed: " << e << std::endl; return false; }
    sdcl::cl_mem mk(sdcl::cl_ulong f, size_t n) { sdcl::cl_int e = 0; auto m = clCreateBuffer(ctx, f, n, nullptr, &e); return e ? nullptr : m; }
};


// =====================================================================
// PRODUCTION LAYER: options, requests, pipeline, CLI, server mode
// =====================================================================
static const char* kDefaultPrompt   = "photograph of a beautiful empty scene, highest quality settings";
static const char* kDefaultNegative = "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured";
static constexpr int kBatch    = UNET_BATCH;
static constexpr int kMaxSteps = 100;
static constexpr int kTokens   = 77;
static constexpr int kPix      = 512;                       // model resolution (fixed by the compiled graphs)

enum class MaskMode { Point, MaxPool };

struct Opts {                                               // fixed for the lifetime of the process
    std::string modelDir = "models", commonDir = "models";
    std::string fullBin, shallowBin;                        // explicit binaries override modelDir
    std::string fullGraph = "unet_full", shallowGraph = "unet_shallow";
    std::string vocab = "vocab.json", merges = "merges.txt";
    int interval = 3, order = 1;                            // cache: full step every N steps, Taylor order
    bool lambdaAxis = false, lastFull = false, baseline = false;
    BufferPool::Mode alloc = BufferPool::RPCMEM;
    bool gpu = true, zeroCopy = true, perf = true;
    int core = 7;                                           // host thread pin (-1 = none)
    int coarse = 300, fine = 40;                            // Poisson Jacobi sweeps
    MaskMode maskMode = MaskMode::Point;
    bool invertMask = false;
    bool serve = false, timing = false;
    int warmup = 0, repeat = 1;
    bool sceneCache = true;                                 // reuse the masked-image latent while image/mask are unchanged
};

struct Request {                                            // may change on every call
    std::string prompt = kDefaultPrompt, negative = kDefaultNegative;
    std::string image = "image.raw", mask = "mask.raw", output = "sd_output.png";
    int steps = 12;
    float cfg = 2.0f;                                       // <= 1.0 skips the negative (unconditional) branch
    uint32_t seed = 42;
};

struct Timings {                                            // milliseconds
    double load = 0, enc = 0, text = 0, denoise = 0, full = 0, shallow = 0, forecast = 0, host = 0,
           decode = 0, blend = 0, e2e = 0, write = 0, total = 0;
    int nFull = 0, nShallow = 0;
    bool encCached = false;
};

// ----------------------------- small helpers -----------------------------
static uint64_t fnv1a(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}
static bool endsWithNoCase(const std::string& s, const char* suf) {
    const size_t n = std::strlen(suf);
    if (s.size() < n) return false;
    for (size_t i = 0; i < n; ++i)
        if (std::tolower(static_cast<unsigned char>(s[s.size() - n + i])) != suf[i]) return false;
    return true;
}
static bool parseInt(const std::string& s, long lo, long hi, long& out) {
    char* e = nullptr; errno = 0;
    const long v = std::strtol(s.c_str(), &e, 10);
    if (s.empty() || *e != '\0' || errno != 0 || v < lo || v > hi) return false;
    out = v; return true;
}
static bool parseFloat(const std::string& s, double lo, double hi, double& out) {
    char* e = nullptr; errno = 0;
    const double v = std::strtod(s.c_str(), &e);
    if (s.empty() || *e != '\0' || errno != 0 || !(v >= lo && v <= hi)) return false;
    out = v; return true;
}

// Bilinear resize of an interleaved 8-bit image to kPix x kPix, output float in [0,1].
static void resizeToModel(const uint8_t* src, int sw, int sh, int ch, std::vector<float>& dst) {
    dst.assign(size_t(kPix) * kPix * ch, 0.f);
    const float fx = float(sw) / kPix, fy = float(sh) / kPix;
    for (int y = 0; y < kPix; ++y) {
        const float sy = std::min(std::max((y + 0.5f) * fy - 0.5f, 0.f), float(sh - 1));
        const int y0 = int(sy), y1 = std::min(y0 + 1, sh - 1); const float ty = sy - y0;
        for (int x = 0; x < kPix; ++x) {
            const float sx = std::min(std::max((x + 0.5f) * fx - 0.5f, 0.f), float(sw - 1));
            const int x0 = int(sx), x1 = std::min(x0 + 1, sw - 1); const float tx = sx - x0;
            for (int c = 0; c < ch; ++c) {
                const float a = src[(size_t(y0) * sw + x0) * ch + c], b = src[(size_t(y0) * sw + x1) * ch + c];
                const float d = src[(size_t(y1) * sw + x0) * ch + c], e = src[(size_t(y1) * sw + x1) * ch + c];
                const float v = (1 - ty) * ((1 - tx) * a + tx * b) + ty * ((1 - tx) * d + tx * e);
                dst[(size_t(y) * kPix + x) * ch + c] = v * (1.0f / 255.0f);
            }
        }
    }
}

// Image: *.raw = 512*512*3 float32 NHWC in [0,1]; anything else is decoded with stb_image (if built in).
static bool loadImage512(const std::string& path, std::vector<float>& out, std::string& err) {
    if (endsWithNoCase(path, ".raw")) {
        if (!readRawFloats(path, out, size_t(kPix) * kPix * 3)) { err = "cannot read raw image '" + path + "' (need 512*512*3 float32)"; return false; }
        return true;
    }
#if defined(SD_HAVE_STB_IMAGE)
    int w = 0, h = 0, c = 0;
    uint8_t* d = stbi_load(path.c_str(), &w, &h, &c, 3);
    if (!d) { err = "cannot decode image '" + path + "': " + stbi_failure_reason(); return false; }
    resizeToModel(d, w, h, 3, out);
    stbi_image_free(d);
    return true;
#else
    err = "'" + path + "' is not .raw and this build has no stb_image.h (put stb_image.h next to the source and rebuild)";
    return false;
#endif
}

// Mask: white / 1.0 = region to inpaint. *.raw = 512*512 float32; images are thresholded at 0.5.
static bool loadMask512(const std::string& path, bool invert, std::vector<float>& out, std::string& err) {
    if (endsWithNoCase(path, ".raw")) {
        if (!readRawFloats(path, out, size_t(kPix) * kPix)) { err = "cannot read raw mask '" + path + "' (need 512*512 float32)"; return false; }
    } else {
#if defined(SD_HAVE_STB_IMAGE)
        int w = 0, h = 0, c = 0;
        uint8_t* d = stbi_load(path.c_str(), &w, &h, &c, 1);
        if (!d) { err = "cannot decode mask '" + path + "': " + stbi_failure_reason(); return false; }
        resizeToModel(d, w, h, 1, out);
        stbi_image_free(d);
#else
        err = "'" + path + "' is not .raw and this build has no stb_image.h"; return false;
#endif
    }
    size_t on = 0;
    for (float& v : out) { v = (v > 0.5f) != invert ? 1.0f : 0.0f; on += (v > 0.5f); }
    if (on == 0) { err = "mask selects no pixels (check polarity: white = inpaint, or use --invert-mask)"; return false; }
    return true;
}

static void downsampleMask64(const std::vector<float>& m512, std::vector<float>& m64, MaskMode mode) {
    m64.assign(64 * 64, 0.f);
    for (int h = 0; h < 64; ++h)
        for (int w = 0; w < 64; ++w) {
            if (mode == MaskMode::Point) { m64[h * 64 + w] = m512[(h * 8) * kPix + (w * 8)] > 0.5f ? 1.f : 0.f; continue; }
            float m = 0.f;                                   // max-pool over the 8x8 block: every masked pixel is regenerated
            for (int j = 0; j < 8 && m < 0.5f; ++j) for (int i = 0; i < 8; ++i) m = std::max(m, m512[(h * 8 + j) * kPix + (w * 8 + i)]);
            m64[h * 64 + w] = m > 0.5f ? 1.f : 0.f;
        }
}

// Masked pixels are zero in the VAE-encoder input; others are mapped [0,1] -> [-1,1] -> UFIXED_POINT_16.
static void quantizeMaskedImage(const float* img, const float* mask, uint16_t* out, float scale, int zp) {
    const float inv = 1.0f / (scale > 0.f ? scale : 1.0f);
    const float a = 2.0f * inv, b = float(zp) - inv;
    const uint16_t zq = satU16(float(zp));                  // value 0.0 -> q = zp
    const size_t n = size_t(kPix) * kPix;
    for (size_t i = 0; i < n; ++i) {
        const bool m = mask[i] > 0.5f;
        const float* p = img + i * 3; uint16_t* o = out + i * 3;
        o[0] = m ? zq : satU16(p[0] * a + b);
        o[1] = m ? zq : satU16(p[1] * a + b);
        o[2] = m ? zq : satU16(p[2] * a + b);
    }
}

static bool writeImageFile(const std::string& path, const void* rgb) {
    if (endsWithNoCase(path, ".raw")) {
        std::ofstream f(path, std::ios::binary);
        if (!f) return false;
        f.write(static_cast<const char*>(rgb), std::streamsize(size_t(kPix) * kPix * 3));
        return bool(f);
    }
    if (endsWithNoCase(path, ".jpg") || endsWithNoCase(path, ".jpeg"))
        return stbi_write_jpg(path.c_str(), kPix, kPix, 3, rgb, 95) != 0;
    return stbi_write_png(path.c_str(), kPix, kPix, 3, rgb, kPix * 3) != 0;
}

// CLIP tokens: pad to 77 with the final (end-of-text) token; if too long keep the end token last.
static void fitTokens(std::vector<int32_t>& t) {
    if (t.empty()) t.push_back(0);
    if (t.size() > size_t(kTokens)) { const int32_t eos = t.back(); t.resize(kTokens); t[kTokens - 1] = eos; }
    else t.resize(kTokens, t.back());
}

struct UnetIo {                       // per-engine tensor indices
    int ts = -1, sample = -1, emb = -1, cacheIn = -1, out = -1, cacheOut = -1;
};
struct UnetReg {                      // per-engine registered buffers
    IoRef ts, sample, emb[2], out[2], cache[2][3];
};

static bool resolveIo(QnnModelEngine& e, int B, bool wantCacheIn, bool wantCacheOut, UnetIo& io, const char* what) {
    const uint64_t nS = (uint64_t)B * 16 * 64 * 64, nE = (uint64_t)B * 77 * 768, nO = (uint64_t)B * 4 * 64 * 64;
    for (size_t i = 0; i < e.numIn(); ++i) {
        const uint64_t n = e.inElems(i);
        if (n == nS) io.sample = (int)i; else if (n == nE) io.emb = (int)i; else if (n == (uint64_t)B) io.ts = (int)i;
        else io.cacheIn = (int)i;
    }
    for (size_t i = 0; i < e.numOut(); ++i) { if (e.outElems(i) == nO) io.out = (int)i; else io.cacheOut = (int)i; }
    const bool ok = io.ts >= 0 && io.sample >= 0 && io.emb >= 0 && io.out >= 0 &&
                    (wantCacheIn ? (io.cacheIn >= 0 && e.numIn() == 4) : e.numIn() == 3) &&
                    (wantCacheOut ? (io.cacheOut >= 0 && e.numOut() == 2) : e.numOut() >= 1);
    if (!ok) std::cerr << "[FATAL] " << what << " I/O does not match (B=" << B << "): in=" << e.numIn() << " out=" << e.numOut() << std::endl;
    if (ok && (e.inType(io.ts) != QNN_DATATYPE_FLOAT_32 || e.inType(io.sample) != QNN_DATATYPE_FLOAT_32 ||
               e.inType(io.emb) != QNN_DATATYPE_FLOAT_32 || e.outType(io.out) != QNN_DATATYPE_FLOAT_32)) {
        std::cerr << "[FATAL] " << what << " uses non-FLOAT32 ts/sample/emb/out" << std::endl; return false; }
    return ok;
}

// =====================================================================
// Pipeline: models stay loaded between requests. Member order matters for destruction
// (engines are destroyed before the buffer pool and the runtime).
// =====================================================================
class Pipeline {
public:
    bool init(const Opts& o);
    bool run(const Request& rq, Timings& T, std::string& err, bool writeOutput = true);
    bool warmup(int n);
    const Opts& opts() const { return o_; }

private:
    bool runScene(const Request& rq, const std::vector<float>& img, const std::vector<float>& mask,
                  Timings& T, std::string& err, bool writeOutput);

    Opts o_;
    QnnRuntime rt_;
    BufferPool pool_;
    Fence fence_;
    QnnModelEngine textEncoder_, unetFull_, unetShallow_, vaeDecoder_, vaeEncoder_;
    GpuBlender gpu_;
    CLIPTokenizer tokenizer_;

    UnetIo ioF_, ioS_;
    UnetReg rF_, rS_;
    SharedBuf sSample_, sTs_, sEmb_[2], sOut_[2], sCache_[2][3], sVaeIn_, sVaeOut_, sFinal_;
    CacheRing ring_[2];
    std::vector<float> prevD_;

    // quantisation parameters, read from the compiled binaries (real = (q - zp) * scale)
    float teScale_ = 0.0009303585393354297f;      int32_t teZp_ = 30063;
    float decInScale_ = 0.00034003707696683705f;  int32_t decInZp_ = 34382;
    float decOutScale_ = 0.000015259021893143654f; int32_t decOutZp_ = 0;
    float encInScale_ = 0.00003051804378628731f;  int32_t encInZp_ = 32768;
    float encOutScale_ = 0.00010813291009981185f; int32_t encOutZp_ = 42195;

    bool hasCache_ = false, gpuOn_ = false, ready_ = false;
    uint64_t cacheElems_ = 0; size_t cacheElemBytes_ = 4;
    float qF_ = 1, qI_ = 1; int32_t oF_ = 0, oI_ = 0;

    // scene cache: the masked-image latent is reused while image and mask are unchanged
    bool sceneValid_ = false; uint64_t sceneHash_ = 0;
    std::vector<float> img_, mask512_, mask64_;
    std::vector<uint16_t> ufixMasked_, encRaw_;
};

bool Pipeline::init(const Opts& o) {
    o_ = o;
    if (o_.core >= 0) pinToCore(o_.core);
    const size_t HW = 64 * 64, latent4Size = 4 * HW, sample16Size = 16 * HW;
    const size_t imagePixelCount = size_t(kPix) * kPix * 3, textEmbElements = size_t(kTokens) * 768;

    STAGE("runtime init");
    if (!rt_.init()) { std::cerr << "[FATAL] QNN HTP backend init failed (check LD_LIBRARY_PATH / ADSP_LIBRARY_PATH)" << std::endl; return false; }
    if (o_.perf) rt_.setupPerf();
    if (!pool_.init(o_.alloc)) { std::cerr << "[FATAL] buffer pool init failed" << std::endl; return false; }
    fence_.init();

    // Hard-coded quantisation values are only a fallback; they are replaced below by the binaries' own metadata.
    textEncoder_.addInputTensor(2, "tokens", QNN_DATATYPE_INT_32, {1, 77});
    textEncoder_.addOutputTensor(707, "text_embedding", QNN_DATATYPE_UFIXED_POINT_16, {1, 77, 768}, teScale_, -teZp_);
    if (!textEncoder_.init(rt_, "text_encoder", o_.commonDir + "/text_encoder.serialized.bin", "stable_diffusion_v1_5_text_encoder")) return false;

    const std::string fullBin = o_.fullBin.empty() ? o_.modelDir + "/unet_full_ctx.bin" : o_.fullBin;
    const std::string shallowBin = o_.shallowBin.empty() ? o_.modelDir + "/unet_shallow_ctx.bin" : o_.shallowBin;
    STAGE("init unet_full");
    if (!unetFull_.init(rt_, "unet_full", fullBin, o_.fullGraph)) return false;
    if (!o_.baseline) {
        STAGE("init unet_shallow");
        if (!unetShallow_.init(rt_, "unet_shallow", shallowBin, o_.shallowGraph)) return false;
    }
    vaeDecoder_.addInputTensor(1, "latent", QNN_DATATYPE_UFIXED_POINT_16, {1, 64, 64, 4}, decInScale_, -decInZp_);
    vaeDecoder_.addOutputTensor(426, "image", QNN_DATATYPE_UFIXED_POINT_16, {1, 512, 512, 3}, decOutScale_, -decOutZp_);
    if (!vaeDecoder_.init(rt_, "vae_decoder", o_.commonDir + "/vae.serialized.bin", "stable_diffusion_v1_5_vae")) return false;
    vaeEncoder_.addInputTensor(1, "image_input", QNN_DATATYPE_UFIXED_POINT_16, {1, 512, 512, 3}, encInScale_, -encInZp_);
    vaeEncoder_.addOutputTensor(1285, "latent_output", QNN_DATATYPE_UFIXED_POINT_16, {1, 64, 64, 4}, encOutScale_, -encOutZp_);
    if (!vaeEncoder_.init(rt_, "vae_encoder", o_.commonDir + "/vae_encoder_htp.serialized.bin", "vae_encoder")) return false;

    {   // read the quantisation parameters from the binaries (removes the old hard-coded constants)
        float sc; int32_t off;
        if (textEncoder_.outQuant(0, sc, off) && sc > 0) { teScale_ = sc; teZp_ = -off; }
        if (vaeDecoder_.inQuant(0, sc, off) && sc > 0)   { decInScale_ = sc; decInZp_ = -off; }
        if (vaeDecoder_.outQuant(0, sc, off) && sc > 0)  { decOutScale_ = sc; decOutZp_ = -off; }
        if (vaeEncoder_.inQuant(0, sc, off) && sc > 0)   { encInScale_ = sc; encInZp_ = -off; }
        if (vaeEncoder_.outQuant(0, sc, off) && sc > 0)  { encOutScale_ = sc; encOutZp_ = -off; }
        if (g_verbose) std::cout << "[Quant] text out " << teScale_ << "/" << teZp_ << " dec in " << decInScale_ << "/" << decInZp_
                                 << " dec out " << decOutScale_ << "/" << decOutZp_ << " enc in " << encInScale_ << "/" << encInZp_
                                 << " enc out " << encOutScale_ << "/" << encOutZp_ << std::endl;
    }

    STAGE("validate unet io");
    hasCache_ = unetFull_.numOut() == 2;
    if (!o_.baseline && !hasCache_) { std::cerr << "[FATAL] full graph lacks the cache output (export with dual_dlc_export_aimet.py, or use --no-cache with a plain U-Net)" << std::endl; return false; }
    if (!resolveIo(unetFull_, kBatch, false, hasCache_, ioF_, "unet_full")) return false;
    if (!o_.baseline && !resolveIo(unetShallow_, kBatch, true, false, ioS_, "unet_shallow")) return false;

    if (hasCache_) {
        cacheElems_ = unetFull_.outElems(ioF_.cacheOut);
        const Qnn_DataType_t cacheDt = unetFull_.outType(ioF_.cacheOut);
        if (!o_.baseline && (unetShallow_.inElems(ioS_.cacheIn) != cacheElems_ || unetShallow_.inType(ioS_.cacheIn) != cacheDt)) {
            std::cerr << "[FATAL] cache tensor mismatch between graphs (elems/dtype)" << std::endl; return false; }
        if (cacheDt == QNN_DATATYPE_UFIXED_POINT_16) {
            cacheElemBytes_ = 2;
            if (!unetFull_.outQuant(ioF_.cacheOut, qF_, oF_) || (!o_.baseline && !unetShallow_.inQuant(ioS_.cacheIn, qI_, oI_))) {
                std::cerr << "[FATAL] 16-bit cache without quant params" << std::endl; return false; }
            if (o_.baseline) { qI_ = qF_; oI_ = oF_; }
            if (qF_ != qI_ || oF_ != oI_) std::cerr << "[Warn] cache grids differ: using the requantising forecast kernel" << std::endl;
        } else if (cacheDt != QNN_DATATYPE_FLOAT_32) { std::cerr << "[FATAL] unsupported cache dtype 0x" << std::hex << cacheDt << std::dec << std::endl; return false; }
        if (g_verbose) std::cout << "[Cache] elems=" << cacheElems_ << " dtype=" << (cacheElemBytes_ == 4 ? "fp32" : "ufix16")
                                 << " bytes/slot=" << cacheElems_ * cacheElemBytes_ << std::endl;
    }

    STAGE("alloc + register buffers");
    const int nBranchMax = (kBatch == 2) ? 1 : 2;
    const size_t embBytes = textEmbElements * 4, latBytes = latent4Size * 4;
    bool ok = pool_.alloc(sSample_, kBatch * sample16Size * 4) && pool_.alloc(sTs_, 64);
    if (kBatch == 2) ok = ok && pool_.alloc(sEmb_[0], 2 * embBytes) && pool_.alloc(sOut_[0], 2 * latBytes);
    else             ok = ok && pool_.alloc(sEmb_[0], embBytes) && pool_.alloc(sEmb_[1], embBytes) &&
                              pool_.alloc(sOut_[0], latBytes) && pool_.alloc(sOut_[1], latBytes);
    if (hasCache_) for (int b = 0; b < nBranchMax; ++b) for (int s = 0; s < 3; ++s) ok = ok && pool_.alloc(sCache_[b][s], cacheElems_ * cacheElemBytes_);
    ok = ok && pool_.alloc(sVaeIn_, latent4Size * 2) && pool_.alloc(sVaeOut_, imagePixelCount * 2) && pool_.alloc(sFinal_, imagePixelCount);
    if (!ok) { std::cerr << "[FATAL] buffer allocation failed" << std::endl; return false; }
    sTs_.bytes = kBatch * 4;

    auto regUnet = [&](QnnModelEngine& e, const UnetIo& io, UnetReg& r, bool isFull) {
        const bool zc = o_.zeroCopy;
        r.ts = e.registerBuf(true, io.ts, sTs_, zc);
        r.sample = e.registerBuf(true, io.sample, sSample_, zc);
        r.emb[0] = e.registerBuf(true, io.emb, sEmb_[0], zc);
        r.out[0] = e.registerBuf(false, io.out, sOut_[0], zc);
        if (kBatch == 1) { r.emb[1] = e.registerBuf(true, io.emb, sEmb_[1], zc); r.out[1] = e.registerBuf(false, io.out, sOut_[1], zc); }
        if (isFull ? hasCache_ : true)
            for (int b = 0; b < nBranchMax; ++b) for (int s = 0; s < 3; ++s)
                r.cache[b][s] = isFull ? e.registerBuf(false, io.cacheOut, sCache_[b][s], zc)
                                       : e.registerBuf(true, io.cacheIn, sCache_[b][s], zc);
        e.select(true, io.ts, r.ts); e.select(true, io.sample, r.sample);
    };
    regUnet(unetFull_, ioF_, rF_, true);
    if (!o_.baseline) regUnet(unetShallow_, ioS_, rS_, false);
    IoRef rVaeIn = vaeDecoder_.registerBuf(true, 0, sVaeIn_, o_.zeroCopy), rVaeOut = vaeDecoder_.registerBuf(false, 0, sVaeOut_, o_.zeroCopy);
    vaeDecoder_.select(true, 0, rVaeIn); vaeDecoder_.select(false, 0, rVaeOut);
    const int regFail = unetFull_.regFailures() + unetShallow_.regFailures() + vaeDecoder_.regFailures();
    if (regFail) std::cerr << "[Warn] " << regFail << " buffer(s) fell back to raw pointers (slower)" << std::endl;

    STAGE("tokenizer + gpu");
    if (!tokenizer_.load(o_.vocab, o_.merges)) { std::cerr << "[FATAL] tokenizer load failed ('" << o_.vocab << "', '" << o_.merges << "')" << std::endl; return false; }
    if (o_.gpu) {
        gpuOn_ = gpu_.init() && gpu_.bind(sVaeOut_, sFinal_);
        if (!gpuOn_) std::cerr << "[Warn] GPU seam blending unavailable -> CPU composite" << std::endl;
        else if (g_verbose) std::cout << "[GPU] ready | zero-copy in=" << gpu_.importedIn << " out=" << gpu_.importedOut << std::endl;
    }
    ready_ = true;
    return true;
}

bool Pipeline::run(const Request& rq, Timings& T, std::string& err, bool writeOutput) {
    T = Timings{};
    const auto t0 = Clock::now();
    std::vector<float> img, mask;
    if (!loadImage512(rq.image, img, err) || !loadMask512(rq.mask, o_.invertMask, mask, err)) return false;
    const double loadMs = msSince(t0);
    const bool ok = runScene(rq, img, mask, T, err, writeOutput);
    T.load = loadMs;
    T.total = loadMs + T.e2e + T.write;
    return ok;
}

bool Pipeline::warmup(int n) {
    std::vector<float> img(size_t(kPix) * kPix * 3, 0.5f), mask(size_t(kPix) * kPix, 0.f);
    for (int y = 192; y < 320; ++y) for (int x = 192; x < 320; ++x) mask[size_t(y) * kPix + x] = 1.f;
    Request rq;
    rq.steps = o_.baseline ? 2 : std::max(3, o_.interval + 2);   // exercises the full and the shallow graph
    for (int i = 0; i < n; ++i) {
        Timings T; std::string err;
        if (!runScene(rq, img, mask, T, err, false)) { std::cerr << "[FATAL] warm-up failed: " << err << std::endl; return false; }
    }
    sceneValid_ = false;                                          // the dummy scene must not be mistaken for a real one
    return true;
}

bool Pipeline::runScene(const Request& rq, const std::vector<float>& imgIn, const std::vector<float>& maskIn,
                        Timings& T, std::string& err, bool writeOutput) {
    T = Timings{};
    if (!ready_) { err = "pipeline not initialised"; return false; }
    if (rq.steps < 2 || rq.steps > kMaxSteps) { err = "steps must be in [2," + std::to_string(kMaxSteps) + "]"; return false; }
    if (!(rq.cfg >= 0.0f && rq.cfg <= 30.0f)) { err = "cfg must be in [0,30]"; return false; }
    const auto tStart = Clock::now();

    const size_t HW = 64 * 64, latent4Size = 4 * HW, sample16Size = 16 * HW;
    const size_t imagePixelCount = size_t(kPix) * kPix * 3, maskPixelCount = size_t(kPix) * kPix, textEmbElements = size_t(kTokens) * 768;
    const size_t latBytes = latent4Size * 4;
    const bool singleBranch = (kBatch == 1) && !(rq.cfg > 1.0f + 1e-4f);       // cfg<=1: conditional branch only (half the U-Net work)
    const int nBranch = (kBatch == 2) ? 1 : (singleBranch ? 1 : 2);
    const int slotOf[2] = {singleBranch ? 1 : 0, 1};                            // emb/out slot per branch (0 = negative, 1 = positive)

    // ---------------- scene: VAE-encode only when image or mask changed ----------------
    uint64_t h = fnv1a(imgIn.data(), imgIn.size() * sizeof(float));
    h = fnv1a(maskIn.data(), maskIn.size() * sizeof(float), h);
    const bool changed = !o_.sceneCache || !sceneValid_ || h != sceneHash_;
    auto t0 = Clock::now();
    if (changed) {
        sceneValid_ = false;
        img_ = imgIn; mask512_ = maskIn;
        downsampleMask64(mask512_, mask64_, o_.maskMode);
        ufixMasked_.resize(imagePixelCount); encRaw_.resize(latent4Size);
        quantizeMaskedImage(img_.data(), mask512_.data(), ufixMasked_.data(), encInScale_, encInZp_);
        if (!vaeEncoder_.execute({ufixMasked_.data()}, {ufixMasked_.size() * 2}, {encRaw_.data()}, {latent4Size * 2})) { err = "VAE encoder execution failed"; return false; }
        {
            CpuAccess w(sSample_, true);
            for (int b = 0; b < kBatch; ++b) {
                float* s = sSample_.f32() + b * sample16Size;
                std::memcpy(s + 4 * HW, mask64_.data(), HW * 4);
                for (size_t p = 0; p < HW; ++p) for (size_t c = 0; c < 4; ++c)
                    s[(5 + c) * HW + p] = applyVaeEncScale(dequantizeFromUFix16(encRaw_[p * 4 + c], encOutScale_, encOutZp_));
                std::memset(s + 9 * HW, 0, 7 * HW * 4);
            }
        }
        if (gpuOn_ && !gpu_.setScene(img_, mask512_)) { std::cerr << "[Warn] GPU scene upload failed -> CPU composite" << std::endl; gpuOn_ = false; }
        sceneHash_ = h; sceneValid_ = true;
    } else T.encCached = true;
    T.enc = changed ? msSince(t0) : 0.0;

    // ---------------- text ----------------
    t0 = Clock::now();
    std::vector<uint16_t> cRaw(textEmbElements), uRaw;
    {
        std::vector<int32_t> pTok = tokenizer_.encode(rq.prompt); fitTokens(pTok);
        if (!textEncoder_.execute({pTok.data()}, {pTok.size() * 4}, {cRaw.data()}, {cRaw.size() * 2})) { err = "text encoder execution failed"; return false; }
        if (!singleBranch) {
            std::vector<int32_t> uTok = tokenizer_.encode(rq.negative); fitTokens(uTok);
            uRaw.resize(textEmbElements);
            if (!textEncoder_.execute({uTok.data()}, {uTok.size() * 4}, {uRaw.data()}, {uRaw.size() * 2})) { err = "text encoder execution failed (negative prompt)"; return false; }
        }
    }
    {
        float* embU = sEmb_[0].f32();
        float* embC = (kBatch == 2) ? sEmb_[0].f32() + textEmbElements : sEmb_[1].f32();
        CpuAccess w0(sEmb_[kBatch == 2 ? 0 : 1], true);
        for (size_t i = 0; i < textEmbElements; ++i) embC[i] = dequantizeFromUFix16(cRaw[i], teScale_, teZp_);
        if (!singleBranch) {
            CpuAccess w1(sEmb_[0], true);
            for (size_t i = 0; i < textEmbElements; ++i) embU[i] = dequantizeFromUFix16(uRaw[i], teScale_, teZp_);
        }
    }
    T.text = msSince(t0);

    // ---------------- initial noise + forecasting state ----------------
    float* lat = sSample_.f32();
    float* lat2 = (kBatch == 2) ? sSample_.f32() + sample16Size : nullptr;
    {
        CpuAccess w(sSample_, true);
        std::mt19937 rng(rq.seed); std::normal_distribution<float> g(0.0f, 1.0f);
        for (size_t i = 0; i < latent4Size; ++i) lat[i] = g(rng);
        if (lat2) std::memcpy(lat2, lat, latBytes);
    }
    prevD_.assign(latent4Size, 0.0f);
    ring_[0] = CacheRing{}; ring_[1] = CacheRing{};
    DPMSolverSchedule schedule(rq.steps);

    // ---------------- denoise ----------------
    const auto denoiseStart = Clock::now();
    const float guidance = singleBranch ? 1.0f : rq.cfg;
    for (int step = 0; step < rq.steps; ++step) {
        const float ts = schedule.timesteps[step];
        { CpuAccess w(sTs_, true); sTs_.f32()[0] = ts; if (kBatch == 2) sTs_.f32()[1] = ts; }
        const bool fullStep = o_.baseline || (step % o_.interval == 0) || (o_.lastFull && step == rq.steps - 1);
        const double x = o_.lambdaAxis ? double(schedule.lambda_t[step]) : double(step);
        double tU = 0, tF = 0;
        for (int b = 0; b < nBranch; ++b) {
            const int eb = (kBatch == 2) ? 0 : slotOf[b];
            if (fullStep) {
                unetFull_.select(true, ioF_.emb, rF_.emb[eb]); unetFull_.select(false, ioF_.out, rF_.out[eb]);
                if (hasCache_) unetFull_.select(false, ioF_.cacheOut, rF_.cache[b][ring_[b].fullTarget()]);
                auto e0 = Clock::now(); if (!unetFull_.executeBound()) { err = "U-Net (full) execution failed"; return false; } tU += msSince(e0);
            } else {
                CacheRing& R = ring_[b];
                int slot = R.t;
                const TaylorCoef k = makeTaylorCoef(o_.order, R.haveHistory(), x, R.xT, R.xPrev);
                if (k.useHistory) {
                    auto f0 = Clock::now();
                    slot = R.pred;
                    CpuAccess rt_(sCache_[b][R.t], false), rp_(sCache_[b][R.prev], false), wp_(sCache_[b][R.pred], true);
                    if (cacheElemBytes_ == 4)
                        forecastF32(static_cast<float*>(sCache_[b][R.pred].ptr), static_cast<const float*>(sCache_[b][R.t].ptr),
                                    static_cast<const float*>(sCache_[b][R.prev].ptr), cacheElems_, k.a, k.b);
                    else
                        forecastU16(static_cast<uint16_t*>(sCache_[b][R.pred].ptr), static_cast<const uint16_t*>(sCache_[b][R.t].ptr),
                                    static_cast<const uint16_t*>(sCache_[b][R.prev].ptr), cacheElems_, makeU16Forecast(k, qF_, oF_, qI_, oI_));
                    tF += msSince(f0);
                }
                unetShallow_.select(true, ioS_.emb, rS_.emb[eb]); unetShallow_.select(false, ioS_.out, rS_.out[eb]);
                unetShallow_.select(true, ioS_.cacheIn, rS_.cache[b][slot]);
                auto e0 = Clock::now(); if (!unetShallow_.executeBound()) { err = "U-Net (shallow) execution failed"; return false; } tU += msSince(e0);
            }
        }
        if (fullStep) { for (int b = 0; b < nBranch; ++b) if (hasCache_) ring_[b].commitFull(x); T.full += tU; ++T.nFull; }
        else { T.shallow += tU; ++T.nShallow; }
        T.forecast += tF;
        auto h0 = Clock::now();
        {
            const float* noiseU = (kBatch == 2) ? sOut_[0].f32() : (singleBranch ? sOut_[1].f32() : sOut_[0].f32());
            const float* noiseC = (kBatch == 2) ? sOut_[0].f32() + latent4Size : sOut_[1].f32();
            CpuAccess rU(sOut_[kBatch == 2 ? 0 : (singleBranch ? 1 : 0)], false), rC(sOut_[kBatch == 2 ? 0 : 1], false), wL(sSample_, true);
            fusedStep(lat, lat2, noiseU, noiseC, prevD_.data(), latent4Size, guidance, makeCoef(schedule, step));
        }
        T.host += msSince(h0);
        if (g_verbose) std::cout << "[STEP] " << step << " kind=" << (fullStep ? "full" : "shallow") << " unet_ms=" << tU << " forecast_ms=" << tF << std::endl;
    }
    T.denoise = msSince(denoiseStart);

    // ---------------- decode + seam blend ----------------
    t0 = Clock::now();
    {
        CpuAccess w(sVaeIn_, true);
        uint16_t* q = static_cast<uint16_t*>(sVaeIn_.ptr);
        CpuAccess r(sSample_, false);
        for (size_t p = 0; p < HW; ++p) for (size_t c = 0; c < 4; ++c)
            q[p * 4 + c] = quantizeToUFix16(applyVaeDecScale(lat[c * HW + p]), decInScale_, decInZp_);
    }
    std::future<double> gpuFut;
    int fenceFd = -1;
    if (gpuOn_) {
        fenceFd = fence_.arm();
        if (fenceFd < 0) { err = "fence creation failed"; return false; }
        gpuFut = std::async(std::launch::async, [this, fenceFd]() -> double {
            if (!Fence::wait(fenceFd, 5000)) { std::cerr << "[Fence] timeout waiting for NPU" << std::endl; return -1.0; }
            return gpu_.blend(sVaeOut_, sFinal_, decOutScale_, float(-decOutZp_), o_.coarse, o_.fine);
        });
    }
    const bool decOk = vaeDecoder_.executeBound();
    if (gpuOn_) { if (!fence_.signal()) std::cerr << "[Fence] signal failed" << std::endl; }
    if (!decOk) { if (gpuOn_) gpuFut.wait(); if (fenceFd >= 0) close(fenceFd); err = "VAE decoder execution failed"; return false; }
    T.decode = msSince(t0);
    if (gpuOn_) {
        T.blend = gpuFut.get(); close(fenceFd);
        if (T.blend < 0) { err = "GPU seam blend failed"; return false; }
    } else {                                                    // CPU hard-mask composite; the background is copied exactly
        auto c0 = Clock::now();
        CpuAccess r(sVaeOut_, false), w(sFinal_, true);
        const uint16_t* rgb = static_cast<const uint16_t*>(sVaeOut_.ptr); uint8_t* out = static_cast<uint8_t*>(sFinal_.ptr);
        for (size_t i = 0; i < maskPixelCount; ++i) {
            const bool m = mask512_[i] > 0.5f;
            for (int c = 0; c < 3; ++c) {
                const size_t idx = i * 3 + c;
                const float v = m ? dequantizeFromUFix16(rgb[idx], decOutScale_, decOutZp_) * 255.0f : img_[idx] * 255.0f;
                out[idx] = static_cast<uint8_t>(std::clamp(v + 0.5f, 0.0f, 255.0f));   // round, do not truncate
            }
        }
        T.decode += msSince(c0);
    }
    T.e2e = T.enc + T.text + T.denoise + T.decode + T.blend;      // image loading and file output are reported separately

    if (writeOutput) {
        const auto w0 = Clock::now();
        CpuAccess r(sFinal_, false);
        if (!writeImageFile(rq.output, sFinal_.ptr)) { err = "cannot write output '" + rq.output + "'"; return false; }
        T.write = msSince(w0);
    }
    (void)tStart;
    return true;
}

// =====================================================================
// CLI
// =====================================================================
static void printUsage(const char* a0) {
    std::cout <<
"Usage: " << a0 << " [options]            (positional form: " << a0 << " \"<prompt>\" <steps>)\n"
"\n"
"Inputs / outputs\n"
"  -p, --prompt TEXT       positive prompt            (default: \"" << kDefaultPrompt << "\")\n"
"  -n, --negative TEXT     negative prompt            (default: \"" << kDefaultNegative << "\")\n"
"  -s, --steps N           DPM-Solver++ (2M, Karras) steps, 2..100   (default 12; measured range 8-12)\n"
"  -g, --cfg X             guidance scale (default 2.0). X<=1 skips the negative branch: ~2x faster, negative prompt ignored\n"
"      --seed N            initial-noise seed (default 42)\n"
"  -i, --image FILE        input image: .raw (512*512*3 float32, [0,1]) or PNG/JPG if built with stb_image.h\n"
"  -m, --mask FILE         mask: .raw (512*512 float32) or PNG/JPG; white/1.0 = region to inpaint (default mask.raw)\n"
"      --invert-mask       treat black as the region to inpaint\n"
"  -o, --output FILE       output .png / .jpg / .raw uint8 RGB (default sd_output.png)\n"
"\n"
"Speed / quality\n"
"      --interval N        run the full U-Net every N steps, shallow graph + forecast in between (default 3; 1 = no cache)\n"
"      --preset NAME       quality (N=2) | balanced (N=3) | fast (N=4)\n"
"      --order 0|1         Taylor order of the cache forecast (default 1; 0 = reuse the newest feature)\n"
"      --axis index|lambda forecast axis (default index)\n"
"      --last-full         force the final step to run the full graph\n"
"      --no-cache          full graph every step (needs only unet_full; any plain U-Net binary works)\n"
"      --mask-mode point|maxpool   latent-mask downsampling (default point = benchmarked; maxpool is untested on device)\n"
"\n"
"Models and runtime\n"
"      --model-dir DIR     directory with unet_full_ctx.bin and unet_shallow_ctx.bin   (default models)\n"
"      --common-dir DIR    directory with text_encoder / vae / vae_encoder_htp .serialized.bin   (default models)\n"
"      --full-bin FILE --shallow-bin FILE --full-graph NAME --shallow-graph NAME   explicit overrides\n"
"      --vocab FILE --merges FILE   CLIP tokenizer files (default vocab.json, merges.txt)\n"
"      --no-gpu            CPU hard-mask composite instead of the GPU Poisson seam blend\n"
"      --no-zerocopy --no-perf --alloc rpcmem|dmaheap|ahb --core N (-1 = no pinning, default 7)\n"
"      --poisson-coarse N --poisson-fine N     Jacobi sweeps (default 300 / 40)\n"
"\n"
"Modes\n"
"      --serve             keep the models loaded and read requests from stdin (one per line, tab separated key=value:\n"
"                          prompt= negative= steps= cfg= seed= image= mask= output=); replies 'OK<TAB>ms<TAB>file' or 'ERR<TAB>msg';\n"
"                          the VAE encoder is skipped while image and mask are unchanged. 'quit' ends the session.\n"
"      --warmup N          run N dummy generations first (default 0 for one-shot, 1 with --serve)\n"
"      --repeat N          run the request N times and report the mean timing (benchmarking; re-encodes every time)\n"
"      --no-scene-cache    always re-run the VAE encoder (serve mode default: cached while image/mask are unchanged)\n"
"      --timing            print one [TIMING] line per request\n"
"  -v, --verbose           stage / model / step logging\n"
"  -h, --help\n"
"\n"
"Environment (defaults only; options win): SD_MODEL_DIR SD_COMMON_DIR SD_CACHE_INTERVAL SD_ORDER SD_TAYLOR_AXIS SD_LAST_FULL\n"
"  SD_ALLOC SD_NO_GPU SD_NO_ZEROCOPY SD_NO_PERF SD_NO_PIN SD_POISSON_COARSE SD_POISSON_FINE SD_FULL_BIN SD_SHALLOW_BIN SD_MODE=baseline\n"
"Exit codes: 0 ok, 2 usage error, 3 initialisation failed, 4 generation failed.\n";
}

static void applyEnvDefaults(Opts& o) {
    o.baseline = envStr("SD_MODE", "proposed") == "baseline";
    o.modelDir = envStr("SD_MODEL_DIR", "models"); o.commonDir = envStr("SD_COMMON_DIR", "models");
    o.fullBin = envStr("SD_FULL_BIN", ""); o.shallowBin = envStr("SD_SHALLOW_BIN", "");
    o.fullGraph = envStr("SD_FULL_GRAPH", "unet_full"); o.shallowGraph = envStr("SD_SHALLOW_GRAPH", "unet_shallow");
    long v;
    if (parseInt(envStr("SD_CACHE_INTERVAL", "3"), 1, 1000, v)) o.interval = int(v);
    if (parseInt(envStr("SD_ORDER", "1"), 0, 1, v)) o.order = int(v);
    o.lambdaAxis = envStr("SD_TAYLOR_AXIS", "index") == "lambda";
    o.lastFull = envOn("SD_LAST_FULL");
    const std::string a = envStr("SD_ALLOC", "rpcmem");
    o.alloc = a == "dmaheap" ? BufferPool::DMAHEAP : a == "ahb" ? BufferPool::AHB : BufferPool::RPCMEM;
    o.gpu = !envOn("SD_NO_GPU"); o.zeroCopy = !envOn("SD_NO_ZEROCOPY"); o.perf = !envOn("SD_NO_PERF");
    if (envOn("SD_NO_PIN")) o.core = -1;
    if (parseInt(envStr("SD_POISSON_COARSE", "300"), 0, 100000, v)) o.coarse = int(v);
    if (parseInt(envStr("SD_POISSON_FINE", "40"), 0, 100000, v)) o.fine = int(v);
}

struct Args { Opts o; Request r; bool help = false; bool promptSet = false, negSet = false; };

static bool parseArgs(int argc, char** argv, Args& a, std::string& err) {
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i], val; bool hasInline = false;
        if (k.size() > 2 && k[0] == '-' && k[1] == '-') { const size_t eq = k.find('='); if (eq != std::string::npos) { val = k.substr(eq + 1); k = k.substr(0, eq); hasInline = true; } }
        if (k.empty() || k[0] != '-' || k == "-") { pos.push_back(argv[i]); continue; }
        auto need = [&](std::string& out) -> bool {
            if (hasInline) { out = val; return true; }
            if (i + 1 >= argc) { err = "option " + k + " needs a value"; return false; }
            out = argv[++i]; return true;
        };
        std::string s; long iv = 0; double dv = 0;
        auto needInt = [&](long lo, long hi, long& out) -> bool {
            if (!need(s)) return false;
            if (!parseInt(s, lo, hi, out)) { err = "option " + k + ": '" + s + "' is not an integer in [" + std::to_string(lo) + "," + std::to_string(hi) + "]"; return false; }
            return true;
        };
        if (k == "-h" || k == "--help") a.help = true;
        else if (k == "-v" || k == "--verbose") g_verbose = true;
        else if (k == "-p" || k == "--prompt") { if (!need(a.r.prompt)) return false; a.promptSet = true; }
        else if (k == "-n" || k == "--negative") { if (!need(a.r.negative)) return false; a.negSet = true; }
        else if (k == "-s" || k == "--steps") { if (!needInt(2, kMaxSteps, iv)) return false; a.r.steps = int(iv); }
        else if (k == "-g" || k == "--cfg") { if (!need(s)) return false; if (!parseFloat(s, 0.0, 30.0, dv)) { err = "option " + k + ": '" + s + "' is not a number in [0,30]"; return false; } a.r.cfg = float(dv); }
        else if (k == "--seed") { if (!needInt(0, 4294967295L, iv)) return false; a.r.seed = uint32_t(iv); }
        else if (k == "-i" || k == "--image") { if (!need(a.r.image)) return false; }
        else if (k == "-m" || k == "--mask") { if (!need(a.r.mask)) return false; }
        else if (k == "-o" || k == "--output") { if (!need(a.r.output)) return false; }
        else if (k == "--invert-mask") a.o.invertMask = true;
        else if (k == "--interval") { if (!needInt(1, 1000, iv)) return false; a.o.interval = int(iv); }
        else if (k == "--preset") {
            if (!need(s)) return false;
            if (s == "quality") a.o.interval = 2; else if (s == "balanced") a.o.interval = 3; else if (s == "fast") a.o.interval = 4;
            else { err = "unknown preset '" + s + "' (quality|balanced|fast)"; return false; }
        }
        else if (k == "--order") { if (!needInt(0, 1, iv)) return false; a.o.order = int(iv); }
        else if (k == "--axis") { if (!need(s)) return false; if (s != "index" && s != "lambda") { err = "--axis must be index or lambda"; return false; } a.o.lambdaAxis = (s == "lambda"); }
        else if (k == "--last-full") a.o.lastFull = true;
        else if (k == "--no-cache") a.o.baseline = true;
        else if (k == "--mask-mode") { if (!need(s)) return false; if (s == "point") a.o.maskMode = MaskMode::Point; else if (s == "maxpool") a.o.maskMode = MaskMode::MaxPool; else { err = "--mask-mode must be point or maxpool"; return false; } }
        else if (k == "--model-dir") { if (!need(a.o.modelDir)) return false; }
        else if (k == "--common-dir") { if (!need(a.o.commonDir)) return false; }
        else if (k == "--full-bin") { if (!need(a.o.fullBin)) return false; }
        else if (k == "--shallow-bin") { if (!need(a.o.shallowBin)) return false; }
        else if (k == "--full-graph") { if (!need(a.o.fullGraph)) return false; }
        else if (k == "--shallow-graph") { if (!need(a.o.shallowGraph)) return false; }
        else if (k == "--vocab") { if (!need(a.o.vocab)) return false; }
        else if (k == "--merges") { if (!need(a.o.merges)) return false; }
        else if (k == "--no-gpu") a.o.gpu = false;
        else if (k == "--no-zerocopy") a.o.zeroCopy = false;
        else if (k == "--no-perf") a.o.perf = false;
        else if (k == "--alloc") { if (!need(s)) return false; if (s == "rpcmem") a.o.alloc = BufferPool::RPCMEM; else if (s == "dmaheap") a.o.alloc = BufferPool::DMAHEAP; else if (s == "ahb") a.o.alloc = BufferPool::AHB; else { err = "--alloc must be rpcmem, dmaheap or ahb"; return false; } }
        else if (k == "--core") { if (!needInt(-1, 63, iv)) return false; a.o.core = int(iv); }
        else if (k == "--poisson-coarse") { if (!needInt(0, 100000, iv)) return false; a.o.coarse = int(iv); }
        else if (k == "--poisson-fine") { if (!needInt(0, 100000, iv)) return false; a.o.fine = int(iv); }
        else if (k == "--serve") a.o.serve = true;
        else if (k == "--warmup") { if (!needInt(0, 100, iv)) return false; a.o.warmup = int(iv); }
        else if (k == "--repeat") { if (!needInt(1, 10000, iv)) return false; a.o.repeat = int(iv); }
        else if (k == "--timing") a.o.timing = true;
        else if (k == "--no-scene-cache") a.o.sceneCache = false;
        else { err = "unknown option '" + k + "' (see --help)"; return false; }
    }
    if (pos.size() > 2) { err = "too many positional arguments"; return false; }
    if (pos.size() >= 1) { a.r.prompt = pos[0]; a.promptSet = true; }
    if (pos.size() == 2) {
        long v; if (!parseInt(pos[1], 2, kMaxSteps, v)) { err = "steps '" + pos[1] + "' is not an integer in [2," + std::to_string(kMaxSteps) + "]"; return false; }
        a.r.steps = int(v);
    }
    return true;
}

static void printTiming(const Timings& t) {
    std::cout << "[TIMING] total_ms=" << t.total << " e2e_ms=" << t.e2e << " load_ms=" << t.load << " enc_ms=" << t.enc
              << (t.encCached ? "(cached)" : "") << " text_ms=" << t.text << " denoise_ms=" << t.denoise
              << " full_n=" << t.nFull << " full_ms=" << t.full << " shallow_n=" << t.nShallow << " shallow_ms=" << t.shallow
              << " forecast_ms=" << t.forecast << " host_ms=" << t.host << " decode_ms=" << t.decode << " blend_ms=" << t.blend
              << " write_ms=" << t.write << std::endl;
}

// One request per line: tab separated key=value pairs. Values may contain spaces and '='.
static bool parseServeLine(const std::string& line, Request& r, std::string& err) {
    size_t pos = 0;
    while (pos <= line.size()) {
        size_t end = line.find('\t', pos); if (end == std::string::npos) end = line.size();
        const std::string tok = line.substr(pos, end - pos);
        pos = end + 1;
        if (tok.empty()) continue;
        const size_t eq = tok.find('=');
        if (eq == std::string::npos) { err = "expected key=value, got '" + tok + "'"; return false; }
        const std::string k = tok.substr(0, eq), v = tok.substr(eq + 1);
        long iv; double dv;
        if (k == "prompt") r.prompt = v;
        else if (k == "negative" || k == "neg") r.negative = v;
        else if (k == "steps") { if (!parseInt(v, 2, kMaxSteps, iv)) { err = "steps must be an integer in [2," + std::to_string(kMaxSteps) + "]"; return false; } r.steps = int(iv); }
        else if (k == "cfg") { if (!parseFloat(v, 0.0, 30.0, dv)) { err = "cfg must be a number in [0,30]"; return false; } r.cfg = float(dv); }
        else if (k == "seed") { if (!parseInt(v, 0, 4294967295L, iv)) { err = "seed must be a non-negative integer"; return false; } r.seed = uint32_t(iv); }
        else if (k == "image") r.image = v;
        else if (k == "mask") r.mask = v;
        else if (k == "output" || k == "out") r.output = v;
        else { err = "unknown key '" + k + "'"; return false; }
    }
    return true;
}

int main(int argc, char** argv) {
    Args a;
    applyEnvDefaults(a.o);
    std::string err;
    if (!parseArgs(argc, argv, a, err)) { std::cerr << "error: " << err << "\n(run with --help for usage)" << std::endl; return 2; }
    if (a.help) { printUsage(argv[0]); return 0; }
    if (a.r.cfg <= 1.0f + 1e-4f && a.negSet && !a.o.serve)
        std::cerr << "[Warn] --cfg <= 1 skips the negative branch; the negative prompt is ignored" << std::endl;
    if (a.o.baseline) a.o.interval = 1;
    if (a.o.repeat > 1) a.o.sceneCache = false;            // benchmark semantics: every run is a full request

    Pipeline pipe;
    if (!pipe.init(a.o)) return 3;
    const int warm = a.o.warmup > 0 ? a.o.warmup : (a.o.serve ? 1 : 0);
    if (warm > 0 && !pipe.warmup(warm)) return 3;

    if (a.o.serve) {
        std::signal(SIGPIPE, SIG_IGN);
        std::cout << "READY" << std::endl;
        std::string line;
        while (std::getline(std::cin, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            if (line == "quit" || line == "exit") break;
            Request r = a.r;                               // command-line values are the per-request defaults
            Timings t; std::string e;
            if (!parseServeLine(line, r, e) || !pipe.run(r, t, e, true)) { std::cout << "ERR\t" << e << std::endl; continue; }
            std::cout << "OK\t" << t.e2e << "\t" << r.output << std::endl;
            if (a.o.timing) printTiming(t);
        }
        return 0;
    }

    Timings sum; int okRuns = 0;
    for (int i = 0; i < a.o.repeat; ++i) {
        Timings t; std::string e;
        if (!pipe.run(a.r, t, e, i == 0)) { std::cerr << "error: generation failed: " << e << std::endl; return 4; }
        if (a.o.timing && a.o.repeat > 1) printTiming(t);
        sum.e2e += t.e2e; sum.total += t.total; ++okRuns;
        if (i == 0) { std::cout << "Saved " << a.r.output << std::endl; if (a.o.timing) printTiming(t); }
    }
    if (a.o.repeat > 1) std::cout << "[REPEAT] runs=" << okRuns << " mean_e2e_ms=" << sum.e2e / okRuns << " mean_total_ms=" << sum.total / okRuns << std::endl;
    return 0;
}