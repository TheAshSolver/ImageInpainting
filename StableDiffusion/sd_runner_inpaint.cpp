// sd_runner_inpaint_opt.cpp  (v3: zero-copy registration fix)

#include <iostream>
#include <fstream>
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
#include <type_traits>
#include <utility>
#include <dlfcn.h>
#include <cerrno>
#include <sched.h>
#include <unistd.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
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

#ifndef UNET_BATCH
#define UNET_BATCH 1
#endif

using Clock = std::chrono::steady_clock;
static inline double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
#define STAGE(msg) do { std::cout << "[Stage] " << msg << std::endl; } while (0)
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

    bool readMeta(const std::vector<char>& bin, std::vector<GraphMeta>& metas) {
        if (!sysOk) return false;
        QnnSystemContext_Handle_t h = nullptr;
        if (sys.systemContextCreate(&h) != QNN_SUCCESS || !h) return false;
        const QnnSystemContext_BinaryInfo_t* bi = nullptr; Qnn_ContextBinarySize_t bsz = 0;
        bool ok = false;
        if (sys.systemContextGetBinaryInfo(h, const_cast<char*>(bin.data()), bin.size(), &bi, &bsz) == QNN_SUCCESS && bi) {
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
        std::cout << "[Perf] HTP burst / DCVS-off / HMX-max / RPC polling enabled" << std::endl;
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

    bool init(QnnRuntime& rt, const std::string& tag, const std::string& binPath, const std::string& graphHint) {
        m_rt = &rt; m_tag = tag;
        std::ifstream f(binPath, std::ios::binary | std::ios::ate);
        if (!f.is_open()) { std::cerr << "[Init] Cannot open '" << binPath << "': " << std::strerror(errno) << std::endl; return false; }
        std::streamsize sz = f.tellg(); f.seekg(0);
        std::vector<char> buf(sz);
        if (!f.read(buf.data(), sz)) { std::cerr << "[Init] short read " << binPath << std::endl; return false; }
        std::cout << "[Init] " << tag << ": " << binPath << " (" << sz << " bytes)" << std::endl;

        std::string graphName = graphHint;
        std::vector<GraphMeta> metas;
        if (rt.readMeta(buf, metas)) {
            const GraphMeta* pick = nullptr;
            for (auto& m : metas) if (m.name == graphHint) pick = &m;
            if (!pick) {
                pick = &metas[0];
                std::cerr << "[Warn] " << tag << ": graph '" << graphHint << "' not in binary; using '" << pick->name
                          << "' (" << metas.size() << " graph(s) present)" << std::endl;
            }
            graphName = pick->name;
            m_inKeep = pick->in; m_outKeep = pick->out;
            std::cout << "[Meta] " << tag << " graph '" << graphName << "' (from binary):" << std::endl;
        } else {
            std::cout << "[Meta] " << tag << " using hard-coded tensor defs for graph '" << graphName << "':" << std::endl;
        }
        rebuild();
        dumpIo();

        CHECK_QNN_STATUS(rt.qnn.contextCreateFromBinary(rt.backend, rt.device, nullptr, buf.data(), sz, &m_ctx, nullptr),
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

    IoRef registerBuf(bool isInput, size_t idx, const SharedBuf& b, bool allowZeroCopy) {
        IoRef r; r.ptr = b.ptr; r.bytes = b.bytes;
        if (!allowZeroCopy || !m_rt->qnn.memRegister || b.fd < 0) return r;
        Qnn_Tensor_t& t = isInput ? m_in[idx] : m_out[idx];

        Qnn_MemDescriptor_t d = QNN_MEM_DESCRIPTOR_INIT;
        d.memShape = {t.v1.rank, t.v1.dimensions, nullptr};
        d.dataType = t.v1.dataType;
        d.memType  = QNN_MEM_TYPE_ION;
        d.ionInfo.fd = b.fd;

        Qnn_MemHandle_t h = nullptr;
        Qnn_ErrorHandle_t s = m_rt->qnn.memRegister(m_ctx, &d, 1, &h);
        if (s != QNN_SUCCESS || !h) {
            std::cerr << "[Mem] " << m_tag << " memRegister failed (" << (isInput ? "in " : "out ") << idx << ", err " << s
                      << ") -> using RAW pointer fallback" << std::endl;
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

static void downsampleMaskNearest(const std::vector<float>& m512, std::vector<float>& m64) {
    m64.resize(64 * 64);
    for (int h = 0; h < 64; ++h)
        for (int w = 0; w < 64; ++w)
            m64[h * 64 + w] = (m512[(h * 8) * 512 + (w * 8)] > 0.5f) ? 1.0f : 0.0f;
}

int main(int argc, char** argv) {
    int numSteps = (argc > 2) ? std::atoi(argv[2]) : 12;
    if (numSteps < 2) numSteps = 2;
    const float guidanceScale = 2.0f;
    const bool doProfile = envOn("SD_PROFILE");
    const bool zeroCopy  = !envOn("SD_NO_ZEROCOPY");
    std::string outputPath = "sd_output.png";

    if (!envOn("SD_NO_PIN")) pinToCore(7);
    std::cout << "SD1.5 inpaint | steps " << numSteps << " | CFG " << guidanceScale << " | UNET_BATCH " << UNET_BATCH
              << " | zero-copy " << (zeroCopy ? "on" : "off") << std::endl;

    const float TE_OUT_SCALE = 0.0009303585393354297f;   const int32_t TE_OUT_ZP = 30063;
    const float VAE_LATENT_IN_SCALE = 0.00034003707696683705f; const int32_t VAE_LATENT_IN_ZP = 34382;
    const float VAE_IMAGE_OUT_SCALE = 0.000015259021893143654f; const int32_t VAE_IMAGE_OUT_ZP = 0;
    const float VAE_ENC_IN_SCALE = 0.00003051804378628731f;  const int32_t VAE_ENC_IN_ZP = 32768;
    const float VAE_ENC_OUT_SCALE = 0.00010813291009981185f; const int32_t VAE_ENC_OUT_ZP = 42195;

    const size_t HW = 64 * 64, latent4Size = 4 * HW, sample16Size = 16 * HW;
    const size_t imagePixelCount = 512 * 512 * 3, maskPixelCount = 512 * 512, textEmbElements = 77 * 768;
    constexpr int B = UNET_BATCH;

    STAGE("runtime init");
    QnnRuntime rt;
    if (!rt.init()) return -1;
    if (!envOn("SD_NO_PERF")) rt.setupPerf();
    STAGE("rpcmem load");
    RpcMem rpc;
    if (!rpc.load()) return -1;

    QnnModelEngine textEncoder, unet, vaeDecoder, vaeEncoder;

    textEncoder.addInputTensor(2, "tokens", QNN_DATATYPE_INT_32, {1, 77});
    textEncoder.addOutputTensor(707, "text_embedding", QNN_DATATYPE_UFIXED_POINT_16, {1, 77, 768}, TE_OUT_SCALE, TE_OUT_ZP);
    STAGE("init text encoder");
    if (!textEncoder.init(rt, "text_encoder", "models/text_encoder.serialized.bin", "stable_diffusion_v1_5_text_encoder")) return -1;

    unet.addInputTensor(1, "timestep", QNN_DATATYPE_FLOAT_32, {(uint32_t)B});
    unet.addInputTensor(22, "sample", QNN_DATATYPE_FLOAT_32, {(uint32_t)B, 16, 64, 64});
    unet.addInputTensor(305, "encoder_hidden_states", QNN_DATATYPE_FLOAT_32, {(uint32_t)B, 77, 768});
    unet.addOutputTensor(8669, "out_sample", QNN_DATATYPE_FLOAT_32, {(uint32_t)B, 4, 64, 64});
    STAGE("init unet");
    if (!unet.init(rt, "unet", envStr("SD_UNET_BIN", "models/unet_16ch_ctx_qairt249.bin"),
                   envStr("SD_UNET_GRAPH", "graph_ny2_2at0"))) return -1;
    if (doProfile) unet.enableProfiling();

    vaeDecoder.addInputTensor(1, "latent", QNN_DATATYPE_UFIXED_POINT_16, {1, 64, 64, 4}, VAE_LATENT_IN_SCALE, VAE_LATENT_IN_ZP);
    vaeDecoder.addOutputTensor(426, "image", QNN_DATATYPE_UFIXED_POINT_16, {1, 512, 512, 3}, VAE_IMAGE_OUT_SCALE, VAE_IMAGE_OUT_ZP);
    STAGE("init vae decoder");
    if (!vaeDecoder.init(rt, "vae_decoder", "models/vae.serialized.bin", "stable_diffusion_v1_5_vae")) return -1;

    vaeEncoder.addInputTensor(1, "image_input", QNN_DATATYPE_UFIXED_POINT_16, {1, 512, 512, 3}, VAE_ENC_IN_SCALE, VAE_ENC_IN_ZP);
    vaeEncoder.addOutputTensor(1285, "latent_output", QNN_DATATYPE_UFIXED_POINT_16, {1, 64, 64, 4}, VAE_ENC_OUT_SCALE, VAE_ENC_OUT_ZP);
    STAGE("init vae encoder");
    if (!vaeEncoder.init(rt, "vae_encoder", "models/vae_encoder_htp.serialized.bin", "vae_encoder")) return -1;

    STAGE("validate unet io");
    int iTs = -1, iSample = -1, iEmb = -1, oOut = -1;
    for (size_t i = 0; i < unet.numIn(); ++i) {
        uint64_t n = unet.inElems(i);
        if (n == (uint64_t)B * sample16Size)           iSample = (int)i;
        else if (n == (uint64_t)B * textEmbElements)   iEmb = (int)i;
        else if (n == (uint64_t)B)                     iTs = (int)i;
    }
    for (size_t i = 0; i < unet.numOut(); ++i)
        if (unet.outElems(i) == (uint64_t)B * latent4Size) oOut = (int)i;
    if (iTs < 0 || iSample < 0 || iEmb < 0 || oOut < 0 || unet.numIn() != 3 || unet.numOut() != 1) {
        std::cerr << "[FATAL] UNet I/O doesn't match expectations for UNET_BATCH=" << B
                  << " (need in: timestep[" << B << "], sample[" << B << ",16,64,64], encoder_hidden_states[" << B
                  << ",77,768]; out: [" << B << ",4,64,64])." << std::endl;
        return -1;
    }
    if (unet.inType(iTs) != QNN_DATATYPE_FLOAT_32 || unet.inType(iSample) != QNN_DATATYPE_FLOAT_32 ||
        unet.inType(iEmb) != QNN_DATATYPE_FLOAT_32 || unet.outType(oOut) != QNN_DATATYPE_FLOAT_32) {
        std::cerr << "[FATAL] UNet binary uses non-FLOAT32 I/O." << std::endl;
        return -1;
    }

    STAGE("alloc + register unet buffers");
    SharedBuf sSample, sTs, sEmb[2], sOut[2];
    const size_t embBytes = textEmbElements * sizeof(float), latBytes = latent4Size * sizeof(float);
    bool allocOk = rpc.alloc(sSample, B * sample16Size * sizeof(float)) && rpc.alloc(sTs, 64);
    if (B == 2) allocOk = allocOk && rpc.alloc(sEmb[0], 2 * embBytes) && rpc.alloc(sOut[0], 2 * latBytes);
    else        allocOk = allocOk && rpc.alloc(sEmb[0], embBytes) && rpc.alloc(sEmb[1], embBytes)
                                  && rpc.alloc(sOut[0], latBytes) && rpc.alloc(sOut[1], latBytes);
    if (!allocOk) { std::cerr << "[FATAL] rpcmem allocation failed" << std::endl; return -1; }
    sTs.bytes = B * sizeof(float);

    IoRef rTs = unet.registerBuf(true, iTs, sTs, zeroCopy);
    IoRef rSample = unet.registerBuf(true, iSample, sSample, zeroCopy);
    IoRef rEmb[2], rOut[2];
    rEmb[0] = unet.registerBuf(true, iEmb, sEmb[0], zeroCopy);
    rOut[0] = unet.registerBuf(false, oOut, sOut[0], zeroCopy);

    // If B == 1, assign rEmb[1] and rOut[1] as RAW to prevent duplicate registration on the same tensor slot
    if (B == 1) {
        rEmb[1].ptr = sEmb[1].ptr; rEmb[1].bytes = sEmb[1].bytes; rEmb[1].h = nullptr;
        rOut[1].ptr = sOut[1].ptr; rOut[1].bytes = sOut[1].bytes; rOut[1].h = nullptr;
    }

    unet.select(true, iTs, rTs);
    unet.select(true, iSample, rSample);

    float* lat  = sSample.f32();
    float* lat2 = (B == 2) ? sSample.f32() + sample16Size : nullptr;

    STAGE("read image/mask");
    std::vector<float> imageBuf, maskBuf512, maskBuf64;
    if (!readRawFloats("image.raw", imageBuf, imagePixelCount)) { std::cerr << "image.raw (3145728 B) failed\n"; return -1; }
    if (!readRawFloats("mask.raw", maskBuf512, maskPixelCount)) { std::cerr << "mask.raw (1048576 B) failed\n"; return -1; }
    downsampleMaskNearest(maskBuf512, maskBuf64);

    auto totalStart = Clock::now();

    STAGE("vae encode");
    auto t0 = Clock::now();
    std::vector<uint16_t> ufixMasked(imagePixelCount);
    for (size_t i = 0; i < maskPixelCount; ++i) {
        const bool masked = maskBuf512[i] > 0.5f;
        for (int c = 0; c < 3; ++c) {
            size_t idx = i * 3 + c;
            float v = masked ? 0.0f : (imageBuf[idx] * 2.0f - 1.0f);
            ufixMasked[idx] = quantizeToUFix16(v, VAE_ENC_IN_SCALE, VAE_ENC_IN_ZP);
        }
    }
    {
        std::vector<uint16_t> raw(latent4Size);
        if (!vaeEncoder.execute({ufixMasked.data()}, {ufixMasked.size() * 2}, {raw.data()}, {latent4Size * 2})) return -1;
        for (int b = 0; b < B; ++b) {
            float* s = sSample.f32() + b * sample16Size;
            std::memcpy(s + 4 * HW, maskBuf64.data(), HW * sizeof(float));
            for (size_t p = 0; p < HW; ++p)
                for (size_t c = 0; c < 4; ++c)
                    s[(5 + c) * HW + p] = applyVaeEncScale(dequantizeFromUFix16(raw[p * 4 + c], VAE_ENC_OUT_SCALE, VAE_ENC_OUT_ZP));
            std::memset(s + 9 * HW, 0, 7 * HW * sizeof(float));
        }
    }
    std::cout << "[Profile] VAE Encoder: " << msSince(t0) << " ms" << std::endl;

    STAGE("text encode");
    t0 = Clock::now();
    CLIPTokenizer tokenizer;
    if (!tokenizer.load("vocab.json", "merges.txt")) { std::cerr << "tokenizer load failed\n"; return -1; }
    std::string userPrompt = (argc > 1) ? argv[1] : "photograph of a beautiful empty scene, highest quality settings";
    std::string negPrompt  = "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured";
    std::vector<int32_t> pTok = tokenizer.encode(userPrompt), uTok = tokenizer.encode(negPrompt);
    pTok.resize(77, pTok.empty() ? 0 : pTok.back()); uTok.resize(77, uTok.empty() ? 0 : uTok.back());
    std::vector<uint16_t> cRaw(textEmbElements), uRaw(textEmbElements);
    if (!textEncoder.execute({pTok.data()}, {pTok.size() * 4}, {cRaw.data()}, {cRaw.size() * 2})) return -1;
    if (!textEncoder.execute({uTok.data()}, {uTok.size() * 4}, {uRaw.data()}, {uRaw.size() * 2})) return -1;
    float* embU = sEmb[0].f32();
    float* embC = (B == 2) ? sEmb[0].f32() + textEmbElements : sEmb[1].f32();
    for (size_t i = 0; i < textEmbElements; ++i) {
        embU[i] = dequantizeFromUFix16(uRaw[i], TE_OUT_SCALE, TE_OUT_ZP);
        embC[i] = dequantizeFromUFix16(cRaw[i], TE_OUT_SCALE, TE_OUT_ZP);
    }
    std::cout << "[Profile] Text Encoder: " << msSince(t0) << " ms" << std::endl;

    float* noiseU = sOut[0].f32();
    float* noiseC = (B == 2) ? sOut[0].f32() + latent4Size : sOut[1].f32();

    STAGE("schedule + noise");
    DPMSolverSchedule schedule(numSteps);
    {
        std::mt19937 rng(42); std::normal_distribution<float> g(0.0f, 1.0f);
        for (size_t i = 0; i < latent4Size; ++i) lat[i] = g(rng) * schedule.init_noise_sigma;
        if (lat2) std::memcpy(lat2, lat, latBytes);
    }
    std::vector<float> prev_d(latent4Size, 0.0f);

    STAGE("unet warm-up");
    {
        auto w0 = Clock::now();
        sTs.f32()[0] = schedule.timesteps[0]; if (B == 2) sTs.f32()[1] = schedule.timesteps[0];
        unet.select(true, iEmb, rEmb[0]); unet.select(false, oOut, rOut[0]);
        if (!unet.executeBound()) return -1;
        std::cout << "[Profile] UNet warm-up execute: " << msSince(w0) << " ms (discarded; output ignored)" << std::endl;
    }

    STAGE("denoise loop");
    auto denoiseStart = Clock::now();
    for (int step = 0; step < numSteps; ++step) {
        auto s0 = Clock::now();
        const float ts = schedule.timesteps[step];
        sTs.f32()[0] = ts; if (B == 2) sTs.f32()[1] = ts;

        double tU = 0, tC = 0;
        if (B == 2) {
            unet.select(true, iEmb, rEmb[0]); unet.select(false, oOut, rOut[0]);
            auto e0 = Clock::now(); if (!unet.executeBound()) return -1; tU = msSince(e0);
        } else {
            unet.select(true, iEmb, rEmb[0]); unet.select(false, oOut, rOut[0]);   // uncond
            auto e0 = Clock::now(); if (!unet.executeBound()) return -1; tU = msSince(e0);
            unet.select(true, iEmb, rEmb[1]); unet.select(false, oOut, rOut[1]);   // cond
            auto e1 = Clock::now(); if (!unet.executeBound()) return -1; tC = msSince(e1);
        }
        if (doProfile && step == 1) unet.dumpProfile("UNet step 1");

        auto h0 = Clock::now();
        fusedStep(lat, lat2, noiseU, noiseC, prev_d.data(), latent4Size, guidanceScale, makeCoef(schedule, step));
        double tHost = msSince(h0);

        std::cout << "[Step " << (step + 1) << "/" << numSteps << "] t=" << ts
                  << " | unet(u)=" << tU << " ms, unet(c)=" << tC << " ms, host-math=" << tHost
                  << " ms, total=" << msSince(s0) << " ms" << std::endl;
    }
    std::cout << "[Profile] Total Denoising Loop: " << msSince(denoiseStart) << " ms" << std::endl;

    STAGE("vae decode");
    t0 = Clock::now();
    std::vector<uint16_t> ufixLat(latent4Size), ufixRgb(imagePixelCount);
    for (size_t p = 0; p < HW; ++p)
        for (size_t c = 0; c < 4; ++c)
            ufixLat[p * 4 + c] = quantizeToUFix16(applyVaeDecScale(lat[c * HW + p]), VAE_LATENT_IN_SCALE, VAE_LATENT_IN_ZP);
    if (!vaeDecoder.execute({ufixLat.data()}, {latent4Size * 2}, {ufixRgb.data()}, {imagePixelCount * 2})) return -1;

    std::vector<uint8_t> out(imagePixelCount);
    for (size_t i = 0; i < maskPixelCount; ++i) {
        const bool m = maskBuf512[i] > 0.5f;
        for (int c = 0; c < 3; ++c) {
            size_t idx = i * 3 + c;
            float v = m ? dequantizeFromUFix16(ufixRgb[idx], VAE_IMAGE_OUT_SCALE, VAE_IMAGE_OUT_ZP) * 255.0f
                        : imageBuf[idx] * 255.0f;
            out[idx] = static_cast<uint8_t>(std::clamp(v, 0.0f, 255.0f));
        }
    }
    std::cout << "[Profile] VAE Decoder & Composite: " << msSince(t0) << " ms" << std::endl;

    stbi_write_png(outputPath.c_str(), 512, 512, 3, out.data(), 512 * 3);
    std::cout << "Saved " << outputPath << " | end-to-end " << msSince(totalStart) << " ms" << std::endl;
    return 0;
}