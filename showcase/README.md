# QIDK Snapdragon 8 Elite Contention & Thermal Load Engine

**Platform**: Qualcomm Snapdragon 8 Elite (SM8750 / `"sun"`)  
**Target Hardware**: HDK Development Board (`8f27557f`) running Android 15 (API 35)

---

## 1. Overview & Capabilities

The `showcase/loadgen.py` tool provides high-fidelity, closed-loop multi-resource contention generation and closed-loop thermal regulation on the physical Snapdragon 8 Elite development platform.

### Monitored Subsystems & Hardware Mechanisms
1. **CPU Subsystem (Oryon 8-Core)**:
   - 2 Prime cores (up to 4.32 GHz) + 6 Performance cores (up to 3.53 GHz).
   - Real-time closed-loop Proportional-Integral (PID) duty cycle governor measuring `/proc/stat` delta every 200 ms.
2. **GPU Subsystem (Adreno 830)**:
   - Dynamic OpenCL (`/vendor/lib64/libOpenCL.so`) compute kernel driving compute units.
   - Real-time utilization telemetry from `/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage`.
3. **NPU Subsystem (Hexagon v79 HTP)**:
   - Direct execution of real neural network workloads on the Hexagon NPU via Qualcomm SNPE v2.49 (`migan_htp_v79.dlc`).
   - Duty cycle PWM inference scheduling tracking real inference latency (~330 ms) and thermal response across Hexagon HVX/HMX thermal zones (48..54).
4. **Memory Subsystem (LPDDR5x)**:
   - Multi-threaded streaming cacheline thrashers with non-temporal writes.
   - Resident Physical Memory (`VmRSS`) tracked directly via `/proc/self/status`.
   - Independent Memory Controller Core (MCCC) hardware clock telemetry via `/sys/kernel/debug/clk/measure_only_mccc_clk/clk_measure` (547 MHz idle up to 4.76 GHz loaded).
5. **Closed-Loop Thermal Regulation**:
   - Monitored on Zone 18 (`cpuss-0-0`, Oryon CPU Subsystem).
   - Holds chosen temperature targets (Cool 35°C, Warm 42°C, Hot 50°C).
   - **Hard Emergency Safety Cutoff at 85.0°C**: Instantly aborts compute workers to prevent 90°C+ thermal runaway and hardware trips.
6. **Power Provenance**:
   - Tagged strictly as `DERIVED_MODEL` based on the Snapdragon 8 Elite EAS energy model:
     $$P_{\text{total}} = 1.35\text{ W} + 4.65\text{ W} \times \frac{U_{\text{cpu}}}{100} + 3.10\text{ W} \times \frac{U_{\text{gpu}}}{100} + 0.08\text{ W/(GB/s)} \times \text{BW}$$
   - External empirical power is marked as `BLOCKED: EXTERNAL_POWER_METER` pending inline physical analyzer connection.

---

## 2. Command Line Usage

### A. Preflight Hardware Check
Verifies ADB connection, SoC platform, native binary permissions, OpenCL Adreno driver, Hexagon DSP SNPE runtime, MCCC clock access, and thermal zones:
```bash
python showcase/loadgen.py --check
```

### B. Preset Contention States
Executes one of the standard research contention levels ($L_0$ to $L_4$) with target temperature holding:
```bash
# Level 2 Contention (35% CPU, 32% GPU, 30% NPU, 2560 MB RAM) held at 42°C:
python showcase/loadgen.py --level L2 --temp 42 --duration 15

# Level 3 Contention (65% CPU, 60% GPU, 60% NPU, 4096 MB RAM) held at 50°C:
python showcase/loadgen.py --level L3 --temp 50 --duration 20
```

### C. Custom Multi-Resource & Live Terminal Dashboard
Allows specifying individual subsystem targets with a live terminal status dashboard:
```bash
python showcase/loadgen.py \
    --cpu 60 \
    --gpu 40 \
    --npu 30 \
    --ram 3072 \
    --membw 15 \
    --temp 50 \
    --max-temp 85 \
    --duration 30 \
    --dashboard
```

---

## 3. Contention Levels Preset Reference

| Level | State Name | Target CPU | Target GPU | Target NPU | Target RAM RSS | Target DDR BW | Thermal Target |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **L0** | Quiescent Idle | 0% | 0% | 0% | 0 MB | 0.0 GB/s | 35.0°C |
| **L1** | Light Contention | 15% | 12% | 10% | 1024 MB | ~2.7 GB/s | 35.0°C |
| **L2** | Balanced Load | 35% | 32% | 30% | 2560 MB | ~10.0 GB/s | 42.0°C |
| **L3** | Heavy Contention | 65% | 60% | 60% | 4096 MB | ~19.0 GB/s | 50.0°C |
| **L4** | Extreme Thrashing | 85% | 85% | 85% | 4096 MB | ~27.0 GB/s | 50.0°C (Cutoff 85°C) |

---

## 4. Safety & Clean Interrupts
- The tool registers POSIX signal handlers (`SIGINT`, `SIGTERM`).
- Hitting `Ctrl+C` immediately triggers a device cleanup trap:
  ```bash
  adb shell "killall -9 qidk_load_generator snpe-net-run 2>/dev/null"
  ```
- This guarantees zero zombie threads, frees all allocated DDR buffer pools, and restores Oryon core clock governance.
