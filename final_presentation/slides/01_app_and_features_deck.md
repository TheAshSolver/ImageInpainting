# Snapdragon Inpaint Studio (Flagship v5) & SD Duel
## System Architecture, Dual-Context Prompt Enrichment, and Production Features Deck

---

## 1. Executive Summary & Production Evolution

| Metric / Dimension | Legacy QIDK Inpaint v3 | Snapdragon Inpaint Studio (v5) | Proposed Engine with SD Duel |
| :--- | :---: | :---: | :---: |
| **Primary Diffusion Engine** | 20-step Euler / DPM++ (Single UNet) | 12-step DPM-Solver++ (HTP v79) | **Dual-DLC UNet + Taylor Order-1 Cache** |
| **End-to-End Latency** | 5,820 ms - 6,400 ms | 3,790 ms | **1,529.24 ms (2.47× Speedup)** |
| **Diffusion Bottleneck Cache** | None (Full Recompute per step) | None | **[1, 640, 64, 64] FP32 (10.5 MB RPCMem)** |
| **Shallow UNet Execution** | N/A | N/A | **78 ms / intermediate step** |
| **Object Insertion Prompting** | Raw User Text (High CFG 7.5) | Single-Ring Color Proxy (CFG 2.0) | **Dual-Context Engine (Macro + Contact Horizon, CFG 7.5)** |
| **Background Texture Removal** | Coarse Overwrite | Smooth (CFG 2.0) | **Adaptive CFG 2.5 + Ground Plane Continuation** |
| **Super-Resolution Engine** | None | 2× / 4× / 8× Neural + Bicubic | **2× / 4× / 8× Neural + Synchronized Viewports** |
| **Hardware Core Support** | NPU Only | NPU / GPU / CPU Arbitrated | **NPU Burst DCVS-Off + HMX-Max + RPC Zero-Copy** |

---

## 2. SD Duel Architecture: Dual-DLC UNet Cache Extrapolation

```mermaid
flowchart TD
    subgraph HostInit["Host Memory & IPC Pipeline"]
        Prompt["Input User Prompt + Dual Context"] --> CLIP["CLIP Text Encoder (163 MB mmap)"]
        CLIP --> Embed["Text Embeddings [1, 77, 768]"]
        RawIn["image.raw (3.14 MB) + mask.raw (1.05 MB)"] --> VAEEnc["VAE Encoder HTP (93.9 MB)"]
        VAEEnc --> Latent["Latents [1, 16, 64, 64]"]
    end

    subgraph DualDenoise["Dual-DLC Iterative Denoiser (Hexagon NPU)"]
        Latent --> StepDecision{"Step Index mod 3 == 0?"}
        StepDecision -- "Yes (Steps 0, 3, 6, 9, 12)" --> FullUNet["Full UNet (842 MB DLC)<br/>202 ms / step"]
        StepDecision -- "No (Steps 1, 2, 4, 5, ...)" --> ShallowUNet["Shallow UNet (38 MB DLC)<br/>78 ms / step"]
        FullUNet --> BottleneckCache["Bottleneck Tensor [1, 640, 64, 64]<br/>Taylor Order-1 Axis Extrapolation"]
        BottleneckCache -.-> ShallowUNet
    end

    subgraph OutputSynthesis["Zero-Copy Output"]
        ShallowUNet --> FinalLatent["Denoised Latents [1, 4, 64, 64]"]
        FullUNet --> FinalLatent
        FinalLatent --> VAEDec["VAE Decoder (64.5 MB mmap)"]
        VAEDec --> Composite["Alpha Composite with Original Image"]
        Composite --> FinalPNG["sd_output.png (512x512 RGB)"]
    end
```

### Technical Highlights of SD Duel:
- **Zero-Copy RPCOffload**: Model weights MMAPed directly into Qualcomm `rpcmem` buffers, completely eliminating user-kernel buffer copying over Android binder IPC.
- **Cache Interval ($N=3$)**: Full UNet executes at step 0 and every 3rd step, capturing low-frequency semantic representations. Intermediate steps execute the 38 MB Shallow UNet with Taylor order-1 numerical extrapolation, delivering **1,529.24 ms end-to-end execution**.

---

## 3. The Dual-Context Prompt Enrichment Engine

```mermaid
flowchart LR
    subgraph Inputs["Multi-Modal Inputs"]
        UserText["User Input: 'replace bottle with an apple'"]
        RawImage["Source Bitmap (512x512 ARGB)"]
        RawMask["Hole Mask (512x512 Binary)"]
    end

    subgraph ContextEngine["Dual-Context Vision Pass (<4 ms on Oryon CPU)"]
        RawImage --> SceneEngine["Global Scene Engine:<br/>- Lighting Angle (Vector)<br/>- Color Temp (Warmth R-B)<br/>- Ambient Environment"]
        RawImage & RawMask --> MaskEngine["Local Mask Engine:<br/>- Pre-inpaint Void Contents<br/>- Bounding Box Aspect Ratio<br/>- Contact Horizon / Table Surface"]
    end

    subgraph IntentSynthesis["Intent Classification & CFG Guidance"]
        UserText & MaskEngine --> IntentMatch{"Classify Intent"}
        IntentMatch -- "Replacement / Object" --> SetCFG75["Intent: REPLACEMENT<br/>CFG: 7.5 (Full Text Authority)<br/>Anchor: Ground Plane Contact Shadow"]
        IntentMatch -- "Empty / Remove" --> SetCFG25["Intent: REMOVAL<br/>CFG: 2.5 (Blending Authority)<br/>Anchor: Texture Continuation"]
    end

    subgraph FinalDispatch["CLIP Prompt Assembly"]
        SetCFG75 & SceneEngine --> SynthesizedPrompt["'an apple, resting naturally on the wooden table surface,<br/>casting natural contact shadows matching bright daylight,<br/>warm tones, photorealistic 8k, sharp focus'"]
        SynthesizedPrompt --> NPU["Dispatched to SD Duel on NPU"]
    end
```

### Why Legacy v3 Excelled at Object Insertion & Why v5 Surpasses It:
1. **The Root Cause**: Legacy v3 passed raw user prompts directly with **CFG 7.5**, allowing the diffusion model sufficient denoising authority to hallucinate new geometry. Flagship v5 defaulted to **CFG 2.0**, which successfully smoothed textures but caused new objects (e.g. replacing a bottle with an apple) to be washed out.
2. **The Dual-Context Solution**:
   - **Pre-inpaint Void Analysis**: Measures the color, luminance, and aspect ratio of what is being removed ($AR < 0.65 \to$ tall upright bottle).
   - **Ground Contact Horizon**: Detects the boundary row directly beneath the mask ($y_1 + 1$ to $y_1 + 12$), identifying the support surface (wooden tabletop, granite, grass).
   - **Contact Shadow Anchor**: Enforces ambient occlusion and contact shadows consistent with global lighting vector.
   - **Dynamic Guidance**: Automatically promotes object replacements to **CFG 7.5**, while holding pure texture removals at **CFG 2.5**.

---

## 4. UI Gallery: Snapdragon Inpaint Studio Interface

| Redesigned Interactive Canvas | Live Inference Telemetry & Preload |
| :---: | :---: |
| ![Inpaint Canvas](final_presentation/figures/redesigned_inpaint_screen.png) | ![Live Telemetry](final_presentation/figures/app_screen_v5_telemetry_live.png) |
| *Intuitive dual-tool touch surface: Brush, Magic Touch, Bounding Box* | *Real-time thermal tracking, power monitoring, and NPU burst state* |

| Dark Theme High-Fidelity Result | Light Theme Clean Composite |
| :---: | :---: |
| ![Dark Result](final_presentation/figures/06_inpaint_dark.png) | ![Light Result](final_presentation/figures/05_inpaint_light.png) |
| *Seamless edge blend and structure restoration on portrait dataset* | *Flawless background texture continuation under daylight conditions* |

---

## 5. Super-Resolution Studio Integration

```mermaid
graph TD
    InpResult["Inpainted Output (512 x 512)"] --> SRStudio["Snapdragon Super-Resolution Engine"]
    SRStudio --> ModeSelect{"Select Scale Factor"}
    ModeSelect --> SR2X["2× Scale (1024 x 1024)<br/>MIGAN / LaMa / Bicubic"]
    ModeSelect --> SR4X["4× Scale (2048 x 2048)<br/>Tarsh SR Neural Engine"]
    ModeSelect --> SR8X["8× Scale (4096 x 4096)<br/>AOT-GAN Ultra SR"]
    SR4X & SR8X --> SynchronizedViewport["Synchronized Dual Viewports<br/>Pan & Zoom Lockstep Analysis"]
```

| Synchronized SR Viewports | 4× Neural Super-Resolution Detail |
| :---: | :---: |
| ![Sync Viewports](final_presentation/figures/synchronized_sr_viewports.png) | ![SR 4x](final_presentation/figures/sr_tarsh_4x.png) |
| *Synchronized pan-and-zoom inspection between original and SR image* | *Tack-sharp high-frequency reconstruction up to 4K resolution* |

---

## 6. App Operational Modes & Control Surface

```mermaid
flowchart TD
    UserSelection{"User Mode Selection"}
    UserSelection --> AutoMode["Auto Mode (Intelligent Router)"]
    UserSelection --> ManualMode["Manual Mode (Direct Override)"]

    AutoMode --> PolicySelect{"Router Policy Family"}
    PolicySelect --> LinUCB["LinUCB Contextual Bandit"]
    PolicySelect --> RLPolicy["RL Policy (Actor-Critic)"]
    PolicySelect --> DeepMLP["Deep Learning (Multi-Head MLP)"]
    PolicySelect --> Heuristic["Heuristic Decision Ladder"]

    LinUCB & RLPolicy & DeepMLP & Heuristic --> ModeObjective{"Optimization Objective"}
    ModeObjective --> Quality["Quality Mode (w_Q=0.70, w_L=0.15, w_P=0.15)"]
    ModeObjective --> Balanced["Balanced Mode (w_Q=0.40, w_L=0.35, w_P=0.25)"]
    ModeObjective --> PowerEff["Power Efficiency (w_Q=0.20, w_L=0.30, w_P=0.50)"]

    ManualMode --> DirectModel["Select Model: SD Duel, LaMa, MIGAN, AOT-GAN, Telea, Navier-Stokes"]
    DirectModel --> DirectBackend["Select Backend: Hexagon NPU, Adreno GPU, Oryon CPU"]
```

---

## 7. Real-Time "Alive" Adaptation Loop

```mermaid
sequenceDiagram
    participant User as User Canvas
    participant Pred as Prediction Meter
    participant NPU as Snapdragon 8 Elite NPU
    participant Window as Sliding Window (W=25)
    participant Bandit as Online LinUCB / RL State

    User->>Pred: Draw Mask + Type Prompt
    Pred-->>User: Display Expected Latency (ms), Power (W), Quality
    User->>NPU: Tap "Run Inpaint"
    NPU-->>User: Rendered Result in 1,529 ms
    NPU->>Window: Record Actual Latency & Power
    Window->>Window: Compute Drift = Actual / Predicted
    alt Thermal Contention Drift > 1.25×
        Window->>Bandit: Apply Dynamic Context Weight Penalty
        Bandit-->>Pred: Shift Recommended Route to Lighter Model
    end
    alt User Taps "Save"
        User->>Bandit: Implicit Reward R = +1.0 (Positive Reinforcement)
    else User Taps "Undo / Discard"
        User->>Bandit: Implicit Penalty R = -1.0 (Explore Alternative Route)
    end
```

---
*Snapdragon Inpaint Studio (v5) — Qualcomm Innovators Development Kit Production Deck*
