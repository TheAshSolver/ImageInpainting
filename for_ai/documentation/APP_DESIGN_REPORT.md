# Comprehensive App Design & UI/UX Architecture Report
## Qualcomm Snapdragon 8 Elite Neural Inpainting & Super-Resolution Suite (`qidk-inpaint-app-v5`)

---

## Executive Summary

The **Snapdragon 8 Elite Inpainting & Super-Resolution Suite** is engineered with a **cyberpunk, developer-console aesthetic** that prioritizes high information density, instantaneous hardware feedback, and military-grade contrast. Built specifically for on-device AI researchers, hardware engineers, and creative professionals, the design bridges low-level silicon performance metrics (FastRPC socket telemetry, temperature sensors, battery fuel gauges, NPU cycle counters) with tactile, consumer-grade image manipulation tools (Circle-to-Select Magic Touch, GrabCut segmentation, multi-touch pinch-to-zoom, and synchronized Before/After viewports).

The application features a fully reactive **Dual-Theme Engine** (Dark Mode default with a high-contrast Light Mode alternate), custom hardware-accelerated canvas components, letterboxed viewport coordinate normalization, and visual feedback micro-animations.

---

## 1. Design System Philosophy & Principles

```
+---------------------------------------------------------------------------------------------------------+
|                                    CORE DESIGN PILLARS                                                  |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|  1. Dark-First Developer Console Aesthetic                                                              |
|     • Defaults to deep obsidian slate (#0B0F19) and navy surfaces (#161F30).                            |
|     • Eliminates eye strain during prolonged benchmarking and on-device debugging sessions.             |
|     • Delivers maximum contrast for vibrant AI masks (translucent crimson, neon cyan, lavender).        |
|                                                                                                         |
|  2. Real-Time Hardware Transparency (Zero Hidden Latency)                                              |
|     • Telemetry is never tucked into secondary menus; it sits front-and-center.                         |
|     • Every inference run displays silicon temperature (ΔT), active wattage, FastRPC socket duration,  |
|       and memory residency in dedicated HUD pill badges.                                                |
|                                                                                                         |
|  3. Mathematical Separation of Screen Viewport and Bitmap Coordinates                                    |
|     • Screen space ≠ Bitmap space. All canvas interactions are mediated by a unified ViewportMapper    |
|       that letterboxes the canonical 512×512 tensor space with sub-pixel floating-point accuracy.       |
|                                                                                                         |
|  4. Multi-Modal Tactile & Visual Feedback                                                               |
|     • High-frequency glowing touch trails (18 px outer pulse, 7 px solid core).                         |
|     • Pulsing radar ripple animations during asynchronous GrabCut/MediaPipe segmentation.               |
|     • Synchronized lockstep viewport navigation for pixel-level before/after verification.              |
+---------------------------------------------------------------------------------------------------------+
```

---

## 2. Complete Color Palette & Token System

The design system uses a semantic token architecture. Color values are defined programmatically with exact alpha layers and contrast ratios compliant with **WCAG 2.1 AA (4.5:1 text contrast)**.

### 2.1 Primary Theme Palettes (Dark vs. Light)

| Semantic Token | Dark Mode Hex | Light Mode Hex | Visual Purpose & UI Application |
| :--- | :---: | :---: | :--- |
| `background_canvas` | `#0B0F19` | `#F8FAFC` | Root layout background, unmasked canvas letterboxing |
| `surface_card` | `#161F30` | `#FFFFFF` | Elevated container cards (Control, Telemetry, Router, HUD) |
| `surface_input` | `#1E293B` | `#F1F5F9` | Recessed input fields (Prompt EditTexts, Chip bars) |
| `border_subtle` | `#1E293B` | `#CBD5E1` | Card outlines (1 dp stroke), divider rules |
| `text_primary` | `#F8FAFC` | `#0F172A` | Card titles, action button labels, execution metrics |
| `text_secondary` | `#94A3B8` | `#64748B` | Subtitles, field hints, hardware descriptions |
| `toggle_surface` | `#161F30` | `#E2E8F0` | Day/Night mode circular toggle button surface |

```
DARK THEME (Default)                      LIGHT THEME (High Contrast)
┌───────────────────────────────┐        ┌───────────────────────────────┐
│ #0B0F19 Root Background       │        │ #F8FAFC Root Background       │
│  ┌─────────────────────────┐  │        │  ┌─────────────────────────┐  │
│  │ #161F30 Surface Card    │  │        │  │ #FFFFFF Surface Card    │  │
│  │  Border: #1E293B        │  │        │  │  Border: #CBD5E1        │  │
│  │  Text: #F8FAFC          │  │        │  │  Text: #0F172A          │  │
│  │  Input: #1E293B         │  │        │  │  Input: #F1F5F9         │  │
│  └─────────────────────────┘  │        │  └─────────────────────────┘  │
└───────────────────────────────┘        └───────────────────────────────┘
```

---

### 2.2 Semantic Accent & Status Palette

| Accent Color Name | Hex Code | Alpha / Variant | Visual Purpose & Trigger Conditions |
| :--- | :---: | :---: | :--- |
| **Electric Cyan** | `#38BDF8` | `#0284C7` (Dark) | Primary Action Buttons (Upscale, Run Inpaint), Active Tab indicators, Zoom level HUD badges |
| **Emerald Green** | `#10B981` | `#34D399` (Light) | Successful verification badges, Nominal thermal state ($<55^\circ\text{C}$), Preload ready status |
| **Neon Lavender / Purple** | `#C084FC` | `#A855F7` (Glow) | AI Magic Touch lasso boundary, GrabCut thinking pulse, Execution target badge |
| **Amber Warning** | `#F59E0B` | `#D97706` (Border)| Elevated thermal regime ($55^\circ\text{C}\text{--}69^\circ\text{C}$), Large mask coverage warning |
| **Crimson Coral** | `#EF4444` | `#F87171` (Light) | Critical thermal throttling ($T \ge 70^\circ\text{C}$), Error alerts, Process termination |
| **Cobalt Blue** | `#2563EB` | `#1D4ED8` (Pressed)| Active tool mode selection (Brush / Smart / BBox selected state) |
| **Translucent Mask Red** | `rgba(255, 40, 40, 0.47)` | `#78FF2828` | On-screen inpainting hole overlay rendered over canonical photo |
| **Lasso Fill Tint** | `rgba(192, 132, 252, 0.24)` | `#3DC084FC` | Live polygon fill preview when user draws Circle-to-Select lasso |

---

## 3. Typography & Hierarchy

The typography system pairs **modern geometric sans-serif** for UI navigation with **fixed-width monospace** for all numerical and hardware telemetry data.

```
TYPE SCALE HIERARCHY
┌──────────────────────────────────────────────────────────────────────────────────┐
│ DISPLAY / HEADER:  18 sp Bold  • Roboto / Segoe UI • #F8FAFC                     │
│ "⚡ Neural Inpainting Suite"                                                     │
├──────────────────────────────────────────────────────────────────────────────────┤
│ SECTION TITLE:     14 sp Bold  • #38BDF8 (Cyan Accent)                           │
│ "⚙️ Execution Target & Backend"                                                  │
├──────────────────────────────────────────────────────────────────────────────────┤
│ BODY / LABELS:     12 sp - 13 sp Regular • #F8FAFC / #94A3B8                     │
│ "Prompt Guidance: Smooth natural texture, indoor studio lighting"                │
├──────────────────────────────────────────────────────────────────────────────────┤
│ HARDWARE HUD:      11 sp - 12 sp MONOSPACE • #34D399 (Green) / #38BDF8           │
│ "Output 1024x1024 | 3841 ms | FastRPC Daemon | Hexagon HTP v79 | 2.58 W"        │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Font Configurations
* **Primary Sans-Serif (`Typeface.DEFAULT_BOLD` / `Typeface.DEFAULT`)**:
  * Used for: App title, Card headers, Button text, Mode toggles, Dropdown spinners.
  * Line-height: $1.35\times$ font size.
* **Telemetry Monospace (`Typeface.MONOSPACE`)**:
  * Used for: Real-time clock frequencies, Latency breakdowns (`prepMs`, `inferMs`, `postMs`), Voltage/Current readings (`633 mA`, `8.91 V`), Zoom magnification levels (`🔍 x2.4`).
  * Prevents character jitter when numbers fluctuate rapidly during 20 Hz sysfs polling.

---

## 4. Layout Architecture & Screen Real Estate

The app uses a **tabbed single-activity architecture** (`MainActivity`) with a dual-page primary switch:
1. `🎨 INPAINT`: High-speed on-device contextual infill, GrabCut touch lasso, and prompt-guided diffusion.
2. `⚡ SUPER RESOLUTION`: Multi-scale ($2\times, 4\times, 8\times$) sub-pixel tiled neural upscaling with dual synchronized compare viewports.

```
SCREEN COMPOSITION (INPAINT TAB)
┌────────────────────────────────────────────────────────┐
│ [🎨 INPAINT]  [⚡ SUPER RESOLUTION]                     │ ← Primary Top Navigation Tabs
├────────────────────────────────────────────────────────┤
│ ⚡ Neural Inpainting Suite                  [🌙 Mode]  │ ← Header Card & Quick Toggle
│ Qualcomm Snapdragon 8 Elite • Hexagon HTP v79          │
├────────────────────────────────────────────────────────┤
│ [📁 Gallery]    [📷 Camera]    [🖼️ Load Sample (102)]   │ ← Source Input Action Bar
├────────────────────────────────────────────────────────┤
│ ┌────────────────────────────────────────────────────┐ │
│ │                                                    │ │
│ │              INPAINT CANVAS VIEW                   │ │
│ │         (Uniform Letterbox 512×512)                │ │
│ │         • Translucent Red Mask Overlay             │ │ ← Interactive Viewport
│ │         • Real-time Glowing Trail Finger HUD       │ │   (Fixed Aspect Ratio)
│ │         • Animated GrabCut Thinking Pulse          │ │
│ │                                                    │ │
│ └────────────────────────────────────────────────────┘ │
│ [🔍 In] [🔍 Out] [↺ Reset] [🖌️ Brush] [✨ Smart] [🔲 Box]│ ← Floating Canvas Toolstrip
├────────────────────────────────────────────────────────┤
│ 🎯 RECOMMENDED: MIGAN (Portrait Rule Triggered)        │ ← LinUCB Contextual Bandit
│ Faces: 1 | Void: 8.2% | Lat: 59.3 ms | Power: 31 mW    │   Decision HUD
├────────────────────────────────────────────────────────┤
│ 💡 Prompt Guidance (Stable Diffusion 1.5)              │ ← Text Conditioning Card
│ [Clean Background] [Wood Grain] [Foliage] [Water]      │   (Quick Suggestion Chips)
├────────────────────────────────────────────────────────┤
│ ▼ Pipeline Vision-Language Trace (5 Stages)            │ ← Collapsible Debug Drawer
├────────────────────────────────────────────────────────┤
│ ⚙️ Execution Target: [ MIGAN Neural SR ▼ ] [ NPU ▼ ]    │ ← Hardware Spinner Bar
│ [🚀 RUN INPAINT (Hexagon NPU)]                         │ ← Primary Action Button
├────────────────────────────────────────────────────────┤
│ 📊 FastRPC Hardware Telemetry HUD Card                 │
│ 59.3 ms • 31.0 mW • 2.88 mJ • 36.4°C • Battery: 100%   │ ← Post-Run Hardware Card
└────────────────────────────────────────────────────────┘
```

---

## 5. Detailed Component Breakdown

### 5.1 `InpaintCanvasView` (Interactive Masking Engine)

```
VIEWPORT COORDINATE MAPPING (ViewportMapper)
Touch Coordinates (Screen Pixels: x, y)
                  │
                  ▼
┌───────────────────────────────────┐
│        ViewportMapper             │
│  sx = min(viewW/bmpW, viewH/bmpH) │
│  tx = (viewW - bmpW * sx) / 2     │
│  ty = (viewH - bmpH * sy) / 2     │
└─────────────────┬─────────────────┘
                  │
                  ▼
Intrinsic Tensor Coordinates (Canonical 512×512: bx, by)
```

* **Core Features**:
  * **Uniform Letterboxing**: Prevents touch coordinate drift on ultra-wide or tall screens (such as 20:9 or 21:9 aspect ratios).
  * **10-Step Bitmap Undo Stack**: Deep clones 512×512 `Bitmap.Config.ARGB_8888` buffers into a rolling stack for instantaneous zero-latency undo.
  * **Visual Trail Feedback**: Maintains an in-memory queue of `TrailPoint(x, y, timestamp)`. When the user draws, an 18 px wide semi-translucent white glow path fades over 300 ms, leaving a crisp 7 px core path.
  * **Circle-to-Select Magic Touch**: When enabled, drawing a closed loop around any subject triggers an inline **Thinking Pulse** (an animated violet radar wave expanding from the centroid) while running asynchronous GrabCut segmentation in under 180 ms.

---

### 5.2 Super-Resolution Dual-Viewport Layout (`SynchronizedCompareLayout`)

Designed specifically for inspecting sub-pixel reconstruction and seam blending artifacts:

```
SYNCHRONIZED VIEWPORT ARCHITECTURE
┌────────────────────────────────────────────────────────┐
│ SingleImageViewport (Top): BEFORE (512×512 Ground Truth)│
│ • Uniform letterboxed fit                              │
│ • Floating top-left badge: "Before (512 x 512)"        │
├────────────────────────────────────────────────────────┤
│ Divider Bar (6 px Accent Line #1E293B)                 │
├────────────────────────────────────────────────────────┤
│ SingleImageViewport (Bottom): AFTER (1024² - 4096² Out)│
│ • Floating top-left badge: "After (1024 x 1024)"       │
│ • Zoom badge: "🔍 x4.2"                                │
└────────────────────────────────────────────────────────┘
          ▲                                    ▲
          └────────── Transform Sync Bus ──────┘
         (Pinch / Pan on either updates BOTH in lockstep)
```

* **Key Mechanics**:
  * **Lockstep Transformations**: Pinch-to-zoom or single-finger dragging on the top window instantly synchronizes scale and pan coordinates to the bottom window, allowing side-by-side comparison of individual pixels.
  * **Sub-Pixel Nearest Neighbor Filtering**: When zoomed beyond $2.0\times$, anti-aliasing interpolation is disabled (`isFilterBitmap = false`), rendering discrete physical pixels so engineers can verify whether sub-pixel checkerboard grids are properly filled.
  * **Floating 1:1 Reset Control**: Positioned as a floating rounded badge in the top-right corner (`🔍 1:1 RESET`), allowing single-tap restoration to $1.0\times$ view centering.

---

### 5.3 Live Telemetry HUD Bar (`TelemetryBarView`)

Positioned across the top of both inpainting and super-resolution screens:

```
TELEMETRY HUD PILL BAR
┌────────────────────────────────────────────────────────────────────────────────┐
│  🌡️ 36.4°C     │   ⚡ 2.58 W (118 mA)   │   CPU: 18% | GPU: 0%   │   Gov: Normal  │
└────────────────────────────────────────────────────────────────────────────────┘
       ▲                     ▲                         ▲                   ▲
       │                     │                         │                   │
  Color Coded           Active Wattage            Active Cores         Cooling Dev 38
  <55°C Emerald         from PMIC Fuel            Load from sysfs      Kernel Throttling
  >70°C Crimson         Gauge Sensor              kgsl poller          Governor Check
```

* **Dynamic Sensor Polling**: Continuously samples `/sys/class/thermal/`, `/sys/class/power_supply/battery/`, and `/sys/class/kgsl/` on a low-priority daemon background thread at 2 Hz.
* **Color-Coded Thermal Escalation**:
  * $T < 55^\circ\text{C}$: `#10B981` (Emerald Green)
  * $55^\circ\text{C} \le T < 70^\circ\text{C}$: `#F59E0B` (Amber Caution)
  * $T \ge 70^\circ\text{C}$: `#EF4444` (Crimson Danger - Throttling Imminent)

---

### 5.4 LinUCB Contextual Bandit Recommendation Card

Provides continuous explainability for the autonomous model routing engine:

```
ROUTER DECISION CARD
┌────────────────────────────────────────────────────────────────────────────────┐
│ 🎯 RECOMMENDED: MIGAN (RULE_FACE_PORTRAIT)                                     │
│ High facial geometry (1 face detected). Routed to MIGAN (FastRPC 59 ms, 0.62J)│
│ Silicon Temp: 36.2°C • Battery: 100% • Edge Density: 0.024 • Model Preload: OK │
└────────────────────────────────────────────────────────────────────────────────┘
```

* **Dynamic Border & Badge Tinting**:
  * If routed to **MIGAN**: Lavender/Purple accent (`#C084FC`).
  * If routed to **LaMa**: Cyan accent (`#38BDF8`).
  * If routed to **AOT-GAN**: Amber accent (`#F59E0B`).
  * If routed to **Stable Diffusion**: Emerald accent (`#10B981`).

---

## 6. Micro-Interactions & Animation Specifications

```
ANIMATION TIMINGS & CURVES
┌───────────────────────────┬──────────────┬──────────────────┬─────────────────────────────┐
│ Interaction Event         │ Duration     │ Easing Curve     │ Visual Transformation       │
├───────────────────────────┼──────────────┼──────────────────┼─────────────────────────────┤
│ Tool Mode Switch          │ 150 ms       │ FastOutSlowIn    │ Background tint transition  │
│ Magic Touch Thinking Wave │ 1,000 ms loop│ Linear           │ Ring radius expands 20→85px,│
│                           │              │                  │ alpha decays 220→0          │
│ Telemetry Card Expand     │ 250 ms       │ AccelerateDecel  │ Height expands 0→180 dp,    │
│                           │              │                  │ chevron rotates 180°        │
│ Touch Trail Fade-Out      │ 300 ms       │ Decelerate       │ Opacity decays 1.0→0.0      │
│ Theme Mode Day/Night      │ 200 ms       │ Linear           │ Surface colors interpolate  │
└───────────────────────────┴──────────────┴──────────────────┴─────────────────────────────┘
```

---

## 7. Accessibility & Touch Ergonomics

1. **Minimum Touch Target Size**: All interactive buttons, chips, and dropdown spinners maintain a minimum touch bounding box of **$48 \times 48\text{ dp}$**, exceeding Material Design ergonomics guidelines.
2. **Single-Handed Thumb Zone Optimization**: Primary action triggers (`🚀 RUN INPAINT`, `UPSCALE 2X`, Tool selection brush/smart/box) are positioned in the bottom $45\%$ of the physical display.
3. **High-Contrast Text Guarantee**: Contrast ratios between `#F8FAFC` text and `#161F30` card surfaces measure **$13.8:1$**, far surpassing the $4.5:1$ WCAG AA mandate and ensuring full readability under bright outdoor sunlight.
4. **Haptic Confirmation**: Completing an automated GrabCut snap or switching models triggers a short $20\text{ ms}$ haptic pulse via `Vibrator` feedback.

---

## 8. Summary Table of Design Assets & Code References

| Design Subsystem | Key Implementation File | Primary XML / Layout ID | Primary Color Tokens |
| :--- | :--- | :--- | :--- |
| **Theme & Window Controller** | [`MainActivity.kt`](file:///home/tarsh/Desktop/college/ImageInpainting/qidk-inpaint-app-v5/app/src/main/java/com/qualcomm/qidk/inpaint/MainActivity.kt) | `R.id.mainRootLayout` | `#0B0F19`, `#161F30`, `#F8FAFC` |
| **Inpainting Canvas** | [`InpaintCanvasView.kt`](file:///home/tarsh/Desktop/college/ImageInpainting/qidk-inpaint-app-v5/app/src/main/java/com/qualcomm/qidk/inpaint/ui/InpaintCanvasView.kt) | `R.id.inpaintCanvasView` | `#78FF2828`, `#38BDF8`, `#C084FC` |
| **Coordinate Normalizer** | [`ViewportMapper.kt`](file:///home/tarsh/Desktop/college/ImageInpainting/qidk-inpaint-app-v5/app/src/main/java/qidk/fix/ViewportMapper.kt) | Standalone Logic | N/A (Pure Math Engine) |
| **Real-Time Telemetry HUD**| [`TelemetryBarView.kt`](file:///home/tarsh/Desktop/college/ImageInpainting/qidk-inpaint-app-v5/app/src/main/java/qidk/fix/TelemetryBarView.kt) | Programmatic View | `#10B981`, `#F59E0B`, `#EF4444` |
| **Super-Resolution Suite** | [`SuperResolution.kt`](file:///home/tarsh/Desktop/college/ImageInpainting/qidk-inpaint-app-v5/app/src/main/java/qidk/fix/SuperResolution.kt) | `SuperResolutionView` | `#38BDF8`, `#161F30`, `#34D399` |
| **Smart Touch Feedback** | [`SmartGestureController.kt`](file:///home/tarsh/Desktop/college/ImageInpainting/qidk-inpaint-app-v5/app/src/main/java/qidk/fix/SmartGestureController.kt) | Canvas Overlay | `#FFFFFF` (Glow), `#A855F7` (Pulse) |
| **Desktop Companion GUI** | [`desktop_sr_viewer.py`](file:///home/tarsh/Desktop/college/ImageInpainting/desktop_sr_viewer.py) | PyQt5 Desktop View | `#0B0F19`, `#161F30`, `#38BDF8` |
