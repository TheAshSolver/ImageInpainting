#!/usr/bin/env python3
"""
benchmark.py
================================================================================
⚡ QUALCOMM SNAPDRAGON 8 ELITE — UNIFIED BENCHMARK & DIAGNOSTIC SUITE
================================================================================
Unified master benchmark orchestrator and interactive terminal harness for
on-device image inpainting acceleration across Hexagon HTP v79 NPU & Adreno 830 GPU.

Combines and standardizes all project benchmarking and diagnostic capabilities:
  1. Measured On-Device Benchmark Sweep (Fresh monotonic clock & thermal trace)
  2. Stable Diffusion Comparative Benchmark & Hardware Telemetry (12-step vs 20-step)
  3. MI-GAN NPU Hardware Diagnostics & Operator Cycle Audit
  4. Decoupled Two-Phase Batch Sweep (102 evaluation samples)
  5. SNPE HTP Init Cache Builder (Sub-500ms cold start optimization)
  6. Multi-Model Presentation Figures & Visual Analytics (300 DPI)
  7. Qualitative Side-by-Side Inpainting Strips & PPTX Deck Generation
  8. Device Health Probe & Environment Verification Smoke Test

Can be run interactively:
    python benchmark.py
or headlessly via command-line arguments:
    python benchmark.py --fresh --lpips --sd
    python benchmark.py --figures all
    python benchmark.py --smoke-test
"""

import os
import sys
import subprocess
import argparse
import time
import shutil

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
FRESH_BENCH_DIR = os.path.join(SCRIPTS_DIR, "fresh_benchmark")

# ANSI Color Codes
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def print_header(title: str):
    print(f"\n{CYAN}{BOLD}{'=' * 80}{RESET}")
    print(f"{CYAN}{BOLD}⚡ {title}{RESET}")
    print(f"{CYAN}{BOLD}{'=' * 80}{RESET}\n")


def check_adb_device() -> dict:
    """Probes ADB connection to detect attached Qualcomm development device."""
    try:
        res = subprocess.run(["adb", "devices"], capture_output=True, text=True, timeout=5)
        lines = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
        devices = []
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 2:
                devices.append({"serial": parts[0], "status": parts[1]})

        if not devices:
            return {"connected": False, "serial": None, "model": "None", "message": "No ADB devices detected"}

        primary = devices[0]
        if primary["status"] == "device":
            model_res = subprocess.run(
                ["adb", "-s", primary["serial"], "shell", "getprop", "ro.product.model"],
                capture_output=True, text=True, timeout=5
            )
            model_name = model_res.stdout.strip() or "Qualcomm QIDK Snapdragon 8 Elite"
            return {
                "connected": True,
                "serial": primary["serial"],
                "model": model_name,
                "message": f"Connected ({primary['serial']} - {model_name})"
            }
        else:
            return {
                "connected": False,
                "serial": primary["serial"],
                "model": "Unknown",
                "message": f"Device present but unauthorized/offline ({primary['status']})"
            }
    except Exception as e:
        return {"connected": False, "serial": None, "model": "Error", "message": f"ADB check failed: {e}"}


def run_fresh_benchmark(extra_args=None):
    """Executes the measured fresh benchmark sweep."""
    print_header("Executing Measured On-Device Benchmark Sweep (Fresh Harness)")
    script = os.path.join(FRESH_BENCH_DIR, "run_fresh_benchmark.py")
    if not os.path.isfile(script):
        print(f"{RED}Error: Script not found: {script}{RESET}")
        return 1

    cmd = [sys.executable, script]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd).returncode


def run_sd_benchmark(extra_args=None):
    """Executes the dual Stable Diffusion comparative profiler."""
    print_header("Executing Stable Diffusion Head-to-Head NPU Benchmark & Profiler")
    script = os.path.join(SCRIPTS_DIR, "benchmark_sd_models.py")
    if not os.path.isfile(script):
        print(f"{RED}Error: Script not found: {script}{RESET}")
        return 1

    cmd = [sys.executable, script]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd).returncode


def run_migan_diagnostics(extra_args=None):
    """Runs MIGAN hardware diagnostics and layer audit on HTP v79 NPU."""
    print_header("Executing MIGAN Hexagon HTP Hardware Diagnostics & Audit")
    script = os.path.join(SCRIPTS_DIR, "diagnose_migan_hardware.py")
    if not os.path.isfile(script):
        print(f"{RED}Error: Script not found: {script}{RESET}")
        return 1

    cmd = [sys.executable, script]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd).returncode


def run_two_phase_benchmark():
    """Executes the 102-sample decoupled batch benchmark sweep."""
    print_header("Executing Decoupled Two-Phase Batch Benchmark Sweep (102 Samples)")
    script = os.path.join(SCRIPTS_DIR, "run_two_phase_batch_benchmark.py")
    if not os.path.isfile(script):
        print(f"{RED}Error: Script not found: {script}{RESET}")
        return 1

    return subprocess.run([sys.executable, script]).returncode


def build_init_cache():
    """Builds SNPE HTP init cache on device for sub-500ms graph initialization."""
    print_header("Compiling SNPE HTP Init Cache for Instant Zero-Overhead Cold Starts")
    script = os.path.join(FRESH_BENCH_DIR, "make_init_cache.sh")
    if not os.path.isfile(script):
        print(f"{RED}Error: Script not found: {script}{RESET}")
        return 1

    return subprocess.run(["bash", script]).returncode


def generate_figures(figure_set: str = "all"):
    """Generates benchmark presentation figures."""
    print_header(f"Generating Benchmark Visualizations & Figures ({figure_set.upper()})")
    ret = 0

    if figure_set in ("fresh", "all"):
        script = os.path.join(FRESH_BENCH_DIR, "make_figures.py")
        if os.path.isfile(script):
            print(f"{GREEN}→ Generating Fresh Benchmark Figures (Pareto, Radar, Telemetry, CDF)...{RESET}")
            res = subprocess.run([sys.executable, script])
            if res.returncode != 0:
                ret = res.returncode
        else:
            print(f"{YELLOW}Warning: {script} not found.{RESET}")

    if figure_set in ("sd", "all"):
        script = os.path.join(SCRIPTS_DIR, "generate_sd_comparison_figures.py")
        if os.path.isfile(script):
            print(f"{GREEN}→ Generating Stable Diffusion Comparative Figures...{RESET}")
            res = subprocess.run([sys.executable, script])
            if res.returncode != 0:
                ret = res.returncode
        else:
            print(f"{YELLOW}Warning: {script} not found.{RESET}")

    if figure_set in ("publication", "all"):
        script = os.path.join(SCRIPTS_DIR, "generate_presentation_visuals.py")
        if os.path.isfile(script):
            print(f"{GREEN}→ Generating 8 Publication-Grade 300-DPI Figures...{RESET}")
            res = subprocess.run([sys.executable, script])
            if res.returncode != 0:
                ret = res.returncode
        else:
            print(f"{YELLOW}Warning: {script} not found.{RESET}")

    return ret


def generate_qualitative_strips_and_deck():
    """Generates multi-sample inpainting comparison strips and PowerPoint slides."""
    print_header("Generating Qualitative Inpainting Strips & PPTX Executive Deck")
    ret = 0

    strip_script = os.path.join(SCRIPTS_DIR, "render_inpaint_comparisons.py")
    if os.path.isfile(strip_script):
        print(f"{GREEN}→ Rendering Inpainting Side-by-Side Strips...{RESET}")
        res = subprocess.run([sys.executable, strip_script])
        if res.returncode != 0:
            ret = res.returncode

    deck_script = os.path.join(SCRIPTS_DIR, "generate_benchmark_deck.py")
    if os.path.isfile(deck_script):
        print(f"{GREEN}→ Building Benchmark Presentation Slide Deck...{RESET}")
        res = subprocess.run([sys.executable, deck_script])
        if res.returncode != 0:
            ret = res.returncode

    return ret


def run_smoke_test():
    """Checks device readiness, DLC files, DSP runtime libraries, and environment."""
    print_header("Qualcomm Snapdragon 8 Elite Environment Smoke Test")
    dev = check_adb_device()

    print(f"1. ADB Connectivity:")
    if dev["connected"]:
        print(f"   {GREEN}✓ PASS:{RESET} {dev['message']}")
    else:
        print(f"   {RED}✗ FAIL:{RESET} {dev['message']}")
        print(f"   {YELLOW}Note: An active Snapdragon 8 Elite QIDK device must be connected via ADB.{RESET}")
        return 1

    print(f"\n2. Checking On-Device Runtime Staging Directories:")
    test_dirs = ["/data/local/tmp/lama", "/data/local/tmp/sd_runtime"]
    for d in test_dirs:
        res = subprocess.run(["adb", "shell", f"[ -d {d} ] && echo EXISTS"], capture_output=True, text=True)
        if "EXISTS" in res.stdout:
            print(f"   {GREEN}✓ PASS:{RESET} Directory {d} exists on device.")
        else:
            print(f"   {YELLOW}⚠ WARN:{RESET} Directory {d} missing. Push DLCs and runtime binaries as documented in SETUP.md.")

    print(f"\n3. Checking On-Device Model DLCs & Executables:")
    checks = [
        ("/data/local/tmp/lama/migan_htp_v79.dlc", "MIGAN HTP v79 Container"),
        ("/data/local/tmp/lama/lama_dilated.dlc", "LaMa Dilated Container"),
        ("/data/local/tmp/lama/aotgan.dlc", "AOT-GAN Container"),
        ("/data/local/tmp/lama/snpe-net-run", "SNPE Net Run Binary"),
        ("/data/local/tmp/sd_runtime/sd_qidk_runner_inpaint", "SD Inpaint Runner (12-step DPM)"),
        ("/data/local/tmp/sd_runtime/sd_qidk_runner_inefficient", "SD Inefficient Runner (20-step Euler)")
    ]
    for path, name in checks:
        res = subprocess.run(["adb", "shell", f"[ -f {path} ] && echo EXISTS"], capture_output=True, text=True)
        if "EXISTS" in res.stdout:
            print(f"   {GREEN}✓ PASS:{RESET} {name} found at {path}")
        else:
            print(f"   {YELLOW}⚠ MISSING:{RESET} {name} not found at {path}")

    print(f"\n4. Checking Python Environment & Dependencies:")
    deps = ["torch", "torchmetrics", "PIL", "cv2", "numpy", "pandas", "matplotlib"]
    for pkg in deps:
        try:
            __import__(pkg)
            print(f"   {GREEN}✓ PASS:{RESET} Python module '{pkg}' is installed.")
        except ImportError:
            print(f"   {YELLOW}⚠ MISSING:{RESET} Python module '{pkg}' is missing (install with uv pip install).")

    print(f"\n{GREEN}{BOLD}Smoke test complete.{RESET}\n")
    return 0


def interactive_menu():
    """Displays the interactive CLI menu for benchmarking."""
    while True:
        dev = check_adb_device()
        status_color = GREEN if dev["connected"] else YELLOW
        dev_label = dev["message"]

        print(f"\n{CYAN}{BOLD}================================================================================{RESET}")
        print(f"{CYAN}{BOLD}⚡ QUALCOMM SNAPDRAGON 8 ELITE — UNIFIED BENCHMARK & DIAGNOSTIC SUITE{RESET}")
        print(f"{CYAN}{BOLD}================================================================================{RESET}")
        print(f"Target Hardware : {BOLD}Qualcomm Hexagon HTP v79 NPU & Adreno 830 GPU{RESET}")
        print(f"Device Status   : {status_color}{dev_label}{RESET}")
        print(f"{CYAN}--------------------------------------------------------------------------------{RESET}")
        print(f" {BOLD}[1]{RESET} Measured On-Device Benchmark Sweep (Fresh Harness)")
        print(f"     {DIM}Monotonic latency, thermals & quality (PSNR/SSIM/LPIPS) across NPU & GPU{RESET}")
        print(f" {BOLD}[2]{RESET} Stable Diffusion Head-to-Head Comparison & Profiler")
        print(f"     {DIM}12-step DPM-Solver++ (Context Inpaint) vs 20-step Euler (RePaint){RESET}")
        print(f" {BOLD}[3]{RESET} MI-GAN Hexagon HTP Hardware Diagnostics & Cycle Audit")
        print(f"     {DIM}Cold vs warm lifecycle, burst scaling, QuRT hints, DSP cycle audit{RESET}")
        print(f" {BOLD}[4]{RESET} Decoupled Two-Phase Batch Benchmark Sweep (102 Samples)")
        print(f"     {DIM}Automated device batch sweep followed by local quality evaluation{RESET}")
        print(f" {BOLD}[5]{RESET} Compile SNPE HTP Init Cache")
        print(f"     {DIM}One-off graph cache cuts NPU cold start from ~8s to <0.5s{RESET}")
        print(f" {BOLD}[6]{RESET} Generate Benchmark & Telemetry Figures (300 DPI)")
        print(f"     {DIM}Pareto frontiers, radar showdowns, thermal traces, boxplots, CDFs{RESET}")
        print(f" {BOLD}[7]{RESET} Generate Qualitative Inpainting Strips & PPTX Deck")
        print(f"     {DIM}High-resolution composite comparison strips and PowerPoint presentations{RESET}")
        print(f" {BOLD}[8]{RESET} Run Environment Smoke Test & Device Health Probe")
        print(f"     {DIM}Verifies ADB, runtime staging directories, DLCs, and dependencies{RESET}")
        print(f" {BOLD}[0]{RESET} Exit")
        print(f"{CYAN}================================================================================{RESET}")

        try:
            choice = input(f"{BOLD}Select an option [0-8]: {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if choice == "0":
            print(f"{GREEN}Goodbye!{RESET}")
            break
        elif choice == "1":
            sd_choice = input("Include Stable Diffusion in fresh sweep? (y/N): ").strip().lower() == "y"
            lpips_choice = input("Compute LPIPS perceptual loss? (Y/n): ").strip().lower() != "n"
            args = []
            if sd_choice:
                args.append("--sd")
            if lpips_choice:
                args.append("--lpips")
            run_fresh_benchmark(args)
        elif choice == "2":
            lim = input("Limit sample count (Enter for full previous dataset): ").strip()
            args = []
            if lim.isdigit():
                args.extend(["--limit", lim])
            run_sd_benchmark(args)
        elif choice == "3":
            run_migan_diagnostics()
        elif choice == "4":
            run_two_phase_benchmark()
        elif choice == "5":
            build_init_cache()
        elif choice == "6":
            print("\nFigure Set Selection:")
            print("  1) Fresh Benchmark Figures (Pareto, Radar, Telemetry, CDF)")
            print("  2) Stable Diffusion Comparison Figures")
            print("  3) Publication-Grade 8 Figures (300 DPI)")
            print("  4) All Figures")
            f_choice = input("Select [1-4] (default: 4): ").strip()
            f_map = {"1": "fresh", "2": "sd", "3": "publication", "4": "all", "": "all"}
            generate_figures(f_map.get(f_choice, "all"))
        elif choice == "7":
            generate_qualitative_strips_and_deck()
        elif choice == "8":
            run_smoke_test()
        else:
            print(f"{RED}Invalid option: {choice}{RESET}")

        input(f"\n{DIM}Press Enter to return to menu...{RESET}")


def main():
    parser = argparse.ArgumentParser(
        description="Qualcomm Snapdragon 8 Elite — Unified Benchmark & Diagnostic Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py                          # Launch interactive terminal menu
  python benchmark.py --fresh --lpips --sd     # Run full measured fresh benchmark with SD & LPIPS
  python benchmark.py --sd                     # Run dual SD comparative profiler
  python benchmark.py --migan-diag             # Run MI-GAN NPU hardware diagnostics
  python benchmark.py --two-phase              # Run 102-sample two-phase batch sweep
  python benchmark.py --figures all            # Generate all presentation and analytics figures
  python benchmark.py --deck                   # Generate PowerPoint presentation deck
  python benchmark.py --init-cache             # Compile SNPE HTP init cache
  python benchmark.py --smoke-test             # Probe connected device readiness
"""
    )
    parser.add_argument("-m", "--menu", action="store_true", help="Launch interactive terminal menu")
    parser.add_argument("-f", "--fresh", action="store_true", help="Execute measured on-device fresh benchmark sweep")
    parser.add_argument("--lpips", action="store_true", help="Compute LPIPS perceptual distance in fresh benchmark")
    parser.add_argument("--sd", action="store_true", help="Include or run Stable Diffusion benchmark")
    parser.add_argument("-d", "--migan-diag", action="store_true", help="Run MI-GAN hardware diagnostics on Hexagon NPU")
    parser.add_argument("-t", "--two-phase", action="store_true", help="Run decoupled two-phase 102-sample batch sweep")
    parser.add_argument("-c", "--init-cache", action="store_true", help="Build SNPE HTP init cache on device")
    parser.add_argument("-g", "--figures", choices=["fresh", "sd", "publication", "all"], nargs="?", const="all", help="Generate benchmark visualization figures")
    parser.add_argument("--deck", action="store_true", help="Generate qualitative comparisons and presentation slide deck")
    parser.add_argument("--smoke-test", action="store_true", help="Verify ADB connection and on-device runtime staging")
    parser.add_argument("--configs", nargs="*", help="Specific model configs for fresh benchmark (e.g. migan_npu lama_npu)")

    args, unknown = parser.parse_known_args()

    # If no flags provided or --menu explicitly requested, run interactive menu
    if len(sys.argv) == 1 or args.menu:
        interactive_menu()
        return

    # Dispatch requested actions
    if args.smoke_test:
        run_smoke_test()
    if args.init_cache:
        build_init_cache()
    if args.fresh:
        fresh_args = []
        if args.lpips:
            fresh_args.append("--lpips")
        if args.sd:
            fresh_args.append("--sd")
        if args.configs:
            fresh_args.extend(["--configs", *args.configs])
        fresh_args.extend(unknown)
        run_fresh_benchmark(fresh_args)
    elif args.sd:
        run_sd_benchmark(unknown)
    if args.migan_diag:
        run_migan_diagnostics(unknown)
    if args.two_phase:
        run_two_phase_benchmark()
    if args.figures:
        generate_figures(args.figures)
    if args.deck:
        generate_qualitative_strips_and_deck()


if __name__ == "__main__":
    main()
