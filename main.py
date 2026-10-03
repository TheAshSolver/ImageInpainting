#!/usr/bin/env python3
"""
Qualcomm Snapdragon 8 Elite: Edge Image Inpainting Master CLI
Unified entrypoint for Web GUI, Benchmarking, Visuals, and Routing.

Usage:
    python main.py gui                  # Launch the interactive Gradio Web UI
    python main.py visuals              # Generate all 8 publication-grade presentation figures (300 DPI)
    python main.py benchmark            # Run the two-phase decoupled batch benchmark on device
    python main.py route -i <img> [-m <mask>]  # Run the multi-modal decision router
"""

import sys
import os
import argparse
import subprocess
import json

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

def print_banner():
    banner = """
================================================================================
   QUALCOMM SNAPDRAGON 8 ELITE: EDGE INPAINTING PLATFORM
   Accelerators: Hexagon NPU (HTP v79) | Adreno 830 GPU | FastRPC
================================================================================
"""
    print(banner)

def check_adb_status():
    """Checks if a Qualcomm QIDK developer device is connected via ADB."""
    try:
        res = subprocess.run(["adb", "devices"], capture_output=True, text=True, timeout=3)
        lines = [l.strip() for l in res.stdout.split("\n") if l.strip() and not l.startswith("List")]
        if lines:
            print(f"  📱 Connected ADB Device: {lines[0]}")
            return True
        else:
            print("  ⚠️  No active ADB device detected. (Host simulation mode enabled)")
            return False
    except Exception:
        print("  ⚠️  ADB not found on PATH.")
        return False

def cmd_gui(args):
    """Launch the Gradio Web GUI."""
    print("🚀 Launching Interactive Inpainting Web GUI...")
    from src.app_gui import build_app
    demo = build_app()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)

def cmd_visuals(args):
    """Generate presentation visual figures."""
    print("🎨 Generating 8 Publication-Grade Presentation Figures (300 DPI)...")
    script_path = os.path.join(REPO_ROOT, "scripts", "generate_presentation_visuals.py")
    subprocess.run([sys.executable, script_path], check=True)

def cmd_benchmark(args):
    """Run the decoupled two-phase batch benchmark."""
    print("⚡ Executing Two-Phase Decoupled Batch Benchmark Sweep...")
    script_path = os.path.join(REPO_ROOT, "scripts", "run_two_phase_batch_benchmark.py")
    cmd = [sys.executable, script_path]
    if args.samples:
        cmd.extend(["--samples", str(args.samples)])
    subprocess.run(cmd, check=True)

def cmd_route(args):
    """Run the decision router on an image and optional mask."""
    from src.router import classify_and_route
    print(f"🎯 Running Decision Router on: {args.image}")
    decision = classify_and_route(args.image, args.mask)
    
    prof = decision["telemetry"]
    print("\n--- Router Decision Summary ---")
    print(f"  Recommended Architecture : {decision['recommended_model'].upper()}")
    print(f"  Target Hardware Runtime  : {prof['runtime'].upper()} ({prof['runtime_flag']})")
    print(f"  Decision Rule Triggered  : {decision['rule_triggered']}")
    print(f"  Expected Profile Latency : {prof['latency_ms']} ms | Energy: {prof['energy_j']} J")
    print(f"  Justification            : {decision['justification']}")
    print("\n--- Extracted Multi-Modal Features ---")
    for k, v in decision['features'].items():
        print(f"  • {k:22s}: {v}")
    
    if args.json:
        with open(args.json, "w") as f:
            json.dump(decision, f, indent=2)
        print(f"\n✅ Full JSON written to: {args.json}")

def main():
    parser = argparse.ArgumentParser(
        description="Master CLI for Qualcomm Snapdragon 8 Elite Image Inpainting Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py gui --port 7860
    python main.py visuals
    python main.py benchmark --samples 102
    python main.py route -i Benchmark/input_102/image/001.png -m Benchmark/input_102/mask/001.png
        """
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # GUI Command
    p_gui = subparsers.add_parser("gui", help="Launch interactive Gradio Web UI")
    p_gui.add_argument("--host", default="0.0.0.0", help="Host IP (default: 0.0.0.0)")
    p_gui.add_argument("--port", type=int, default=7860, help="Port (default: 7860)")
    p_gui.add_argument("--share", action="store_true", help="Create public Gradio share link")

    # Visuals Command
    p_vis = subparsers.add_parser("visuals", help="Generate publication presentation figures (300 DPI)")

    # Benchmark Command
    p_bm = subparsers.add_parser("benchmark", help="Execute on-device batch benchmarking sweep")
    p_bm.add_argument("--samples", type=int, default=102, help="Number of benchmark samples to evaluate")

    # Router Command
    p_rt = subparsers.add_parser("route", help="Execute multi-modal decision router on an image/mask")
    p_rt.add_argument("-i", "--image", required=True, help="Path to input image")
    p_rt.add_argument("-m", "--mask", default=None, help="Optional path to binary mask")
    p_rt.add_argument("--json", default=None, help="Optional path to save JSON output")

    args = parser.parse_args()

    if not args.command:
        print_banner()
        check_adb_status()
        print("Available Commands:")
        print("  • python main.py gui        -> Launch real-time Gradio Web Canvas")
        print("  • python main.py visuals    -> Generate 8 publication-grade presentation figures (300 DPI)")
        print("  • python main.py benchmark  -> Execute batch benchmarking sweep across NPU/GPU")
        print("  • python main.py route      -> Analyze image & mask with heuristic decision router")
        print("\nRun 'python main.py <command> --help' for command-specific flags.")
        sys.exit(0)

    if args.command == "gui":
        cmd_gui(args)
    elif args.command == "visuals":
        cmd_visuals(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "route":
        cmd_route(args)

if __name__ == "__main__":
    main()
