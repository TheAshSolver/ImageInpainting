import subprocess
import os
import argparse
import shutil

# Canonical model configurations
MODEL_CONFIG = {
    "migan": {
        "dlc": "migan.dlc",
        "runtime": "dsp"
    },
    "aotgan": {
        "dlc": "aotgan.dlc",
        "runtime": "gpu"
    },
    "lama": {
        "dlc": "lama_dilated.dlc",
        "runtime": "gpu"
    }
}


def run_adb_command(command, shell_mode=False):
    """Executes an ADB command and returns (success, stdout)."""
    try:
        if shell_mode:
            print(f"Executing ADB Shell: {command}")
            result = subprocess.run(
                ["adb", "shell", command], 
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
        else:
            print(f"Executing: {' '.join(command)}")
            result = subprocess.run(
                command, 
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
        
        if result.stdout:
            print(result.stdout.strip())
        return True, result.stdout

    except subprocess.CalledProcessError as e:
        print(f"❌ Error executing command.")
        print(f"Exit code: {e.returncode}")
        if e.stderr:
            print(f"Error output: {e.stderr.strip()}")
        elif e.stdout:
            print(f"Output: {e.stdout.strip()}")
        return False, e.stderr or e.stdout


def run_snpe_inference(
    model="migan",
    base_dir="/data/local/tmp/lama", 
    model_name=None, 
    input_list="input_list.txt",
    output_dir="output",
    local_output_dir=None,
    runtime=None,
    clean_device=True,
    clean_local=True
):
    model_key = model.lower() if model else "migan"
    cfg = MODEL_CONFIG.get(model_key, {"dlc": f"{model_key}.dlc", "runtime": "dsp"})
    
    if model_name is None:
        model_name = cfg["dlc"]
    if runtime is None:
        runtime = cfg["runtime"]
    if local_output_dir is None:
        local_output_dir = f"Benchmark/output/{model_key}"

    print(f"\n==================================================")
    print(f"  Model             : {model_key.upper()}")
    print(f"  DLC Container     : {model_name}")
    print(f"  Runtime           : {runtime.upper()}")
    print(f"  Device Working Dir: {base_dir}")
    print(f"  Device Output Dir : {base_dir}/{output_dir}")
    print(f"  Local Isolated Dir: {local_output_dir}")
    print(f"==================================================")

    device_output_path = f"{base_dir}/{output_dir}"

    # 1. Clean on-device output directory before running to prevent stale outputs
    if clean_device:
        print(f"\n🧹 Cleaning on-device output directory ({device_output_path})...")
        clean_cmd = f"rm -rf {device_output_path} && mkdir -p {device_output_path}"
        clean_ok, _ = run_adb_command(clean_cmd, shell_mode=True)
        if clean_ok:
            print("✅ Device output directory cleaned.")
        else:
            print("⚠️ Warning: Could not clean on-device output directory. Continuing...")

    # 2. Clean local isolated folder if requested
    if clean_local and os.path.exists(local_output_dir):
        print(f"🧹 Cleaning local isolated folder ({local_output_dir})...")
        shutil.rmtree(local_output_dir)
    os.makedirs(local_output_dir, exist_ok=True)

    # 3. Construct environment variables 
    env_vars = [
        f"export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{base_dir}/lib:{base_dir}:vendor/lib64:/system/lib64",
        f"export ADSP_LIBRARY_PATH='{base_dir}/dsp/lib;{base_dir}/dsp;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp'",
        f"export PATH=$PATH:{base_dir}/bin:{base_dir}"
    ]
    
    # 4. Construct the snpe-net-run execution arguments
    snpe_args = [
        "snpe-net-run",
        f"--container {model_name}",
        f"--input_list {input_list}",
        f"--output_dir {output_dir}"
    ]

    if runtime == "dsp":
        snpe_args.append("--use_dsp")
    elif runtime == "gpu":
        snpe_args.append("--use_gpu")
    elif runtime == "aip":
        snpe_args.append("--use_aip")

    # 5. Chain everything together correctly with "&&"
    snpe_execution = " ".join(snpe_args)
    commands_to_chain = env_vars + [f"cd {base_dir}", snpe_execution]
    
    full_shell_command = " && ".join(commands_to_chain)

    # 6. Execute Inference
    print(f"\n🚀 Running SNPE inference on {runtime.upper()}...")
    success, _ = run_adb_command(full_shell_command, shell_mode=True)

    if not success:
        print("⚠️ Inference failed. Halting pull process.")
        return False

    # 7. Pull Output into isolated model directory
    print(f"\n📥 Inference complete. Pulling results from device to: {local_output_dir}")
    
    device_output_src = f"{device_output_path}/."
    pull_command = ["adb", "pull", device_output_src, local_output_dir]
    
    pull_success, _ = run_adb_command(pull_command, shell_mode=False)

    if pull_success:
        print(f"\n✅ All outputs successfully pulled to: {local_output_dir}")
        return True
    else:
        print(f"❌ Failed to pull outputs from device.")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SNPE inference on device with pre-execution clean and isolated output pulling."
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=["migan", "aotgan", "lama"],
        default="migan",
        help="Target model: migan, aotgan, lama (default: migan)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Explicit DLC model filename on device (e.g. migan_htp_v79.dlc)"
    )
    parser.add_argument(
        "--runtime",
        type=str,
        choices=["dsp", "gpu", "cpu", "aip"],
        default=None,
        help="SNPE runtime (dsp, gpu, cpu, aip). Defaults to model recommended runtime."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/data/local/tmp/lama",
        help="Base working directory on device (default: /data/local/tmp/lama)"
    )
    parser.add_argument(
        "--input_list",
        type=str,
        default="input_list.txt",
        help="Input list file on device (default: input_list.txt)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output folder name on device (default: output)"
    )
    parser.add_argument(
        "--local_output_dir",
        type=str,
        default=None,
        help="Local directory to pull outputs into (default: Benchmark/output/<model>)"
    )
    parser.add_argument(
        "--no-clean-device",
        action="store_false",
        dest="clean_device",
        help="Skip cleaning on-device output directory before running"
    )
    parser.add_argument(
        "--no-clean-local",
        action="store_false",
        dest="clean_local",
        help="Skip cleaning local isolated directory before pulling"
    )

    args = parser.parse_args()

    print("Checking for connected ADB devices...")
    subprocess.run(["adb", "devices"])
    print("-" * 40)
    
    run_snpe_inference(
        model=args.model,
        base_dir=args.base_dir,
        model_name=args.model_name,
        input_list=args.input_list,
        output_dir=args.output_dir,
        local_output_dir=args.local_output_dir,
        runtime=args.runtime,
        clean_device=args.clean_device,
        clean_local=args.clean_local
    )
