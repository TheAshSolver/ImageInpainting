import subprocess
import os

def run_adb_command(command, shell_mode=False):
    """Executes an ADB command and returns the output."""
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
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Error executing command.")
        print(f"Exit code: {e.returncode}")
        if e.stderr:
            print(f"Error output: {e.stderr.strip()}")
        elif e.stdout:
            print(f"Output: {e.stdout.strip()}")
        return False

def run_snpe_inference(
    base_dir="/data/local/tmp/lama", 
    model_name="migan.dlc", 
    input_list="input_list.txt",
    output_dir="output",
    local_output_dir="Benchmark/output",
    runtime="dsp" 
):
    print(f"--- Starting SNPE Inference on {runtime.upper()} ---")

    # 1. Construct environment variables 
    env_vars = [
        f"export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{base_dir}/lib:{base_dir}",
        f"export ADSP_LIBRARY_PATH='{base_dir}/dsp/lib;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp'",
        f"export PATH=$PATH:{base_dir}/bin:{base_dir}"
    ]
    
    # 2. Construct the snpe-net-run execution arguments
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

    # 3. Chain everything together correctly with "&&"
    snpe_execution = " ".join(snpe_args)
    commands_to_chain = env_vars + [f"cd {base_dir}", snpe_execution]
    
    full_shell_command = " && ".join(commands_to_chain)

    # 4. Execute Inference
    print("\nRunning inference on device...")
    success = run_adb_command(full_shell_command, shell_mode=True)

    if not success:
        print("⚠️ Inference failed. Halting pull process.")
        return

    # 5. Pull Output
    print("\n--- Inference complete. Pulling results ---")
    
    os.makedirs(local_output_dir, exist_ok=True)
    
    device_output_path = f"{base_dir}/{output_dir}/."
    pull_command = ["adb", "pull", device_output_path, local_output_dir]
    
    pull_success = run_adb_command(pull_command, shell_mode=False)

    if pull_success:
        print(f"\n✅ All outputs successfully pulled to: {local_output_dir}")
    else:
        print(f"❌ Failed to pull outputs from device.")

if __name__ == "__main__":
    print("Checking for connected ADB devices...")
    subprocess.run(["adb", "devices"])
    print("-" * 40)
    
    run_snpe_inference(runtime="gpu", model_name="aotgan.dlc")