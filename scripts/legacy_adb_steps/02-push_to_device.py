import subprocess
import os

def run_adb_command(command):
    """Executes a shell command and prints the output."""
    try:
        print(f"Executing: {' '.join(command)}")
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.stdout:
            print(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"❌ Error executing command: {' '.join(command)}")
        print(e.stderr.strip())

def push_to_qidk(
    base_target_dir="/data/local/tmp/lama",
    local_raw_img_dir="Benchmark/input/raw_image",
    local_raw_mask_dir="Benchmark/input/raw_mask",
    local_input_list="input_list.txt"
):
    # Define target directories on the device
    target_img_dir = f"{base_target_dir}/input/raw_image"
    target_mask_dir = f"{base_target_dir}/input/raw_mask"

    # Step 1: Create the directory structure on the device
    print("Creating directories on the device...")
    run_adb_command(["adb", "shell", "mkdir", "-p", target_img_dir])
    run_adb_command(["adb", "shell", "mkdir", "-p", target_mask_dir])

    # Step 2: Push the raw images
    if os.path.exists(local_raw_img_dir):
        print("\nPushing raw images...")
        # Using "/." at the end of the local path pushes the contents of the directory
        run_adb_command(["adb", "push", f"{local_raw_img_dir}/.", target_img_dir])
    else:
        print(f"⚠️ Local image directory not found: {local_raw_img_dir}")

    # Step 3: Push the raw masks
    if os.path.exists(local_raw_mask_dir):
        print("\nPushing raw masks...")
        run_adb_command(["adb", "push", f"{local_raw_mask_dir}/.", target_mask_dir])
    else:
        print(f"⚠️ Local mask directory not found: {local_raw_mask_dir}")

    # Step 4: Push the input_list.txt
    if os.path.exists(local_input_list):
        print("\nPushing input_list.txt...")
        run_adb_command(["adb", "push", local_input_list, base_target_dir])
    else:
        print(f"⚠️ Local input_list.txt not found: {local_input_list}")

    print("\n✅ All pushes completed successfully!")

if __name__ == "__main__":
    # Ensure adb is connected and a device is found before running
    print("Checking for connected ADB devices...")
    subprocess.run(["adb", "devices"])
    print("-" * 40)
    
    push_to_qidk()