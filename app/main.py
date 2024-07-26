import subprocess
import sys
import os


def run_command(command):
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Failed to execute command: {command}")
        print(stderr.decode())
        sys.exit(1)
    return stdout.decode()


def main():
    # Create a virtual environment
    run_command("python3 -m venv video-stream-detection-env")

    # Activate the virtual environment
    if os.name == "posix":  # POSIX is for UNIX-like OS
        run_command("source video-stream-detection-env/bin/activate")
    elif os.name == "nt":  # NT is for Windows
        run_command("myenv\\Scripts\\activate")
    else:
        print("OS not supported")
        sys.exit(1)

    # Install requirements
    run_command("pip3 install -r requirements.txt")

    # Run vit_fire_detection.py to save the model
    run_command("python3 ./app/model/video_stream_detection.py")

    # Build the Docker image
    run_command("docker build --no-cache -t video_stream_detection .")


if __name__ == "__main__":
    main()
