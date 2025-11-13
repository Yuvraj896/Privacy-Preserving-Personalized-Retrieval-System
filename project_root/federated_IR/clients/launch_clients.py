# federated_IR/clients/launch_clients.py
import subprocess
import sys
import time
from pathlib import Path
import argparse

def launch_clients(
    server: str,
    start_id: int,
    num_clients: int,
    data_dir: str,
    stagger: float = 3.0,
    use_dp: bool = False,
    dp_noise: float = 1.0,
    dp_max_grad: float = 1.0,
    dp_sample_rate: float = 0.02
):
    """
    Launch multiple client processes in parallel.
    stagger: seconds between starting each client
    """
    procs = []

    client_py = Path(__file__).resolve().parent / "client.py"

    # Try virtualenv Python, fallback to current interpreter
    venv_python = Path(__file__).resolve().parents[1] / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        venv_python = Path(sys.executable)

    project_root = Path(__file__).resolve().parents[1]

    for i in range(start_id, start_id + num_clients):
        cmd = [
            str(venv_python),
            str(client_py),
            "--server", server,
            "--data_dir", data_dir,
            "--client_id", str(i)
        ]

        if use_dp:
            cmd += [
                "--use_dp",
                "--dp_noise", str(dp_noise),
                "--dp_max_grad", str(dp_max_grad),
                "--dp_sample_rate", str(dp_sample_rate)
            ]

        print(f"[Launcher] Launching client {i}: {' '.join(cmd)}")

        # Use stdout/stderr as None to directly print to console
        p = subprocess.Popen(
            cmd,
            cwd=str(project_root),  # <-- run from project root
            stdout=None,
            stderr=None,
        )
        procs.append((i, p))
        time.sleep(stagger)

    print(f"[Launcher] Launched {len(procs)} client processes.")

    try:
        while True:
            alive = [(i, p) for i, p in procs if p.poll() is None]
            if not alive:
                print("[Launcher] All clients exited.")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("[Launcher] Terminating clients...")
        for i, p in procs:
            p.terminate()
        print("[Launcher] All clients terminated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, required=True, help="Server address (IP:port)")
    parser.add_argument("--start_id", type=int, required=True, help="Starting client ID")
    parser.add_argument("--num_clients", type=int, required=True, help="Number of clients to launch")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data folder")
    parser.add_argument("--stagger", type=float, default=2.0, help="Seconds to wait between launching clients")
    parser.add_argument("--use_dp", action="store_true", help="Enable differential privacy")
    parser.add_argument("--dp_noise", type=float, default=1.0)
    parser.add_argument("--dp_max_grad", type=float, default=1.0)
    parser.add_argument("--dp_sample_rate", type=float, default=0.02)
    args = parser.parse_args()

    launch_clients(
        server=args.server,
        start_id=args.start_id,
        num_clients=args.num_clients,
        data_dir=args.data_dir,
        stagger=args.stagger,
        use_dp=args.use_dp,
        dp_noise=args.dp_noise,
        dp_max_grad=args.dp_max_grad,
        dp_sample_rate=args.dp_sample_rate
    )
