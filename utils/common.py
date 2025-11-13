
import os, argparse

def add_xml_arg(p):
    p.add_argument("--xml", type=str, default="models/g1_dual_arm.xml", help="Path to MuJoCo XML")
    return p

def add_train_args(p):
    p.add_argument("--total-steps", type=int, default=400_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=100_000)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--render", action="store_true")
    return p

def make_run_dir(prefix, run_name=None):
    import time, pathlib
    if run_name is None:
        run_name = time.strftime("%Y%m%d-%H%M%S")
    path = pathlib.Path("runs")/f"{prefix}_{run_name}"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)
