#!/usr/bin/env python3
"""
Plot Task 3.4–3.6 results from a rosbag2 recording.

Usage:
  python3 plot_task3_bag.py <bag_folder> --eta /CSEI/control/eta --u /CSEI/control/u_cmd --joy /joy

Creates PNG plots in the current directory.
"""

import argparse
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

def read_bag(bag_path: str, topics: list[str]):
    bag_path = str(Path(bag_path).resolve())
    storage_options = StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")

    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    # Resolve topic types
    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    missing = [t for t in topics if t not in topic_types]
    if missing:
        raise RuntimeError(f"Topics not in bag: {missing}\nAvailable: {sorted(topic_types)}")

    msgs = {t: [] for t in topics}
    times = {t: [] for t in topics}

    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic not in msgs:
            continue
        msg_type = get_message(topic_types[topic])
        msg = deserialize_message(data, msg_type)
        msgs[topic].append(msg)
        times[topic].append(t_ns * 1e-9)  # seconds

    return times, msgs

def extract_joy(msgs):
    # Returns t, axes0.., buttons0.. as arrays
    t = np.array(msgs["t"])
    axes = np.array([m.axes for m in msgs["msg"]], dtype=float)
    buttons = np.array([m.buttons for m in msgs["msg"]], dtype=float)
    return t, axes, buttons

def extract_multiarray(msgs):
    t = np.array(msgs["t"])
    data = np.array([m.data for m in msgs["msg"]], dtype=float)
    return t, data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", help="rosbag2 folder (the folder that contains metadata.yaml)")
    ap.add_argument("--joy", default="/joy")
    ap.add_argument("--eta", default="/tmr4243/state/eta")
    ap.add_argument("--u",   default="/tmr4243/command/u")
    ap.add_argument("--out_prefix", default="task3")
    args = ap.parse_args()

    topics = [args.joy, args.eta, args.u]
    t, m = read_bag(args.bag, topics)

    # Package per topic
    joy = {"t": t[args.joy], "msg": m[args.joy]}
    eta = {"t": t[args.eta], "msg": m[args.eta]}
    u   = {"t": t[args.u],   "msg": m[args.u]}

    t_joy, axes, buttons = extract_joy(joy)
    t_eta, eta_arr = extract_multiarray(eta)
    t_u, u_arr = extract_multiarray(u)

    # Plot ETA
    plt.figure()
    for i, lab in enumerate(["x", "y", "psi"]):
        if eta_arr.shape[1] > i:
            plt.plot(t_eta, eta_arr[:, i], label=lab)
    plt.legend()
    plt.xlabel("t [s]")
    plt.ylabel("eta")
    plt.title("Vessel position/orientation (eta)")
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_eta.png", dpi=200)

    # Plot u_cmd
    plt.figure()
    labs = ["u0(tunnel)", "u1(az1)", "u2(az2)", "alpha1", "alpha2"]
    for i, lab in enumerate(labs):
        if u_arr.shape[1] > i:
            plt.plot(t_u, u_arr[:, i], label=lab)
    plt.legend()
    plt.xlabel("t [s]")
    plt.ylabel("command")
    plt.title("Command vector u = [u0,u1,u2,a1,a2]")
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_u.png", dpi=200)

    # Plot a few joystick axes to help your report
    plt.figure()
    # show first 6 axes (common for many controllers)
    for i in range(min(axes.shape[1], 6)):
        plt.plot(t_joy, axes[:, i], label=f"axis[{i}]")
    plt.legend()
    plt.xlabel("t [s]")
    plt.ylabel("axes")
    plt.title("/joy axes (first 6)")
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_joy_axes.png", dpi=200)

    print("Saved:",
          f"{args.out_prefix}_eta.png, {args.out_prefix}_u.png, {args.out_prefix}_joy_axes.png")

if __name__ == "__main__":
    main()
