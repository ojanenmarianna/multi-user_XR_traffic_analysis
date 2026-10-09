"""
helpers.py - Common utilities, configuration, packet parsers, and statistical models
for HotMobile 2027 MetaClassroom multi-user XR passive traffic analysis.
"""

from pathlib import Path
import glob
import struct
import collections
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Global Configuration & Protocol Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent if "__file__" in locals() else Path(".").resolve()
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"

import os
from dotenv import load_dotenv

load_dotenv()

# Cloud Server IP & UDP Game Port
SERVER_IP = os.environ.get("SERVER_IP", "127.0.0.1")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "30780"))

# Meta Quest 3 Headsets
_headset_ips_str = os.environ.get("HEADSET_IPS", "")
if _headset_ips_str:
    HEADSET_IPS = [ip.strip() for ip in _headset_ips_str.split(",")]
else:
    HEADSET_IPS = [f"10.0.0.{i}" for i in range(1, 10)]

HEADSET_LABELS = {ip: f"HMD-.{ip.split('.')[-1]}" for ip in HEADSET_IPS}

# Protocol Op-codes
MSG_PING = 0x03
MSG_PONG = 0x04

# Session Timeline Phases (minutes from session t0 = 14:37:56)
SESSION_PHASES = [
    ("Phase 0: Room Join", 0.0, 14.0, "#F0F0F0"),
    ("Scenario 1: Lecture", 14.0, 27.0, "#DDF2EB"),
    ("Intermission", 27.0, 29.0, "#FBF0E4"),
    ("Scenario 2: Task Execution", 29.0, 36.0, "#FCE7D8"),
    ("Phase 3: Room Leave", 36.0, 43.4, "#F0F0F0"),
]

# Time binning configuration
BIN_SEC = 30

# High-contrast colorblind-safe palette (Okabe-Ito / ACM style)
COLOR_UP = "#0072B2"      # Blue (Upstream / HMD -> Server)
COLOR_DOWN = "#D55E00"    # Vermilion (Downstream / Server -> HMD)
COLOR_FIT = "#000000"     # Black for analytical model curves
COLOR_REF = "#64748B"     # Slate grey for reference / mode lines
COLOR_SERVER_PING = COLOR_DOWN  # Vermilion for Server-initiated ping (s2c)
COLOR_CLIENT_PING = COLOR_UP    # Blue for Client-initiated ping (c2s)

# ---------------------------------------------------------------------------
# Typography & Matplotlib Styling
# ---------------------------------------------------------------------------

def set_publication_style():
    """Sets consistent ACM-compatible publication typography."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.titlesize": 11,
        "lines.linewidth": 1.5,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "figure.autolayout": False,
        "savefig.dpi": 300,
    })


# ---------------------------------------------------------------------------
# Low-Level PCAP & PCAP-NG Extraction (Zero External Dependency)
# ---------------------------------------------------------------------------

def get_pcap_files(data_dir: Path = None):
    """Finds merged or chunked PCAP/PCAP-NG files in data_dir."""
    if data_dir is None:
        data_dir = DATA_DIR
    data_dir = Path(data_dir)
    pcap_files = sorted(glob.glob(str(data_dir / "08062026_VM*.pcap")))
    if not pcap_files:
        pcap_files = sorted(glob.glob(str(data_dir / "*.pcap*")))
    return pcap_files


def iter_packets(path: str):
    """
    Yields (timestamp_epoch, raw_frame_bytes) from classic PCAP or PCAP-NG captures.
    Optimized binary streaming without requiring tshark or scapy.
    """
    with open(path, "rb") as f:
        first4 = f.read(4)
        if len(first4) < 4:
            return
        magic = struct.unpack("<I", first4)[0]

        # 1. Classic libpcap format
        if magic in (0xA1B2C3D4, 0xA1B23C4D, 0xD4C3B2A1, 0x4D3CB2A1):
            endian = "<" if magic in (0xA1B2C3D4, 0xA1B23C4D) else ">"
            is_nano = magic in (0xA1B23C4D, 0x4D3CB2A1)
            ts_div = 1e9 if is_nano else 1e6
            f.read(20)  # Remainder of 24-byte global header
            while True:
                hdr = f.read(16)
                if len(hdr) < 16:
                    break
                ts_sec, ts_sub, caplen, _ = struct.unpack(f"{endian}IIII", hdr)
                data = f.read(caplen)
                if len(data) < caplen:
                    break
                yield ts_sec + ts_sub / ts_div, data

        # 2. PCAP Next Generation (PCAP-NG, Section Header Block 0x0A0D0D0A)
        elif magic == 0x0A0D0D0A:
            f.seek(0)
            endian = "<"
            interface_resol = []
            while True:
                hdr = f.read(8)
                if len(hdr) < 8:
                    break
                btype, blen = struct.unpack(f"{endian}II", hdr)
                if blen < 12:
                    break
                body = f.read(blen - 12)
                f.read(4)  # Trailing Block Total Length

                if btype == 0x0A0D0D0A:  # Section Header Block (SHB)
                    bom = struct.unpack("<I", body[:4])[0]
                    endian = "<" if bom == 0x1A2B3C4D else ">"
                    interface_resol = []
                elif btype == 0x00000001:  # Interface Description Block (IDB)
                    ts_res = 1e-6
                    opt_off = 8
                    while opt_off + 4 <= len(body):
                        code, olen = struct.unpack(f"{endian}HH", body[opt_off:opt_off + 4])
                        if code == 0:
                            break
                        val = body[opt_off + 4:opt_off + 4 + olen]
                        if code == 9 and len(val) >= 1:
                            r_val = val[0]
                            ts_res = (2 ** (-(r_val & 0x7F))) if (r_val & 0x80) else (10 ** (-r_val))
                        pad = (4 - (olen % 4)) % 4
                        opt_off += 4 + olen + pad
                    interface_resol.append(ts_res)
                elif btype == 0x00000006:  # Enhanced Packet Block (EPB)
                    if_id, ts_hi, ts_lo, caplen, origlen = struct.unpack(f"{endian}IIIII", body[:20])
                    res = interface_resol[if_id] if if_id < len(interface_resol) else 1e-6
                    ts = ((ts_hi << 32) | ts_lo) * res
                    yield ts, body[20:20 + caplen]
                elif btype == 0x00000003:  # Simple Packet Block (SPB)
                    origlen = struct.unpack(f"{endian}I", body[:4])[0]
                    caplen = min(origlen, len(body) - 4)
                    yield 0.0, body[4:4 + caplen]


def iter_game_udp_packets(path: str):
    """
    Yields (ts, src_ip, dst_ip, payload_bytes) for MetaClassroom UDP game traffic
    (port 30780 exchanged with SERVER_IP).
    """
    for ts, frame in iter_packets(path):
        if len(frame) < 14:
            continue
        # IPv4 check (EtherType 0x0800)
        if struct.unpack("!H", frame[12:14])[0] != 0x0800:
            continue
        ip = frame[14:]
        if len(ip) < 20 or ip[9] != 17:  # Protocol 17 = UDP
            continue
        ihl = (ip[0] & 0x0F) * 4
        src = ".".join(str(b) for b in ip[12:16])
        dst = ".".join(str(b) for b in ip[16:20])

        if src != SERVER_IP and dst != SERVER_IP:
            continue

        udp = ip[ihl:]
        if len(udp) < 8:
            continue
        sport, dport = struct.unpack("!HH", udp[0:4])
        if SERVER_PORT not in (sport, dport):
            continue

        payload = udp[8:]
        yield ts, src, dst, payload


# ---------------------------------------------------------------------------
# Traffic Extraction & Classification
# ---------------------------------------------------------------------------

def extract_game_traffic(pcap_files=None, cache_csv: Path = None):
    """
    Parses PCAP files and extracts all game UDP packets classified by:
    - timestamp
    - headset_ip
    - direction ('upstream' vs 'downstream')
    - payload_len
    - is_heartbeat
    - msg_type
    
    If cache_csv exists and contains data, loads from cache; otherwise extracts from PCAPs.
    """
    if cache_csv is None:
        cache_csv = DATA_DIR / "classified_traffic.csv"

    if cache_csv and cache_csv.exists() and cache_csv.stat().st_size > 100:
        print(f"Loading cached game traffic ...")
        df = pd.read_csv(cache_csv)
        if len(df) > 0:
            return df

    if pcap_files is None:
        pcap_files = get_pcap_files()

    if not pcap_files:
        raise FileNotFoundError(f"No PCAP files found in {DATA_DIR}")

    print(f"Parsing game traffic from {len(pcap_files)} PCAP file(s)...")
    records = []
    for path in pcap_files:
        print(f"  Reading {path}...")
        for ts, src, dst, payload in iter_game_udp_packets(path):
            direction = "downstream" if src == SERVER_IP else "upstream"
            headset_ip = dst if direction == "downstream" else src

            if headset_ip not in HEADSET_IPS:
                continue

            pay_len = len(payload)
            msg_type = payload[0] if pay_len >= 1 else -1

            # Strict heartbeat opcode and payload length matching
            # Server-initiated: 3 or 11 bytes. Client-initiated: 18 bytes.
            is_heartbeat = (msg_type in (MSG_PING, MSG_PONG)) and (pay_len in (3, 11, 18))

            records.append((ts, headset_ip, direction, pay_len, is_heartbeat, msg_type))

    df = pd.DataFrame(
        records,
        columns=["timestamp", "headset_ip", "direction", "payload_len", "is_heartbeat", "msg_type"]
    )
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    if cache_csv and len(df) > 0:
        cache_csv.parent.mkdir(parents=True, exist_ok=True)
        print(f"Caching classified traffic to {cache_csv} ...")
        df.to_csv(cache_csv, index=False)

    return df


# ---------------------------------------------------------------------------
# Heartbeat Matching, RTT Decomposition, and Loss Audit
# ---------------------------------------------------------------------------

def extract_heartbeats_and_rtt(pcap_files=None, max_rtt_sec: float = 2.0):
    """
    Decouples client-initiated (c2s) and server-initiated (s2c) heartbeat pairs
    by matching transaction IDs (bytes 1-2) across the middlebox.

    Returns:
    - df_rtt: DataFrame of matched RTT samples (timestamp, headset_ip, direction, rtt_ms)
    - df_loss: DataFrame auditing pings sent, pongs received, lost heartbeats, and loss %
    """
    if pcap_files is None:
        pcap_files = get_pcap_files()

    pending_c2s = collections.defaultdict(lambda: collections.defaultdict(collections.deque))
    pending_s2c = collections.defaultdict(lambda: collections.defaultdict(collections.deque))

    audit_stats = collections.defaultdict(lambda: {
        "c2s": {"sent": 0, "matched": 0},
        "s2c": {"sent": 0, "matched": 0}
    })

    rtt_samples = []

    for path in pcap_files:
        for ts, src, dst, payload in iter_game_udp_packets(path):
            if len(payload) < 3:
                continue
            msg_type = payload[0]
            if msg_type not in (MSG_PING, MSG_PONG):
                continue

            pid = payload[1:3]
            if src == SERVER_IP:
                headset_ip = dst
                direction = "s2c" # Server ping or server pong
            else:
                headset_ip = src
                direction = "c2s" # Client ping or client pong

            if headset_ip not in HEADSET_IPS:
                continue

            if msg_type == MSG_PING:
                audit_stats[headset_ip][direction]["sent"] += 1
                if direction == "c2s":
                    pending_c2s[headset_ip][pid].append(ts)
                else:
                    pending_s2c[headset_ip][pid].append(ts)
            else:  # MSG_PONG
                if direction == "s2c":  # Server responding to client ping
                    q = pending_c2s[headset_ip][pid]
                    if q:
                        ping_ts = q.popleft()
                        diff_sec = ts - ping_ts
                        if 0 <= diff_sec <= max_rtt_sec:
                            audit_stats[headset_ip]["c2s"]["matched"] += 1
                            rtt_samples.append({
                                "timestamp": ts,
                                "headset_ip": headset_ip,
                                "direction": "c2s",
                                "rtt_ms": diff_sec * 1000.0
                            })
                else:  # Client responding to server ping
                    q = pending_s2c[headset_ip][pid]
                    if q:
                        ping_ts = q.popleft()
                        diff_sec = ts - ping_ts
                        if 0 <= diff_sec <= max_rtt_sec:
                            audit_stats[headset_ip]["s2c"]["matched"] += 1
                            rtt_samples.append({
                                "timestamp": ts,
                                "headset_ip": headset_ip,
                                "direction": "s2c",
                                "rtt_ms": diff_sec * 1000.0
                            })

    df_rtt = pd.DataFrame(rtt_samples)
    if not df_rtt.empty:
        df_rtt.sort_values("timestamp", inplace=True)
        df_rtt.reset_index(drop=True, inplace=True)

    loss_records = []
    for ip in HEADSET_IPS:
        for direction in ["c2s", "s2c"]:
            sent = audit_stats[ip][direction]["sent"]
            matched = audit_stats[ip][direction]["matched"]
            lost = max(0, sent - matched)
            loss_pct = (lost / sent * 100.0) if sent > 0 else 0.0
            loss_records.append({
                "headset_ip": ip,
                "direction": direction,
                "pings_sent": sent,
                "pongs_received": matched,
                "lost_heartbeats": lost,
                "loss_pct": round(loss_pct, 3)
            })

    df_loss = pd.DataFrame(loss_records)
    return df_rtt, df_loss


# ---------------------------------------------------------------------------
# Throughput Time Series Aggregation
# ---------------------------------------------------------------------------

def compute_throughput_timeseries(df_all: pd.DataFrame, bin_sec: int = BIN_SEC):
    """
    Computes per-headset and aggregate throughput time series in kbps,
    excluding heartbeat control frames.
    """
    t0 = df_all["timestamp"].min()
    df_data = df_all[~df_all["is_heartbeat"]].copy()
    df_data["bin"] = ((df_data["timestamp"] - t0) // bin_sec).astype(int)

    # Per-headset throughput
    per_hs = df_data.groupby(["bin", "direction", "headset_ip"])["payload_len"].sum().reset_index()
    per_hs["kbps"] = per_hs["payload_len"] * 8.0 / (bin_sec * 1000.0)
    per_hs["t_min"] = per_hs["bin"] * bin_sec / 60.0

    # Aggregate stats across headsets (mean, std, min, max)
    stats_hs = per_hs.groupby(["bin", "direction", "t_min"])["kbps"].agg(
        mean="mean",
        std="std",
        min="min",
        max="max"
    ).reset_index()

    # Total aggregate throughput
    total_agg = df_data.groupby(["bin", "direction"])["payload_len"].sum().reset_index()
    total_agg["kbps"] = total_agg["payload_len"] * 8.0 / (bin_sec * 1000.0)
    total_agg["t_min"] = total_agg["bin"] * bin_sec / 60.0

    # Active headset count per bin
    active_hs = df_data.groupby("bin")["headset_ip"].nunique().reset_index(name="active_headsets")
    active_hs["t_min"] = active_hs["bin"] * bin_sec / 60.0

    return per_hs, stats_hs, total_agg, active_hs


# ---------------------------------------------------------------------------
# Statistical Distribution Modeling (GMM & Log-Normal)
# ---------------------------------------------------------------------------

def fit_downstream_gmm(df_all: pd.DataFrame, n_components: int = 4, random_state: int = 42):
    """
    Fits 4-component Gaussian Mixture Model (GMM) to downstream packet inter-arrival times.
    Returns:
    - gmm_model: fitted sklearn GaussianMixture
    - gmm_modes_df: sorted summary table of modes (mean, std, weight, approx freq)
    - delta_down_clean: cleaned downstream inter-arrival array (5 ms to 200 ms)
    """
    df_down = df_all[df_all["direction"] == "downstream"].copy()
    df_down["delta_t_ms"] = df_down.groupby("headset_ip")["timestamp"].diff() * 1000.0

    delta_down = df_down["delta_t_ms"].dropna()
    delta_down_clean = delta_down[(delta_down >= 5.0) & (delta_down <= 200.0)].values

    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(delta_down_clean.reshape(-1, 1))

    # Sort parameters by mode mean
    order = np.argsort(gmm.means_.flatten())
    means = gmm.means_.flatten()[order]
    stds = np.sqrt(gmm.covariances_.flatten())[order]
    weights = gmm.weights_.flatten()[order]

    modes_df = pd.DataFrame({
        "Mode": [f"Mode {i+1}" for i in range(n_components)],
        "Mean (ms)": np.round(means, 2),
        "Std (ms)": np.round(stds, 2),
        "Weight (%)": np.round(weights * 100.0, 2),
        "Approx Freq (Hz)": np.round(1000.0 / means, 1)
    })

    return gmm, modes_df, delta_down_clean, (means, stds, weights)


def fit_payload_distributions(df_all: pd.DataFrame):
    """
    Fits Log-Normal distribution to downstream application payload sizes.
    Returns (shape, loc, scale), downstream payloads, upstream payloads.
    """
    s_down = df_all[(df_all["direction"] == "downstream") & (~df_all["is_heartbeat"])]["payload_len"].values
    s_up = df_all[(df_all["direction"] == "upstream") & (~df_all["is_heartbeat"])]["payload_len"].values

    shape_d, loc_d, scale_d = stats.lognorm.fit(s_down[s_down > 0])

    return (shape_d, loc_d, scale_d), s_down, s_up


def evaluate_ks_tests(delta_down_clean, gmm_params_ignored, s_down, lognorm_params_ignored, sample_size=5000, random_state=42):
    """Evaluates Kolmogorov-Smirnov goodness-of-fit tests using a train/test split."""
    from sklearn.model_selection import train_test_split
    from sklearn.mixture import GaussianMixture

    # 1. GMM Train/Test Split
    ipt_train, ipt_test = train_test_split(delta_down_clean, test_size=0.5, random_state=random_state)
    
    gmm = GaussianMixture(n_components=4, random_state=random_state)
    gmm.fit(ipt_train.reshape(-1, 1))
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_.flatten()
    
    def gmm_cdf(x):
        res = np.zeros_like(x, dtype=float)
        for m, s, w in zip(means, stds, weights):
            res += w * stats.norm.cdf(x, loc=m, scale=s)
        return res

    np.random.seed(random_state)
    n_ipt = min(sample_size, len(ipt_test))
    ipt_sub = np.random.choice(ipt_test, size=n_ipt, replace=False)
    ks_stat_gmm, p_val_gmm = stats.kstest(ipt_sub, gmm_cdf)

    # 2. Log-Normal Train/Test Split
    pos_payload = s_down[s_down > 0]
    pay_train, pay_test = train_test_split(pos_payload, test_size=0.5, random_state=random_state)
    
    shape_d, loc_d, scale_d = stats.lognorm.fit(pay_train)
    
    n_pay = min(sample_size, len(pay_test))
    pay_sub = np.random.choice(pay_test, size=n_pay, replace=False)
    ks_stat_lognorm, p_val_lognorm = stats.kstest(
        pay_sub,
        "lognorm",
        args=(shape_d, loc_d, scale_d)
    )

    return (ks_stat_gmm, p_val_gmm), (ks_stat_lognorm, p_val_lognorm)
