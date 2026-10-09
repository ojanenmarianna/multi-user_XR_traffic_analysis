# Multi-User XR Traffic Analysis Pipeline

This repository contains the passive traffic analysis pipeline used to analyze MetaClassroom session data and model the traffic dynamics of multi-user XR experiences.

## Overview
The provided code processes network packet headers to infer protocol behavior, decompose round-trip times (RTT) across WAN and wireless hops, characterize throughput, and build generative statistical models (such as Gaussian Mixture Models) for downstream and upstream payloads.

## Files
- `analysis_pipeline.ipynb`: The primary Jupyter Notebook walking through data loading, filtering, statistical modeling, and plotting.
- `helpers.py`: Python module containing shared helper functions for flow extraction, packet parsing, modeling, and plotting used by the notebook.
- `requirements.txt`: Python package dependencies.

## Setup Instructions

1. **Environment Setup:** It is recommended to use a virtual environment.
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Data Directory:** Place your PCAP files or processed CSV/Pickle data frames in a `data/` directory (or modify the path in the notebook).

3. **Run Notebook:** Start the Jupyter Notebook server.
   ```bash
   jupyter notebook analysis_pipeline.ipynb
   ```

## Note on Privacy
The analysis operates on encrypted transport-layer packet headers, timings, and byte sizes. No payload decryption or deep packet inspection of voice/video/application logic is performed.
