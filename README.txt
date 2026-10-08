Multiplexing AI model demo

Repository
https://github.com/LanghammerLab-projects/multiplexing-ml-demo

What it does
Runs the included trained transformer on the real preprocessed 64-pixel
optical sensor sequence and writes a prediction/attention plot for the
held-out Jakarta day example. The repository includes the Python source,
checkpoint and sample data; no separate demo attachment is needed.

1. System requirements
Python with NumPy, Matplotlib and PyTorch. The tested version combination is:
  Python 3.9.7, 64-bit
  NumPy 1.26.3
  Matplotlib 3.4.3
  PyTorch 2.3.0 (the tested installation used build 2.3.0+cu118)
  Microsoft Windows 11 Home, version 10.0.26200, 64-bit

The CPU demo test on 8 October 2026 succeeded with this combination. Other
operating systems and dependency versions have not been verified for this
checklist. The direct dependencies are recorded in requirements.txt and their
tested versions in requirements-tested.txt; pip resolves their dependencies.

No non-standard hardware is required. CPU is the default device. A CUDA-capable
NVIDIA GPU is optional, with a compatible PyTorch/CUDA installation.

2. Installation guide
Clone this repository or download and extract it using GitHub's Code menu.
Open a terminal in the repository folder. With Python 3.9, run:
  python -m pip install -r requirements-tested.txt

This installs the tested direct dependency versions. The test used an existing
CUDA-enabled PyTorch 2.3.0+cu118 installation but executed inference on the CPU.
The original unpinned installation command is also available:
  python -m pip install -r requirements.txt

Typical installation time is estimated at 2-10 minutes on a normal desktop
or laptop with internet access; usually under one minute if PyTorch is already
installed. This is an estimate, not a fresh-install benchmark, and depends on
internet speed and package availability.

3. Demo
From the repository folder:
  python run_demo.py

To select the tested device explicitly:
  python run_demo.py --device cpu

Input: Preprocessed Data/jakarta_demo_sequence.npz
Checkpoint: model_checkpoint.pt
Expected output: demo_predictions.png

The colored traces are normalized model predictions, the black dashed traces
are normalized gas targets, and the bottom panel is attention over the 64
optical readout pixels. The supplied data's gas labels, in output order, are
H2, CO, CO2 and NO2.

Measured demo runtime: approximately 12 seconds on an Intel Core i7-10750H
laptop CPU at 2.60 GHz with the tested environment, including imports, full
sample-sequence inference and PNG generation. Allow roughly one minute on a
comparable desktop; slower hardware or larger inputs can take longer.
Installation time is additional. No GPU was used for this test.

4. Instructions for use with your own data
The demo reads already preprocessed, model-compatible NumPy .npz files.
Prepare these arrays:
  X: float32, shape (T, 64), optical inputs in the checkpoint's training
     normalization and physical pixel order. The standalone script does not
     standardize these inputs internally.
  y: float32, shape (T, 4), time-aligned gas targets in the checkpoint's
     normalized target scales, used for the target-versus-prediction plot.
  plot_mask: bool, shape (T,), selecting points shown in the plot.
  plot_time_hours: float32, shape (sum(plot_mask),), selected-point times
     in hours and in the same order as plot_mask.
  label: scalar Unicode string naming the measurement.
  gas_names: Unicode string array, shape (4,), in checkpoint output order.

The supplied example contains T=35,481 points and 11,836 selected plot points.
Its raw_indices array is included for reference and is not used by this demo.
Preserve pixel order, target order, units and training normalization.

Run a prepared custom file:
  python run_demo.py --data path/to/your_sequence.npz --output your_predictions.png

To use a different compatible trained checkpoint:
  python run_demo.py --data path/to/your_sequence.npz --checkpoint path/to/model.pt

The checkpoint architecture, normalization and gas output order must match
the data. This inference demo does not train a new model or provide a workflow
for reproducing all quantitative manuscript results.

Licence
The software and accompanying software documentation are distributed under
the MIT License; see LICENSE. Reuse, modification and redistribution are
permitted provided the copyright and licence notice are retained.
