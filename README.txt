Multiplexing AI model demo

What it does
Runs a trained transformer on the included preprocessed 64-pixel optical sensor sequence and writes a prediction/attention plot for the held-out Jakarta day example.

Requirements
Python 3.9+ with numpy, matplotlib and torch.

Install
From this folder:
python -m pip install -r requirements.txt

Expected install time
Usually 2-10 minutes on a normal laptop with internet access. If PyTorch is already installed, typically under 1 minute.

Run
From this folder:
python run_demo.py

Expected output
demo_predictions.png

The colored traces are model predictions, black dashed traces are normalized gas targets, and the bottom panel is attention over the 64 optical readout pixels.
