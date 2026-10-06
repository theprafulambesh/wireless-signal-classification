# Wireless Signal Classification

Deep-learning project for wireless signal and modulation classification.

## Current status

The project is currently in the channel-simulator validation stage.

Completed or partially completed components include:

- Digital signal generation
- RRC pulse shaping
- AWGN channel
- Multipath channel
- Rayleigh fading
- Carrier-frequency offset
- Phase offset
- Timing offset
- Sampling-rate offset

Dataset generation and CNN training have not started yet.

## Project structure

```text
configs/       Configuration files
data/          Generated datasets; not committed
experiments/   Experiment artifacts; not committed
models/        Trained models; not committed
notebooks/     Exploratory notebooks
results/       Generated plots and results; not committed
src/           Project source code

## Environment setup

```bash
python -m venv wsc
source wsc/Scripts/activate
pip install -r requirments.txt

## Planned work
1. Finish validating individual channel impairments
2. Validate combined impairments
3. Generate the IQ dataset
4. Build and train the CNN
5. Evaluate modulation-classification performance
