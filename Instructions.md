# Hybrid IDS for Automotive Systems

This repository contains the implementation, datasets, and evaluation results for a **Hybrid Intrusion Detection System (IDS)** combining rule-based and machine learning approaches for CAN bus and V2X networks.

### This Project is created and owned by Rakesh Elamaran for academic purposes.
---

## 📂 Project Structure

```
project/
├── logs/ # Raw CAN logs (DoS, fuzzing, spoofing, replay, stealth, uds, normal)
├── preprocessed_csv/ # Processed CSV datasets
├── scripts/ # Python scripts for attacks, preprocessing, training, IDS
├── results/ # Models, reports, confusion matrices, metrics, figures
├── v2x/ # V2X spoofing & replay scripts and datasets
├── compare_ids.py # Binary vs Hybrid vs Rule comparison
├── parse.py # Log parsing utility
```

## Setup

```

# Clone repo
git clone https://github.com/rakeshelamaran98/Hybrid_IDS_Automotive.git
cd Hybrid_IDS_Automotive

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

```

# Step-By-Step Instructions

## Step 1: Collect CAN Logs (Attack Simulation)

Run ICSim or your custom CAN setup. Example logs are already in logs/:

dos/ → flooding attack

fuzzing/ → random/semi-random payloads

spoofing/ → false signals

replay/ → replayed valid frames

stealth/ → low-frequency injection

uds/ → diagnostic misuse

normal/ → benign baseline

Example:

candump vcan0 > logs/dos/dos_YYYYMMDD_HHMMSS.log


## Step 2: Preprocess Logs → CSV

```
python scripts/preprocess_to_csv.py --input logs/ --output preprocessed_csv/
```

Generates datasets like:

dos.csv, spoofing.csv, replay.csv, stealth.csv, uds.csv, normal.csv

Combined datasets:
all_attacks_combined.csv, all_with_rules.csv, all_with_rules_with_iat.csv

## Step 3: Rule-Based IDS

Run threshold-based IDS (DoS, replay, UDS easily detectable):

```
python scripts/rule_based_ids.py --input preprocessed_csv/all_with_rules.csv
```
Output: alerts + summary metrics

## Step 4: Train Machine Learning IDS

```
python scripts/train_ids.py --input preprocessed_csv/all_attacks_combined.csv

```
Models saved in results/models/:

RandomForest_binary.pkl
ExtraTrees_multiclass.pkl
etc.

Reports saved in results/reports/:
Confusion matrices (*_cm.csv)
Classification reports (*_report.txt)

## Step 5: Hybrid IDS Evaluation

Run combined rule + ML evaluation:

```
python scripts/hybrid_eval.py --input preprocessed_csv/all_with_rules_with_iat.csv
```

Outputs in results/hybrid/reports/:

Confusion matrices (binary_HYBRID_cm.csv, multiclass_HYBRID_cm.csv)
Metrics comparison (metrics_binary_ml_vs_hybrid.csv, metrics_multiclass_ml_vs_hybrid.csv)
Figures: ROC, feature importance, comparative bar charts

## Step 6: Comparative Analysis

Generate bar charts + LaTeX tables:

```
python compare_ids.py
```

Outputs:
binary_ids_comparison.png
multiclass_ids_comparison.png

## Step 7: V2X Spoofing & Replay

Inside v2x/:

Generate normal & malicious traffic:

```
python v2x/v2x_generate.py
python v2x/v2x_attack_spoof.py
python v2x/v2x_attack_replay.py
```
Detect anomalies:

```
python v2x/v2x_detect.py
```

Datasets: v2x_normal.csv, v2x_attack_spoof.csv, v2x_attack_replay.csv

## Results
All results (confusion matrices, metrics, figures) are available in results/.

## Reproducability

Reproducibility

Use provided logs (logs/) or generate new ones via ICSim.
Preprocess → preprocessed_csv/
Run rule-based IDS (rule_based_ids.py)
Train ML IDS (train_ids.py)
Evaluate Hybrid IDS (hybrid_eval.py)
Compare (compare_ids.py)
Extend to V2X with v2x/ scripts


--X--
