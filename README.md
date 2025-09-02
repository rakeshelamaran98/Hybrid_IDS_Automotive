# Hybrid IDS for Automotive

This repository contains the source code, python files, datasets, and results.

## Files
- Python scripts for preprocessing, attacks, training, and evaluation.  
- CAN bus and V2X attack datasets (preprocessed).  
- Reports and visualisations (confusion matrices, ROC, bar charts).  
- Trained ML models (`.pkl`).

## 📂 Project Structure
```
project/
├── logs/ # Raw CAN logs (DoS, fuzzing, spoofing, replay, stealth, uds, normal) (Kindly Check Google Drive Link)
├── preprocessed_csv/ # Processed CSV datasets (Kindly Check Google Drive Link)
├── scripts/ # Python scripts for attacks, preprocessing, training, IDS
├── results/ # Models (Kindly Check Google Drive Link), reports, confusion matrices, metrics, figures
├── v2x/ # V2X spoofing & replay scripts and datasets
├── compare_ids.py # Binary vs Hybrid vs Rule comparison
├── parse.py # Log parsing utility
```

## Dataset Links

Note: Due to large file sizes, datasets must be stored externally (Google Drive, Dropbox, etc.) and linked here

- Logs - https://drive.google.com/drive/folders/13YxvNKhaWvaBA7b2hwM8LXW-oH-_gOxk?usp=sharing
- preprocessed_csv datasets - https://drive.google.com/drive/folders/13YxvNKhaWvaBA7b2hwM8LXW-oH-_gOxk?usp=sharing
- results/models - https://drive.google.com/drive/folders/13YxvNKhaWvaBA7b2hwM8LXW-oH-_gOxk?usp=sharing

## How to Run
To reproduce the experiments and results:  
Check **[`instructions.md`](instructions.md)** for detailed step-by-step guidance.  

---

## Requirements
- Python 3.9+  
- Virtual environment recommended (`python3 -m venv .venv`).  
- Dependencies are listed in `requirements.txt`.


## Author
Developed by **Rakesh Elamaran**  
