# SkinLite: Hierarchical Capsule Network combined With Lightweight Model for Skin Lesion Classification

**SkinLite** is a deep learning framework that combines a
lightweight convolutional backbone with a hierarchical
Capsule Network (H-CapsNet) to perform multi-level skin
lesion classification
on the [ISIC 2018 Task 3 dataset](https://challenge.isic-archive.com/data/).
This project is designed to provide a benchmark of using Capsules Networks
in specific medical images dataset,
focusing on both efficiency and interpretability.

---

## 🧠 Project Highlights

- ✅ Multi-level classification (coarse, medium, fine)
- ✅ Custom hierarchical Capsule Network (H-CapsNet)
- ✅ Lightweight MobileNetV2-based feature extractor
- ✅ Custom class-balancing loader with per-class multipliers
- ✅ Dynamic gamma weighting and loss routing
- ✅ Real decoder reconstructions
- ✅ Evaluation on full ISIC 2018 test set with metrics and visualizations

---

## 📁 Project Structure
**Well-Organized Project to ease the later improvements,
the major works were in < agin / Backbone_Hcapsule folder > others were side tests are implemented,
to ensure consistency and for error handling and Debugging**
```
skinlite/
│    └── again/                              # The last and best experment folder
│         ├── Backbone_Hcapsule/
│         │   └── MH_layers.py               # MobileNetV2-based hierarchical backbone
│         │
│         ├── capsule/
│         │   ├── caps_layer.py             # PrimaryCaps, DigitCaps, Decoder definitions
│         │   └── Hcaps_data_loader.py      # Custom dataset loader with oversampling & undersampling combination
│         │
│         ├── mobile/
│         │   ├── mobile_model.py           # Baseline MobileNetV2 model
│         │   └── mobile_data_loader.py     # MobileNet-specific data pipeline
│         │
│         ├── setting.py                    # Global hyperparameters and config
│         ├── train.py                      # Training script for H-CapsNet
│         ├── test.py                       # Evaluation and metrics
│         ├── plot_metrics.py               # Visualize loss/accuracy curves
│         ├── transforms.py                 # Augmentation and preprocessing
│         └── utils.py                      # Utility functions (e.g., metrics, I/O)
│
├── Capsules_on_ISIC_dataset_report.pdf     # Final project report
├── readme.md
└── requirements.txt
```

---

##  Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/WajeehAlamoudi/skinlite.git
cd skinlite
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Follow instructions on [ISIC 2018](https://challenge.isic-archive.com/data/) to download image data and place in:

```
/dataset/images/train/
               /val/
               /test/
```

Label CSVs are already included in `/dataset/labels`.

---

## 💠 Configuration

All training and model hyperparameters are controlled via:

```python
again/setting.py
```

Adjust:
- Models: **MobileNet V2, H-Caps, Combined H-caps**
- `CLASS_MULTIPLIER` (custom oversampling & undersampling)
- `EPOCHS`, `BATCH_SIZE`, `DEVICE`
- Data paths & Optimizers configurations
- And other

---

## 🏋️‍♂️ Training
### Recommendation: run the train in Colab or environment that is ```Cuda``` powered.

```bash
python again/train.py
```

- Model checkpoints are saved when macro F1 improves
- History is logged to a `.pkl` file for future plots
- Model is saves in `.pth` and `.pt` formats in both completion and early stop scenarios
- Logs, print lines and others to keep on track with a train process
- **Can customize the H-Caps by adjusting the configurations: in_channels, kernel,
and compute the output shape to have properly fed into next layers**

---

# 📊 Evaluation

This section outlines the evaluation process for trained models in this project. Two scripts are provided:

**For trained model results, please re-view the PDF provided with project**

1. **`test.py`**: Used to evaluate models and calculate performance metrics.
2. **`plot_metrics.py`**: Used to visualize evaluation metrics.

## Running the Evaluation

To evaluate a model, use the `test.py` script with the following command:
### Note: The trained models are provided on my Google Drive account, please send me a request via email or let me know to share the models.
### Arguments:
- **`--model_path`**:
  - Type: `str`
  - **Required**: Yes
  - Description: Path to the trained model file (`.pt` or `.pth`).

- **`--model_type`**:
  - Type: `str`
  - **Required**: Yes
  - Choices: `{"Hcaps", "mobile"}`
  - Description: Specify the type of model being tested.

### Example Command:
```bash
python again/test.py --model_path models/HCapsNet.pth --model_type Hcaps
```

This command tests the Capsule Network (`Hcaps`) model stored in the file `models/HCapsNet.pth`.

Outputs:
- Classification report
- Confusion matrix
- Accuracy per hierarchy level
- Sample predictions and reconstructions

---

## 📊 Example Results

| Level        | Accuracy | Macro F1 |
|--------------|----------|----------|
| Coarse       | 77.6%    | —        |
| Medium       | 84.9%    | —        |
| Fine (7-cls) | 81.1%    | 72.3%    |

---

## 📚 Citations

If you use this repo, consider citing:

```bibtex
@article{alamoudi2025skinlite,
  title={SkinLite: Lightweight Hierarchical Capsule Network for Multi-Level Skin Lesion Classification},
  author={Abdulrahman Alamoudi},
  journal={GitHub},
  year={2025},
  note={\url{https://github.com/WajeehAlamoudi/skinlite}}
}
```

