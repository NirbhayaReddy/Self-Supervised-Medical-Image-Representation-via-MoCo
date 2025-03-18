# Self-Supervised Medical Image Representation via MoCo

This project delves into the application of **Self-Supervised Learning (SSL)** techniques for medical image representation using the **CheXpert** dataset. By leveraging SSL frameworks such as **MoCo (Momentum Contrast)**, **SimCLR**, **BYOL**, and **SwAV**, the project aims to classify multiple thoracic diseases effectively while minimizing reliance on labeled data. It also addresses challenges like class imbalance and uncertainty in medical imaging datasets.

---

## Repository Overview

This repository contains the following files:

- **`CV_Final_Project.ipynb`**: A Jupyter notebook implementing the complete pipeline for training SSL models and fine-tuning them for multi-label classification on the CheXpert dataset.
- **`train_split.csv`**: Training split of the CheXpert dataset with associated labels.
- **`test_split.csv`**: Test split of the CheXpert dataset with associated labels.
- **`requirements.txt`**: List of dependencies required to run the project.

---

## Dataset

The project utilizes the **CheXpert** dataset, which contains chest X-rays annotated for 14 conditions. Due to data-sharing restrictions, users must download the dataset separately. The dataset includes multi-label annotations where each image may be associated with multiple pathologies. 

### Key Dataset Features:

- **Uncertainty Labels**: Labels can be positive, negative, or uncertain. This project treats uncertain labels as negative (U-Zeros approach).

- **Class Imbalance**: Some conditions (e.g., pneumonia) are underrepresented compared to others (e.g., pleural effusion). Techniques like WeightedRandomSampler and Focal Loss are used to address this imbalance.

- **Preprocessing**: Images are resized to 224x224 pixels, denoised, normalized using ImageNet statistics, and augmented with transformations like random flipping, rotation, and brightness adjustments.


| Class Name                 | Description                 |
|----------------------------|-----------------------------|
| No Finding                 | Normal chest X-ray          |
| Enlarged Cardiomediastinum | Enlarged mediastinum area   |
| Cardiomegaly               | Enlarged heart              |
| Lung Opacity               | Abnormal lung opacity       |
| Lung Lesion                | Lesions in lung tissue      |
| Edema                      | Fluid accumulation in lungs |
| Consolidation              | Solidification in lungs     |
| Pneumonia                  | Lung infection              |
| Atelectasis                | Collapsed lung tissue       |
| Pneumothorax               | Air in pleural cavity       |
| Pleural Effusion           | Fluid in pleural cavity     |
| Pleural Other              | Other pleural abnormalities |
| Fracture                   | Bone fractures              |
| Support Devices            | Medical devices visible     |

---

## Key Features

### 1. Preprocessing Pipeline
- **Image Processing**: Includes resizing, denoising, histogram equalization, and normalization for optimal model input.
- **Handling Missing Data**: Uncertain labels are treated as negative to simplify training.
- **Data Augmentation**: Model-specific augmentations such as random cropping, color jitter, Gaussian blur, and multi-crop (for SwAV) enhance robustness.

### 2. Self-Supervised Learning Methods
This project implements four prominent SSL frameworks:
- **MoCo (Momentum Contrast)**: Uses a memory bank to generate negative pairs and a momentum encoder for stable updates.
- **SimCLR**: Relies on strong augmentations for contrastive learning with in-batch negative samples.
- **BYOL (Bootstrap Your Own Latent)**: Avoids negative pairs by enforcing consistency between two augmented views of an image.
- **SwAV (Swapping Assignments between Views)**: Employs clustering-based learning with multi-crop augmentations.

### 3. Class Imbalance Mitigation
To address class imbalance issues:
- **WeightedRandomSampler**: Adjusts sampling probabilities for underrepresented classes.
- **Focal Loss**: Focuses on hard-to-classify samples to improve recall for minority classes.

### 4. Optimization Techniques
The following optimization strategies were implemented:
- **Mixed Precision Training (AMP)**: Reduces memory usage and accelerates training.
- **AdamW Optimizer**: Enhances weight regularization during training.
- **OneCycleLR Scheduler**: Dynamically adjusts learning rates for stable convergence.
- **Gradient Clipping**: Prevents exploding gradients during backpropagation.

### 5. Interpretability with Grad-CAM
Grad-CAM visualizations were used to interpret model predictions by highlighting regions of interest in chest X-rays that influenced classification decisions. This ensures that the model focuses on clinically relevant features.

---

## Results

### Evaluation Metrics
The models were evaluated using:
- **AUC (Area Under the Curve)**: Measures overall performance across all thresholds.
- **F1 Score**: Balances precision and recall.
- **Cohen's Kappa**: Assesses agreement between model predictions and ground truth labels.
- **Average Precision**: Highlights precision for minority classes.

### Key Findings
1. BYOL demonstrated robust performance across all data sizes but excelled particularly in low-data scenarios.
2. MoCo achieved the highest AUC when trained on 100% labeled data due to its memory bank mechanism that maintains long-term consistency in representation learning.
3. Grad-CAM visualizations confirmed that models focused on clinically relevant regions during predictions.

---

## Requirements

Install the dependencies using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

---

## Usage Instructions

1. Download the CheXpert dataset and place it in your working directory.
2. Ensure `train_split.csv` and `test_split.csv` are located in the same directory as the notebook.
3. Run the Jupyter notebook `CV_Final_Project.ipynb`, following step-by-step instructions to train SSL models and evaluate their performance.

---

## Future Scope

This project opens avenues for further exploration:
- **Segmentation Tasks**: Incorporating segmentation during preprocessing could isolate clinically relevant regions (e.g., lungs, heart) to improve model accuracy.
- **Multi-Modality Analysis**: Combining chest X-rays with other imaging modalities (e.g., CT scans) could enhance diagnostic accuracy further.
- **Real-Time Deployment**: Developing frameworks for real-world clinical implementation would validate robustness and utility.

---

## Limitations

While promising results were achieved, certain limitations remain:
- Computational resources constrained state-of-the-art performance.
- Extensive hyperparameter tuning or larger batch sizes were not explored due to resource limitations.
With advanced computational resources, this project could achieve significant improvements and become an ideal solution for medical image representation.

---

## Authors

This project was developed as part of NYU Computer Vision CS-GY 6643:

- Nirbhaya Reddy G  
- Chirag Mahajan  
- Mohammed Basheeruddin  
- Shubham Goel  

---

## References

1. **CheXpert Dataset**  
   Stanford Machine Learning Group. *CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison.*  
   [https://stanfordmlgroup.github.io/competitions/chexpert/](https://stanfordmlgroup.github.io/competitions/chexpert/)

2. **Momentum Contrast (MoCo)**  
   He, Kaiming, et al. *Momentum Contrast for Unsupervised Visual Representation Learning.*  
   [https://arxiv.org/abs/1911.05722](https://arxiv.org/abs/1911.05722)

3. **SimCLR**  
   Chen, Ting, et al. *A Simple Framework for Contrastive Learning of Visual Representations.*  
   [https://arxiv.org/abs/2002.05709](https://arxiv.org/abs/2002.05709)

4. **BYOL (Bootstrap Your Own Latent)**  
   Grill, Jean-Bastien, et al. *Bootstrap Your Own Latent - A New Approach to Self-Supervised Learning.*  
   [https://arxiv.org/abs/2006.07733](https://arxiv.org/abs/2006.07733)

5. **SwAV (Swapping Assignments between Views)**  
   Caron, Mathilde, et al. *Unsupervised Learning of Visual Features by Contrasting Cluster Assignments.*  
   [https://arxiv.org/abs/2006.09882](https://arxiv.org/abs/2006.09882)