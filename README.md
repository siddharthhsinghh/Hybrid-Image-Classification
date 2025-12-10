# Hybrid CNN–Token Based Image Classification on CIFAR-10
This repository contains an end-to-end implementation of a **hybrid CNN–token based image classification model** developed using **TensorFlow and Keras**. The project evaluates model performance on both **balanced** and **moderately imbalanced** versions of the CIFAR-10 dataset, using **class-weighted learning** and **Grad-CAM visual explanations** for interpretability.
📌 Project Overview
Traditional Convolutional Neural Networks (CNNs) perform well on small image datasets due to their ability to learn local features. However, they often struggle to capture global context. On the other hand, Vision Transformers model global relationships well but require very large datasets and heavy computation.
This project proposes a **lightweight hybrid approach** that:
- Uses a CNN backbone for **local feature extraction**
- Converts CNN feature maps into **token representations** for global context modeling
- Evaluates performance on **balanced and class-imbalanced datasets**
- Applies **class weighting** to improve minority class predictions
- Uses **Grad-CAM** for visual model explainability
🎯 Objectives
- Build a hybrid CNN–token based image classification model
- Evaluate performance on balanced and imbalanced CIFAR-10 datasets
- Study the effect of **class imbalance** on prediction performance
- Improve minority class detection using **class-weighted loss**
- Visualize important regions using **Grad-CAM**
- Perform a **comparative analysis** with baseline approaches
📂 Dataset
- **Dataset Used:** CIFAR-10
- **Total Images:** 60,000 (32×32 RGB)
- **Number of Classes:** 10
- **Train/Test Split:** 50,000 / 10,000
Balanced Dataset
- Original CIFAR-10 class distribution
Imbalanced Dataset
- Moderate imbalance created by reducing:
  - Cat (Class 3)
  - Deer (Class 4)
  - Dog (Class 5)
- These classes are reduced to **20% of their original samples**
- Other classes remain unchanged
🔄 Data Preprocessing
- Data Cleaning: Not required (clean benchmark dataset)
- Missing Values: None present
- Encoding: Integer labels (0–9)
- Normalization: Pixel values scaled to **[0, 1]**
- Data Augmentation:
  - Random Horizontal Flip
  - Random Cropping (32×32)
- Dataset Splitting:
  - 90% Training
  - 10% Validation
  - Official CIFAR-10 Test Set for evaluation
🧠 Model Architecture
The proposed model consists of:
1. **CNN Backbone**
   - Conv(32) → ReLU → MaxPooling
   - Conv(64) → ReLU → MaxPooling
   - Conv(128) → ReLU
2. **Patch Token Extraction**
   - Tokens extracted from CNN feature maps
   - Patch size: 2×2 → 16 tokens
3. **Token Embedding**
   - Dense projection to 128 dimensions
4. **Global Representation**
   - Token-wise Global Average Pooling
   - CNN Global Average Pooling
5. **Feature Fusion**
   - Concatenation of CNN and token features
6. **Classification Head**
   - Dense(128) → ReLU → Dropout(0.3)
   - Dense(10) → Softmax
⚙️ Training Details
- Framework: TensorFlow + Keras
- Optimizer: Adam
- Learning Rate: 0.001
- Loss Function:
  - Balanced Data → Sparse Categorical Cross-Entropy
  - Imbalanced Data → Weighted Cross-Entropy
- Batch Size: 128
- Epochs:
  - Demo: 8
  - Final Training: 40–80 (recommended)
- Best Model Weights Saved using `ModelCheckpoint`
📊 Evaluation Metrics
- Accuracy
- Precision
- Recall
- Macro F1-Score
- Confusion Matrix
📈 Visualizations
The following visual outputs are generated:
- Class distribution bar charts (balanced vs imbalanced)
- Training and validation accuracy curves
- Training and validation loss curves
- Confusion matrices
- Grad-CAM heatmaps for explainability
🔍 Key Findings
- Balanced training provides stable and high accuracy.
- Imbalanced training without class weighting results in poor minority-class recall.
- Applying **class weights significantly improves** F1-score and recall of minority classes.
- Grad-CAM confirms that the model focuses on meaningful object regions.
🆚 Comparative Analysis
Comparison performed between:
- Baseline CNN
- Transformer-based approach (optional)
- Proposed Hybrid CNN–Token Model
Results show:
- CNN performs well locally but lacks global context.
- Transformers require large datasets.
- The hybrid approach combines both advantages and performs better for small, imbalanced datasets.
