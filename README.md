# Image Classification with EfficientNet and DenseNet

## Overview

This repository contains code for image classification tasks using two powerful deep learning models: **EfficientNet** and **DenseNet**. The implementation utilizes TensorFlow and Keras to preprocess images, train models, and evaluate performance on a dataset stored in Parquet format.

## Features

- **Data Processing**: Load and preprocess images from a Parquet file.
- **Model Training**: Train image classification models using EfficientNetB0 and DenseNet201 architectures.
- **Performance Evaluation**: Evaluate models based on accuracy, precision, recall, F1 score, RMSE, MAE, and MAPE.
- **Visualization**: Display training history and confusion matrix for model performance analysis.

## Technology Stack

- **Programming Language**: Python
- **Libraries**: 
  - Pandas for data manipulation
  - NumPy for numerical operations
  - TensorFlow and Keras for deep learning
  - Matplotlib and Seaborn for visualization

## Usage

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Required Packages**:
   Ensure you have Python installed, then install the required packages:
   ```bash
   pip install pandas numpy tensorflow pillow scikit-learn seaborn matplotlib
   ```

3. **Prepare Your Dataset**:
   Ensure you have a dataset in Parquet format with images and corresponding labels.
   
![image](https://github.com/gulkorkut/nasctg2-transfer-learning/assets/94754805/82b2ebb3-34c6-4369-809e-5c27a546e1a7)

5. **Run the Code**:
   Execute the training scripts for EfficientNet and DenseNet models:
   ```python
   python efficientnet_classification.py
   python densenet_classification.py
   ```

6. **Analyze the Results**:
   After running the scripts, review the printed evaluation metrics and visualizations generated for training loss, accuracy, and confusion matrix.

## Code Structure

- `efficientnet_classification.py`: Contains code for loading data, preprocessing images, training the EfficientNet model, and evaluating its performance.
- `densenet_classification.py`: Contains code for the DenseNet model, similar in structure to the EfficientNet script.

## Contribution

Contributions are welcome! Please open an issue or submit a pull request for any improvements or suggestions.

## License

This project is licensed under the MIT License.


