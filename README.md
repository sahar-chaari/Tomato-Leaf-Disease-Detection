# 🍅 Tomato Leaf Disease Detection

🔍 **Overview**

This project aims to detect diseases in tomato leaves using three different machine learning models: **YOLO11** , **YOLO12** and **ResNet50**. The models are trained on the [Tomato Leaf Disease Dataset](https://universe.roboflow.com/bryan-b56jm/tomato-leaf-disease-ssoha/dataset/63) from **Roboflow**. The goal is to assist farmers in identifying and managing plant diseases more efficiently using AI.

---

🚀 **Features**

- 🍃 **Multiple Models**: Three models for disease detection — YOLOv11m, YOLOv12m and ResNet50.
- 🌾 **Training Notebooks**: Jupyter Notebooks for training each model on the provided dataset.
- 📊 **Dataset**: Dataset from Roboflow with various tomato leaf diseases, including bacterial spot, early blight, late blight, and more.

---

🛠️ **Installation**

**1. Clone the Repository**

```bash
git clone https://github.com/yourusername/Tomato-Leaf-Disease-Detection.git
cd Tomato-Leaf-Disease-Detection
```

**2. Create a Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows
```

**3. Install Dependencies**

Install necessary libraries manually based on the needs of your project (e.g., PyTorch, TensorFlow, OpenCV, etc.). Below is a general list of dependencies to get started:

- `torch` (for PyTorch models)
- `torchvision` (for image transformations)
- `roboflow` (for loading the dataset)
- `opencv-python` (for image processing)
- `pandas` (for data manipulation)
- `matplotlib` (for visualizations)

---

📁 **Project Structure**

```
Tomato-Leaf-Disease-Detection/
├── models/
│   ├── YOLOv11m.pt          # YOLOv11 model
│   ├── YOLOv12m.pt         # YOLOv12 model
│   └── ResNet50.py         # ResNet50 model
├── notebooks/
│   ├── YOLOv11m.ipynb    # Training notebook for YOLOv11
│   ├── YOLOv12m.ipynb    # Training notebook for YOLOv12
│   └── ResNet50.ipynb    # Training notebook for ResNet50
├── dataset/
│   └── README.md                # Dataset details and Roboflow download link              
└── README.md              # Project overview
```

---

💡 **How to Train Models**

- Each model has a corresponding training notebook:
    - `YOLOv11m.ipynb`: Train the YOLOv11 model on the dataset.
    - `YOLOv12m.ipynb`: Train the YOLOv12 model on the dataset.
    - `ResNet50.ipynb`: Train the ResNet50 model on the dataset.

Follow the instructions in each notebook to run the training process.

---

📝 **Dataset**

The dataset is sourced from [Roboflow: Tomato Leaf Disease Dataset](https://universe.roboflow.com/bryan-b56jm/tomato-leaf-disease-ssoha/dataset/63). For details on the dataset and how to access it, check the `dataset/README.md` file.

---

💡 **Future Improvements**

- 📈 Fine-tune the models for better accuracy with more data.
- 🌍 Deploy the models for real-time disease detection on mobile applications.
- 🔄 Experiment with additional models (e.g., Faster R-CNN, EfficientNet).
- 📉 Provide detailed performance metrics for each model.

---

🔗 **Resources**

- [Roboflow Dataset](https://universe.roboflow.com/bryan-b56jm/tomato-leaf-disease-ssoha/dataset/63)
- [YOLO11](https://docs.ultralytics.com/fr/models/yolo11/)
- [ResNet50](https://keras.io/api/applications/resnet/#resnet50-function)
