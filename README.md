
📜 License


This project is licensed under the MIT License.

👨‍💻 Author

Badal Singh 
AI/ML Developer | Deep Learning Enthusiast

🌟 Acknowledgment

Special thanks to:

NIH Malaria Dataset
TensorFlow Community
OpenCV Contributors
📬 Contact

For queries or collaboration:

Email:singh.badal3375@gmail.com 
LinkedIn:https://www.linkedin.com/in/badalsingh91/


Malaria Detection Using Deep Learning


📌 Project Overview

This project is a Malaria Detection System built using Deep Learning (CNN - Convolutional Neural Network) to classify cell images as Parasitised or Uninfected. The model analyses microscopic blood smear images and helps in early malaria diagnosis with high accuracy.

The system can be integrated into:

Web Applications (Flask/Django/Streamlit)
Mobile Health Apps
Clinical Decision Support Systems
🎯 Objectives

Detect malaria parasites from blood smear images automatically
Reduce manual diagnostic errors
Provide faster and more accurate predictions
Assist healthcare professionals in diagnosis
🧠 Deep Learning Model Used

Model Type: Convolutional Neural Network (CNN)

Architecture:
Conv2D Layer
MaxPooling2D
Dropout Layer
Flatten Layer
Dense Layers
Output Layer (Binary Classification)
📂 Dataset

Dataset Name: Malaria Cell Images Dataset
Source: NIH / Kaggle
Classes:

Parasitized
Uninfected
Dataset Structure:
dataset/
│── train/
│   ├── Parasitized/
│   └── Uninfected/
│── test/
│   ├── Parasitized/
│   └── Uninfected/
🛠️ Technologies Used
Python
TensorFlow / Keras
OpenCV
NumPy
Matplotlib

Streamlit / Flask (for deployment)
⚙️ Installation
Step 1: Clone Repository

git clone https://github.com/yourusername/malaria-detection-deep-learning.git
cd malaria-detection-deep-learning
Step 2: Install Dependencies
pip install -r requirements.txt

Step 3: Run Training
python train.py
Step 4: Run Prediction App
streamlit run app.py
📊 Model Training
Preprocessing:
Image Resizing (128x128)
Normalization
Data Augmentation
Train-Test Split
Training Parameters:
Epochs: 10–25
Batch Size: 32
Optimiser: Adam
Loss Function: Binary Cross-Entropy
📈 Performance Metrics
Accuracy: ~90–95%
Precision
Recall
F1-Score
🚀 Features
Upload blood smear image
Real-time malaria prediction
User-friendly interface
High accuracy
Deployable on the cloud
📷 Sample Output
Prediction: Parasitised
Confidence Score: 97.5%
📁 Project Structure
malaria-detection/
│── app.py
│── train.py
│── model.h5
│── requirements.txt
│── README.md
│── dataset/
│── templates/
│── static/
🔍 Future Enhancements
Multi-class parasite stage detection
Mobile app deployment
Explainable AI (Grad-CAM visualisation)
Cloud integration for hospitals
🤝 Contribution

Contributions are welcome!

Steps:
Fork the repository
Create a new branch
Commit changes
Submit a Pull Request
📜 License

This project is licensed under the MIT License.

 
