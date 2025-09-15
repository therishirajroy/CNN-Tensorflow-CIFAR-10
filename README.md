# CNN CIFAR-10 Image Classifier (Model+App)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-ff4b4b.svg)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-FF6F00.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A simple web application built with **Streamlit** to deploy a **Convolutional Neural Network (CNN)** model trained on the CIFAR-10 dataset.

<p align="center">
  <img src="[ADD_A_SCREENSHOT_OR_GIF_OF_YOUR_APP_HERE]" alt="App Demo Screenshot" width="700"/>
</p>

---

## 📋 Table of Contents

- [About The Project](#-about-the-project)
- [Key Features](#-key-features)
- [Built With](#-built-with)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#-usage)
- [File Structure](#-file-structure)
- [License](#-license)

---

## 📖 About The Project

This project serves as a practical demonstration of deploying a deep learning model. A CNN was trained to classify images into one of the 10 classes of the **CIFAR-10 dataset**. The trained model (`cifar10nn.keras`) is then served through an interactive web interface created using the Streamlit framework.

The application allows users to upload their own images and receive a real-time classification prediction from the model.

---

## ✨ Key Features

* **Interactive UI**: A clean and simple user interface powered by Streamlit.
* **Real-Time Inference**: Upload an image and get an instant prediction from the CNN model.
* **Sidebar Navigation**: Easy navigation between the "About," "Inference," and "Student Info" pages.
* **Model Information**: The "About" page provides a brief explanation of the model architecture and the dataset.

---

## 🛠️ Built With

This project was built using the following technologies:

* **Python**: The core programming language.
* **Streamlit**: For building and deploying the web application.
* **TensorFlow / Keras**: For loading and using the pre-trained deep learning model.
* **Pillow**: For image manipulation.
* **NumPy**: For numerical operations and data preprocessing.

---

## 🏁 Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

Make sure you have Python 3.8+ and pip installed on your system.

### Installation

1.  **Clone the repository**
    ```sh
    git clone [https://github.com/therishirajroy/CNN-Tensorflow-CIFAR-10.git](https://github.com/therishirajroy/CNN-Tensorflow-CIFAR-10.git)
    ```
2.  **Navigate to the project directory**
    ```sh
    cd CNN-Tensorflow-CIFAR-10
    ```
3.  **Create a `requirements.txt` file**
    If you don't have one, create a file named `requirements.txt` and add the following lines:
    ```
    streamlit
    tensorflow
    Pillow
    numpy
    ```
4.  **(Optional but Recommended) Create a Virtual Environment**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
5.  **Install the required packages**
    ```sh
    pip install -r requirements.txt
    ```

---

## 🔧 Usage

Once the installation is complete, you can run the application with a single command:

```sh
streamlit run app.py
```

Your web browser will automatically open to the application's URL.

1.  Navigate to the **Inference** page using the sidebar.
2.  Click the "Upload an image..." button to select an image file (`.jpg`, `.jpeg`, or `.png`).
3.  The app will display the image and the model's predicted label below it.

---

## 📂 File Structure

The project is organized as follows:

```
.
├── app.py              # The main Streamlit application script
├── cifar10nn.keras     # The pre-trained Keras model file
├── requirements.txt    # List of Python dependencies
└── README.md           # This file
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
