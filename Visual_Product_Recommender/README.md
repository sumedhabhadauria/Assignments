# Visual Product Recommender

## Project Overview

The Visual Product Recommender is a deep learning based fashion recommendation system developed during the Celebal Technologies Internship.

The system recommends visually similar fashion products using ResNet50 feature extraction and FAISS similarity search.

---

## Features

- Upload a fashion product image
- Deep feature extraction using ResNet50
- Fast similarity search using FAISS
- Displays Top-K visually similar products
- Professional Streamlit web interface
- Metadata-based recommendation filtering

---

## Technologies Used

- Python
- TensorFlow / Keras
- ResNet50
- FAISS
- Streamlit
- NumPy
- Pandas

---

## Project Structure

```
Visual_Product_Recommender/

│── app.py
│── requirements.txt
│── README.md

├── Models/
├── Outputs/
├── images/
├── Notebooks/
```

---

## Installation

Install the required libraries

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## Workflow

1. Upload product image
2. Extract ResNet50 features
3. Search similar products using FAISS
4. Display Top-K recommendations

---

## Developed By

**Sumedha Bhadauria**
Celebal Technologies Internship Project