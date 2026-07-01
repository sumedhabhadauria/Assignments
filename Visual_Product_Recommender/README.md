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

## Note

Some generated files and the dataset are not included in this repository because they exceed GitHub's storage and file size limits.

The following files and folders have been excluded:

- images/ (Fashion Product Images dataset - 44,441 images, approximately 13.8 GB)
- Models/faiss_index.index
- Models/siamese_model.keras
- Outputs/embeddings.npy

To run the complete project locally:

1. Download and place the Fashion Product Images dataset inside the `images/` folder.
2. Generate the required model and index files by running the notebooks in the `Notebooks/` folder in sequence.

---

## Developed By

**Sumedha Bhadauria**
Celebal Technologies Internship Project