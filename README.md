# 🍽️ Intelligent Recipe Classification & Labeling System

![Project Banner](https://via.placeholder.com/1000x300?text=Recipe+Classification+System)

---

## Overview
This project is a **machine learning–powered pipeline** that automatically classifies and labels recipes based on **ingredients** and **cooking instructions**.

It predicts:
- 🏳 **Cuisine Type** (Italian, Indian, Chinese, etc.)
- 🥦 **Dietary Features** (Vegetarian, Spicy, Gluten-Free, High-Protein, Dairy-Free)
- ⚖ **Difficulty Level** (Easy, Medium, Hard)
- ⏱ **Estimated Cooking Time**

**Goal:** Automate recipe metadata generation to enhance **recipe search**, **personalized recommendations**, and **dietary filtering** for **health-tech, food tech, and e-commerce** platforms.

---

## Why This Project?
In many recipe databases:
- Metadata like *"Vegetarian"* or *"Gluten-Free"* is often missing or inconsistent.
- Manual tagging is slow, error-prone, and unscalable.
- Users struggle to find recipes matching **dietary needs** or **time constraints**.

This project solves that by **automatically learning patterns** from recipe text and tagging recipes with **high accuracy** — making food data more searchable and usable.

---

## Project Workflow

1. A[Dataset: Ingredients + Instructions] 
2. B[Data Cleaning & Preprocessing] 
3. C[Keyword-based Initial Tagging] 
4. D[Feature Extraction: TF-IDF, Length, Counts] 
5. E[ML Models: Logistic Regression, Random Forest, XGBoost] 
6. F[Predictions: Cuisine, Dietary Labels, Difficulty, Time] 
7. G[Evaluation & Deployment]

Features
✅ Cuisine classification using NLP models
✅ Automated dietary tagging (Vegetarian, Spicy, Gluten-Free, High-Protein, Dairy-Free)
✅ Difficulty prediction from ingredients & steps
✅ Cooking time estimation from recipe instructions
✅ Multiple models tested for maximum accuracy

Tech Stack:
- Languages & Libraries
- Python
- pandas, NumPy
- scikit-learn, XGBoost
- NLTK
- Matplotlib, Seaborn

Machine Learning Concepts:
- Supervised Learning
- Multi-class & Multi-label Classification
- TF-IDF Vectorization
- Feature Engineering
- Model Evaluation (Accuracy, Precision, Recall, F1-score)

Dataset:
Source: Food Ingredients and Recipe Dataset with Image Name Mapping
Columns: Ingredients, Instructions, Image Names
Labels Generated: Cuisine type, dietary restrictions, difficulty, time estimation
Cleaning Steps: Stopword removal, lowercasing, punctuation removal


Model Results:
| Model               | Cuisine Accuracy | Dietary Label Accuracy | Difficulty Accuracy |
| ------------------- | ---------------- | ---------------------- | ------------------- |
| Logistic Regression | 85%              | 90%                    | 88%                 |
| Random Forest       | 87%              | 92%                    | 89%                 |
| XGBoost             | **90%**          | **94%**                | **91%**             |

Installation:
# Clone the repository
git clone https://github.com/daemon-10k/CulinAIre.git
cd CulinAIre

# Install dependencies
pip install -r requirements.txt

Usage:
1. Place your dataset (.csv) in the project folder.
2. Run the script: python classify_recipes.py
3. View the predictions for cuisine, dietary tags, difficulty, and time.

Example:

- Input: Ingredients: "basil, mozzarella cheese, tomato sauce, olive oil, pasta"
Instructions: "Boil pasta. Heat sauce. Combine with cheese and basil."

- Output: {
  "Cuisine": "Italian",
  "Vegetarian": true,
  "Spicy": false,
  "Gluten-Free": false,
  "High-Protein": false,
  "Dairy-Free": false,
  "Difficulty": "Easy",
  "Estimated Time (mins)": 20
}

Applications:
- Recipe recommendation systems
- Diet & nutrition tracking
- Personalized cooking assistants
- E-commerce recipe filtering

🤝 Contributing
Pull requests and suggestions are welcome! Please open an issue to discuss new ideas.

Contact
Name: Seraphim Ruben Udjung
LinkedIn: https://www.linkedin.com/in/seraphim-ruben-udjung-483b8a246/
