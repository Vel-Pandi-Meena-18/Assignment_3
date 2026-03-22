# Assignment_3

# 🌲 EcoType: Forest Cover Type Prediction Using Machine Learning

## 📌 Overview
EcoType is a machine learning project that predicts the **forest cover type** of a geographical region using cartographic and environmental features such as elevation, slope, hillshade, soil type, and wilderness area.

This project helps in:
- Forest resource management
- Land cover mapping
- Ecological research
- Environmental monitoring

---

## 📊 Dataset Information
- **Dataset Name:** Forest Cover Type Dataset
- **Rows:** 145,890
- **Columns:** 13
- **Target Variable:** `Cover_Type`
- **Number of Classes:** 7

### Input Features
1. Elevation  
2. Aspect  
3. Slope  
4. Horizontal_Distance_To_Hydrology  
5. Vertical_Distance_To_Hydrology  
6. Horizontal_Distance_To_Roadways  
7. Hillshade_9am  
8. Hillshade_Noon  
9. Hillshade_3pm  
10. Horizontal_Distance_To_Fire_Points  
11. Wilderness_Area  
12. Soil_Type  

### Target Output
- `Cover_Type`

---

## ⚙️ Project Workflow

### 1. Data Collection
- Loaded dataset using Pandas

### 2. Data Understanding
- Checked dataset shape, columns, info, and summary statistics
- Verified missing values and duplicate records

### 3. Data Cleaning & Transformation
- Confirmed no missing values
- Confirmed no duplicate rows
- Detected outliers using boxplots

### 4. Feature Engineering
- Encoded categorical-style features for model training
- Saved preprocessing consistency files for deployment

### 5. Exploratory Data Analysis (EDA)
- Performed univariate and bivariate analysis
- Used:
  - Histograms
  - Boxplots
  - Heatmaps
  - Class distribution plots
- Analyzed feature importance using Random Forest

### 6. Class Imbalance Handling
- Checked class distribution before model building

### 7. Feature Selection
- Applied feature importance analysis using Random Forest
- Selected meaningful features for training

### 8. Model Building
Built and evaluated the following models:
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- XGBoost

### 9. Hyperparameter Tuning
- Applied **RandomizedSearchCV** on the best-performing model

### 10. Final Model Saving
- Saved trained model as `.pkl`
- Saved scaler and model feature list for deployment

### 11. Streamlit Deployment
- Built an interactive Streamlit app
- Users can manually enter feature values
- App predicts the forest cover type and shows confidence scores

---

## 🤖 Model Performance

| Model | Accuracy |
|------|----------|
| Logistic Regression | 0.8221 |
| Decision Tree | 0.9378 |
| Random Forest | **0.9561** |
| KNN | 0.9104 |
| XGBoost | 0.9366 |

### ✅ Best Model
**Random Forest** was selected as the final model because it achieved the highest accuracy and strong overall class performance.

---

## 🖥️ Streamlit App Features
- Clean and interactive user interface
- Numeric input fields for continuous features
- Dropdowns for categorical/discrete features
- Prediction output display
- Prediction confidence table
- Top predicted class with confidence percentage

---

## 📂 Project Files

- `Estmapp.py` → Streamlit application
- `forest_model.pkl` → Final trained model
- `scaler.pkl` → Saved scaler
- `model_features.pkl` → Saved feature columns
- `requirements.txt` → Required libraries
- `README.md` → Project documentation

---

## 🚀 How to Run the Project


### 1. Install dependencies
```bash
pip install -r requirements.txt


## 2. Run Streamlit app
streamlit run Estmapp.py
