# Drug Overdose Predictor

This project trains a **Random Forest Regressor** to predict global **drug-related deaths** using real-world data.  
Basically, I fed it a messy CSV, cleaned it up and made the model guess which countries are most likely to get wrecked by rising overdose rates

---

## What It Does

- Loads global drug death data from a CSV file
- Cleans up the nightmare formatting 
- Turns country names into 0/1 columns so the model doesn’t think “Brazil > Canada” for some reason
- Applies **log scaling** to make the data less chaotic  
- Trains a **Random Forest** model to predict overdose deaths  
- Predicts future years (2023–2025)  
- Visualizes:
  - Actual vs Predicted deaths with trend lines  
  - Top features that actually influence the model's predictions  

---
