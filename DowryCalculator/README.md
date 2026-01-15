# Dowry Amount Prediction (Awareness-Only Project)

> ⚠️ **This project is for social awareness and experimentation only.**  
> Dowry is illegal in India under the Dowry Prohibition Act, 1961.  
> This model simulates societal bias and must not be used to justify or promote dowry.

## Overview
An end-to-end machine learning pipeline that predicts a simulated *dowry amount* (in lakhs INR) based on synthetic, biased features. It mirrors the diamond project structure: PyTorch Lightning for training, ONNX export for inference, and a FastAPI web UI.

---
**Use responsibly. Dowry is illegal and unethical.**




# Dummy Dowry Calculation Dataset (Awareness & Experimental Use Only)

## 📌 Overview

This repository contains a **synthetic (dummy) dataset** created for **machine learning experimentation, data analysis, and social awareness purposes only**.

The dataset simulates how **societal bias** (not logic, law, or ethics) often evaluates a groom’s “value” in the context of Indian marriages using superficial attributes such as salary, profession, age, and location.

⚠️ **Important:**  
This dataset does **NOT** promote, support, or justify the dowry system in any form.  
Dowry is illegal in India under the **Dowry Prohibition Act, 1961**.

---

## 🎯 Purpose of the Dataset

- To experiment with **regression and explainable AI models**
- To visualize **societal bias using data**
- To promote **awareness and self-reflection**
- To show how humans are unfairly reduced to numbers

If the predictions or patterns feel uncomfortable — that discomfort is intentional and meaningful.

---

## 🧠 Problem Statement

**Objective:**  
Predict a *society-perceived dowry amount* based on attributes often (wrongly) considered important in marriage negotiations.

This is **not a real valuation** of a human being.

---

## 📊 Dataset Structure

### Input Features

| Feature | Type | Description |
|------|------|------------|
| `age` | Integer | Age of groom (22–40 years) |
| `monthly_salary` | Integer | Monthly income in INR |
| `profession` | Categorical | Job category |
| `education_level` | Categorical | Highest education attained |
| `location` | Categorical | City tier |
| `home_status` | Categorical | Own or rented house |
| `family_wealth` | Categorical | Family economic background |
| `marital_status` | Categorical | Single / Married / Divorced |
| `marriage_type` | Categorical | Love or Arranged marriage |
| `government_job` | Binary | Government job indicator (0 or 1) |

### Target Variable

| Feature | Type | Description |
|------|------|------------|
| `dowry_amount_lakhs` | Float | Simulated dowry value in lakhs INR |

---

## 🧮 How the Target Is Generated

The `dowry_amount_lakhs` value is generated using **rule-based societal assumptions**, such as:

- Higher salary → higher perceived value
- Government jobs receive a significant premium
- Urban locations are valued more than rural
- Love marriages and remarriages receive penalties
- Education and property ownership increase value

These rules are **deliberately biased** to reflect social realities — not correctness.

---

## 🧪 Sample Data

```csv
age,monthly_salary,profession,education_level,location,home_status,family_wealth,marital_status,marriage_type,government_job,dowry_amount_lakhs
26,80000,Software Engineer,Masters,Tier-1,Own,Upper-Middle,Single,Arranged,0,28.5
30,45000,Government Officer,Bachelors,Tier-2,Own,Middle,Single,Arranged,1,32.0
28,120000,Doctor,Masters,Tier-1,Own,Upper,Single,Love,0,38.5
34,60000,Private Job,Bachelors,Tier-3,Rented,Middle,Divorced,Arranged,0,12.5
