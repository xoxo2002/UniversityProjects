# Enhancing Quality Control in Integrated Circuit Manufacturing using AI-based Vision Inspection System

## Overview
This is a **Final Year Project(FYP)** for university. This project aims to enhance the quality control process in Integrated Circuit (IC) manufacturing by using an AI-based Vision Inspection System (AVIS).

## Table of Contents
1. [Introduction](#introduction)
2. [Methodology](#methodology)
3. [Results and Discussion](#results-and-discussion)

---

## 1.0 Introduction
As IC chips are produced in large volumes, manual inspection is inefficient. This project introduces an AI-based system for defect detection, improving accuracy in identifying defective chips while reducing overkill rates. Current rule-based inspection systems lead to increased false positives, misidentifying illumination issues and dust as defects. A more intelligent system is required to enhance efficiency and reduce waste.

## 1.2 Achieved Outcome
The proposed system successfully developed an AI-based approach combining traditional image processing for segmentation and AI-based models for defect detection. This hybrid system achieved an accuracy of 93.72%, significantly reducing overkills. For a demonstration of the system:
Surface defect detection module: [surface-demo](https://youtu.be/p9xHZAjoMDI)
Markings defect detection module: [markings-demo](https://youtu.be/PVxrwRfWdhI)
Combined model: [combined-demo](https://youtu.be/mnirX32ud20)

---

## 2.0 Research Aims, Objectives and Scope
- **Aims**: To create an AI-based system to enhance IC chip quality control.
- **Objectives**:
  1. Segment IC chip images accurately.
  2. Detect IC marking defects with at least 85% accuracy.
  3. Detect surface defects (e.g., cracks) with at least 85% accuracy.
  4. Achieve a processing speed of less than 150 ms per chip.
  
---

## 3.0 Methodology
### 3.1 Tools and Technologies Used:
- **Python**: Primary programming language.
- **TensorFlow & Keras**: For building deep learning models.
- **Pytesseract**: Optical Character Recognition (OCR) for marking defect detection.
- **OpenCV & Pillow**: Image preprocessing and defect extraction.

### 3.2 Data Set
A dataset of 1710 images, consisting of good, overkill, defected, and rejected IC chips.

### 3.3 Modules Developed:
1. **Marking Defect Module**:
   - **Character Recognition**: Tesseract trained with the OCR-A font for accurate IC markings recognition.
2. **Surface Defect Module**:
   - **CNN Model**: Identifies defects such as scratches, dots, and illegible markings.

---
