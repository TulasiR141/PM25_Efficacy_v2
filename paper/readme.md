# MAI Project Module

#### Project title:Spheroid Metrics

#### Team members:

Tulasi Rajgopal
Richard Van Winkle
Sara Prasla


## Overview

This document describes the academic Paper created for the **SpheroidMetrics: Automated Segmentation and Morphological Analysis of Cancer Spheroids in Bright-Field Images** project.

## Design Tool

**Overleaf** - A web-based platform used to write the scientific paper and organize content sections.

## Paper Content Structure

### Introduction - Why is this research important?
- Pancreatic cancer is highly fatal and hard to treat
- 3D spheroids offer tissue-like disease models
- They support studies of tumor growth and drug response
- Morphological metrics reveal growth and treatment effects
- Accurate analysis is crucial for reliable results
- Manual spheroid measurement is slow and subjective
- Inefficient for large datasets or time-point comparisons
- Automation is needed for faster, consistent analysis
- Automation enables accurate measurement of morphological metrics such as area, perimeter, roundness, and brightness

#### Objectives - What are we trying to achieve?
1. **Spheroid Segmentation**: Implement and optimize a model (YOLO/U-Net) to accurately detect spheroid boundaries
2. **Feature Extraction**: Calculate key morphological metrics and raw pixel intensity from segmented regions
3. **Feature Comparison**: Perform comparative analysis of metrics to assess growth and drug-induced morphological changes

### Related work 
- Exploring what are the current studies in the related field

### Data and preprocessing work 
- Data Collection details
- Understanding the Data
- preprocessing steps


### Methodology 
Pipeline stages:

- Pipeline Preparation - YOLOv8 segmentation and Unet
- Model Training and Optimisation
- Output Generation

### Metrics and Feature Extraction  
- Metrics exploration - Morphological and Intensity Metrics


### Evaluation 
- Performance Evaluation
- Metric Calculation Validation


### Conclusion 
- Summary of the work
- Limitations
- Future work


### Project Team - Who is involved?
- **Authors**: Tulasi Rajgopal, Sara Prasla, Richard Van Winkle
- **Project Supervisor**: Prof. Dr. Magda Gregorová
- **Key Stakeholder**: Dr. Dalia Mahdy
- **Institution**: THWS

### Where to find more?
GitHub Repository: https://github.com/thws-mai/PM25_Efficacy

## Files Information


- **Format**: PDF drafts and latex code vesion
- **Design Platform**: overleaf

