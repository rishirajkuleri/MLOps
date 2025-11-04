# Data Drift Detection with Evidently AI

## Overview
This project demonstrates data drift detection using the Evidently AI library on the Audiology dataset from OpenML. The analysis compares reference and production datasets to identify potential distribution shifts in categorical features.

## Libraries Used

### Core Libraries
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning utilities and dataset loading

### Evidently AI Components
- **evidently** (v0.7.0) - ML monitoring and data drift detection
  - `Dataset` - Data wrapper for Evidently analysis
  - `DataDefinition` - Schema definition for categorical and numerical features
  - `Report` - Report generation framework
  - `DataDriftPreset` - Pre-configured drift detection metrics
  - `DataSummaryPreset` - Dataset overview and statistical summaries
  - `CloudWorkspace` - Cloud workspace integration for report storage

### Data Access
- **openml** - Access to OpenML dataset repository

## Dataset

**Dataset Name:** Audiology  
**Source:** OpenML (Dataset ID: 7)  
**Format:** Categorical features dataset  

### Dataset Split
- **Reference Dataset:** First 100 rows (audiology.iloc[:100])
- **Production Dataset:** Remaining rows (audiology.iloc[100:])

### Data Characteristics
- **Total Columns:** 69
- **Column Types:** All categorical features (excluding 'bser' target variable)
- **Target Variable:** 'bser' (excluded from drift analysis)

## Methodology

### Data Schema Definition
All columns except 'bser' are defined as categorical features with no numerical columns specified in the DataDefinition.

### Drift Analysis
The project uses two primary Evidently presets:

1. **DataDriftPreset**
   - Detects distribution changes between reference and production data
   - Uses a drift detection threshold of 0.5
   - Analyzes all 69 categorical columns

2. **DataSummaryPreset**
   - Provides statistical overview of both datasets
   - Shows distribution summaries for each feature

## Results

### Drift Detection Summary
- **Dataset Drift Status:** NOT detected
- **Drift Threshold:** 0.5
- **Drifted Columns:** 14 out of 69
- **Drift Percentage:** 20.29%

The analysis shows that while individual feature drift was detected in approximately 20% of columns, the overall dataset drift threshold was not exceeded, indicating the production data remains within acceptable distribution bounds compared to the reference dataset.

## Environment
- **Platform:** Google Colab
- **Python Version:** 3.12

## Key Features
- Automated drift detection for categorical features
- Visual distribution comparisons
- Statistical summaries of reference vs. production data
- Cloud workspace integration for report persistence
- Comprehensive drift metrics and thresholds

## Conclusion
This project successfully demonstrates the application of Evidently AI for monitoring data quality and detecting distribution shifts in production ML systems. The modular approach using presets makes it easy to implement robust data drift monitoring pipelines.
