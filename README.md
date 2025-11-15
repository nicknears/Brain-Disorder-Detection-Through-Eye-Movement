# Wearable Sensors for Brain Disorder Detection Through Eye Movement Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![R 4.0+](https://img.shields.io/badge/R-4.0+-blue.svg)](https://www.r-project.org/)

A comprehensive research project analyzing eye movement patterns collected from wearable sensors to detect and monitor brain disorders. This repository contains analysis code and documentation for using eye movements as biomarkers for neurological conditions.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Code Documentation](#code-documentation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🎯 Overview

This repository provides computational tools and analysis scripts for processing eye movement data from wearable sensors. The codebase includes:

- Python and R implementations for data analysis
- Feature extraction algorithms
- Statistical analysis tools
- Machine learning classification models
- Interactive Jupyter notebooks

## ✨ Features

- **Comprehensive Data Analysis**: Python and R scripts for complete eye movement data processing
- **Feature Extraction**: Automated extraction of velocity, acceleration, saccade, and fixation metrics
- **Statistical Analysis**: T-tests and other statistical methods to compare groups
- **Machine Learning**: Classification models (Random Forest, SVM) for disorder detection
- **Dimensionality Reduction**: PCA for feature analysis and visualization
- **Interactive Notebooks**: Jupyter notebooks with detailed analysis and visualizations

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- R 4.0 or higher (optional, for R scripts)
- Git

### Python Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/wearable-sensors-brain-disorder-detection.git
cd wearable-sensors-brain-disorder-detection
```

2. Create a virtual environment (recommended):
```bash
# Using venv
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

### R Environment Setup (Optional)

If you plan to use the R scripts:

1. Install required R packages:
```r
install.packages(c("dplyr", "tidyr", "ggplot2", "corrplot", "caret", 
                   "randomForest", "e1071", "pROC", "psych"))
```

2. Or use the R script to install dependencies:
```bash
Rscript install_r_packages.R
```

### Jupyter Notebook Setup

To use the Jupyter notebook:

```bash
# Install Jupyter
pip install jupyter notebook

# Launch Jupyter
jupyter notebook
```

Then open `eye_movement_analysis.ipynb`

## 📖 Usage

### Python Script

Run the main Python analysis script:

```bash
python eye_movement_analysis.py
```

The script includes:
- Data loading and preprocessing
- Feature extraction
- Statistical analysis
- Machine learning classification
- Visualization generation

### R Script

Run the R analysis script:

```r
source("eye_movement_analysis.R")
```

Or from command line:

```bash
Rscript eye_movement_analysis.R
```

### Jupyter Notebook

For interactive analysis:

1. Start Jupyter:
```bash
jupyter notebook
```

2. Open `eye_movement_analysis.ipynb`

3. Run cells sequentially to perform the complete analysis

### Custom Data

To use your own data:

1. Format your data as CSV with columns:
   - `subject_id`: Unique identifier for each subject
   - `time`: Time points
   - `x_position`: X-coordinate of eye position
   - `y_position`: Y-coordinate of eye position
   - `group`: Classification label (e.g., 'healthy', 'disorder')

2. Update the file path in the scripts:
```python
data = pd.read_csv('path/to/your/data.csv')
```

## 📁 Project Structure

```
wearable-sensors-brain-disorder-detection/
│
├── README.md                          # This file
├── LICENSE                            # MIT License
├── .gitignore                         # Git ignore rules
├── .gitattributes                     # Git attributes
├── requirements.txt                   # Python dependencies
│
├── eye_movement_analysis.py           # Main Python analysis script
├── eye_movement_analysis.R            # R analysis script
├── eye_movement_analysis.ipynb        # Jupyter notebook
│
├── data/                              # Data directory (create if needed)
│   └── .gitkeep
│
├── results/                           # Analysis results (generated)
│   ├── figures/
│   └── models/
│
└── docs/                              # Additional documentation
    └── methodology.md
```

## 📖 Code Documentation

### Available Scripts

- **`eye_movement_analysis.py`**: Python module with `EyeMovementAnalyzer` class
  - Data preprocessing and filtering
  - Feature extraction (velocity, acceleration, saccades, fixations)
  - Statistical analysis (t-tests)
  - PCA dimensionality reduction
  - Machine learning classification (Random Forest, SVM)
  - Visualization functions

- **`eye_movement_analysis.R`**: R script with equivalent functionality
  - Data processing functions
  - Statistical analysis
  - Machine learning models
  - Visualization capabilities

- **`eye_movement_analysis.ipynb`**: Interactive Jupyter notebook
  - Step-by-step analysis workflow
  - Interactive visualizations
  - Code examples and explanations

### Data Format

Expected CSV format:
- `subject_id`: Unique identifier for each subject
- `time`: Time points
- `x_position`: X-coordinate of eye position
- `y_position`: Y-coordinate of eye position
- `group`: Classification label (e.g., 'healthy', 'disorder')

### Key Functions

**Python (`eye_movement_analysis.py`)**:
- `EyeMovementAnalyzer.load_data()`: Load data from CSV
- `EyeMovementAnalyzer.preprocess_data()`: Filter and clean data
- `EyeMovementAnalyzer.extract_features()`: Extract eye movement features
- `EyeMovementAnalyzer.perform_statistical_analysis()`: Run t-tests
- `EyeMovementAnalyzer.apply_pca()`: Dimensionality reduction
- `EyeMovementAnalyzer.train_classifier()`: Train ML models
- `EyeMovementAnalyzer.visualize_results()`: Generate plots

**R (`eye_movement_analysis.R`)**:
- `load_eye_movement_data()`: Load data
- `preprocess_eye_data()`: Preprocess data
- `extract_eye_features()`: Extract features
- `perform_statistical_analysis()`: Statistical tests
- `apply_pca()`: PCA analysis
- `train_classifier()`: Train models
- `create_visualizations()`: Generate plots

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style

- Python: Follow PEP 8 style guide
- R: Follow tidyverse style guide
- Include docstrings for all functions
- Add comments for complex logic

### Reporting Issues

If you encounter any issues or have suggestions, please open an issue on GitHub with:
- Description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- System information (OS, Python/R versions)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Additional Resources

- **Code Documentation**: Inline documentation in Python and R scripts
- **Jupyter Notebook**: Interactive analysis with detailed explanations

## 👥 Authors

- Research Team

## 🙏 Acknowledgments

- Researchers and clinicians working on eye movement analysis
- Open-source community for excellent tools and libraries
- Study participants who contributed data

## 📧 Contact

For questions, suggestions, or collaborations, please:

- Open an issue on GitHub
- Contact the research team

## 🔮 Future Work

- [ ] Integration with other physiological signals (multi-modal approach)
- [ ] Real-time analysis capabilities
- [ ] Longitudinal studies for disease progression tracking
- [ ] Validation in diverse populations
- [ ] Development of mobile applications
- [ ] Cloud-based data processing infrastructure
- [ ] Personalized baseline establishment

## 📝 Changelog

### Version 1.0.0 (2024)
- Initial release
- Python and R analysis scripts
- Jupyter notebook implementation
- Comprehensive documentation

---

**Note**: This project is for research and educational purposes. The analysis code uses synthetic data for demonstration. For clinical applications, proper validation with real-world data and regulatory approval are required.

