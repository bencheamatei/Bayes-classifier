# Bayesian Classifiers for Diabetes Prediction

This repository contains a from-scratch Python implementation of two Bayesian classifiers: a **Naive Bayes** and a **Non-Naive Bayes** (which models dependencies between features). The models are designed to classify individuals as diabetic or non-diabetic based on 21 health indicators.

The dataset used is the [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) (BRFSS 2015, 50-50 split) from Kaggle.

## Methodology

### 1. Naive Bayes Classifier

Built on the assumption of conditional independence between features given the class. It calculates the prior probabilities and conditional probabilities using frequency counts, leveraging logarithmic sums during prediction to prevent mathematical underflow.

### 2. Non-Naive Bayes Classifier

In medical data, features are rarely strictly independent (e.g., High Blood Pressure and High Cholesterol are correlated). The Non-Naive model relaxes this assumption:

* **Dependency Selection**: Each feature is allowed to depend on up to `mx_par` (maximum parents) previous features. Parents are selected dynamically by calculating the **Pearson correlation coefficient** ($\vert{}\rho\vert{} \ge 0.01$) to ensure the most relevant relationships are captured while keeping the dependency graph acyclic.

* **Bitmask Optimization**: To avoid the massive computational overhead of looping through $2^k$ parent configurations, this implementation uses a highly optimized bitmasking approach. Parent states are compressed into a single integer, allowing frequencies to be counted in a single vectorized pass using `numpy.bincount`.

### 3. Laplace Smoothing

Both models implement Laplace (additive) smoothing to prevent zero-probability errors when encountering unobserved feature combinations in the test set.

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/bencheamatei/Bayes-classifier.git
   cd Bayes-classifier
   
   
   ```

2. Install the required dependencies:

   ```
   pip install -r requirements.txt
   
   
   ```

3. **Data Setup**: The required dataset is already included in the repository under the `data/` directory:

   * `data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv`

## Usage

### Running the Main Classifiers (`main.py`)

You can run the classifiers to see their training and testing accuracies. The script accepts optional command-line arguments:

```
python main.py [seed] [mx_par] [alpha]


```

* `seed`: Random seed for data splitting (default: 12513).

* `mx_par`: Maximum number of dependencies/parents for the Non-Naive model (default: 1, max: 21).

* `alpha`: Laplace smoothing parameter (default: 1.0).

**Example:**

```
python main.py 42 4 1.0


```

### Running the Experiments (`exp.py`)

The repository includes an experiment script to generate the data used in the project report.

**1. Dependency Evolution (Overfitting Analysis):**
Runs the Non-Naive model for every possible number of parents (0 to 21) to show the bias-variance tradeoff and overfitting.

```
python exp.py 1


```

**2. Model Comparison:**
Compares the custom Bayes classifiers (with binarized data) against industry-standard algorithms like **Random Forest** and **XGBoost** (using raw continuous data).

```
python exp.py 2


```

## Results & Conclusions

Based on 50 randomized train/test splits (80/20), the experimental results demonstrated:

* **Optimal Dependencies**: Allowing just 3 or 4 dependencies (`mx_par=4`) yields the highest test accuracy (\~74.06%).
                
* **Overfitting**: Increasing the allowed dependencies beyond 4 causes severe overfitting. Training accuracy approaches 85%, while test accuracy drops below the baseline Naive model. This highlights the curse of dimensionality, as the number of possible parent configurations grows exponentially ($2^k$).

## Project Structure

* `src/prep.py`: Data loading, training/testing splitting, and median-based binarization logic.

* `src/naive_bayes.py`: Standard Naive Bayes implementation.

* `src/non_naive_bayes.py`: Non-Naive Bayes implementation with bitmask optimization.

* `main.py`: Entry point for standard evaluation.

* `exp.py`: Benchmarking and comparison script.

* `raport-clasificatori-bayes.pdf`: The official Romanian project report 