# Quarks & Gluons
Pythia is a program which specializes in the generation of high energy collision effects that happen in Physics. In the current model, we are going to focus on one of the types of collisions in which quarks and gluons are produced. The model aims to identify the type of particles based on four different features, which are:
1. Transverse Momentum of particle
2. Rapidity of the particle
3. Azimuthal Angle
4. PGD-id

For more details on these specific features, I request you to refer to this link - [DOI link](https://doi.org/10.5281/zenodo.2658763)

## Files involved
1. **main.ipynb** - The main notebook through which the entire model runs.
2. **plot.py** - Python script mainly written using AI with minor tweaks for visualization of certain aspects of the model.
3. **weights.txt** - The final weights acquired after running the model [**Please see the testing section below before testing the weights and bias**].
4. **bias.txt** - The final bias acquired after running the model [**Please see the testing section below before testing the weights and bias**].
5. **Files in the data folder** - The dataset of the model (source already provided **see above**, **extract the .npz by changing it to .zip and then using the 2 binary inside it**).

## Approach
1. **Organizing Data**
2. **Z Score Normalization**
3. **Logistic Regression using Scikit-Learn**

### Organizing Data
1. **Number of jets** - There are exactly data for 100K jets.
2. **Max Multiplicity** - It signifies that all entries of jets with less than max multiplicity are padded with fictitious entries with zero values to ensure that errors regarding dimensions aren't encountered when the model starts to train/test on them.
3. **Features** - The idea is to **flatten these layers** to ensure that logistic regression works on the matrix without any errors since the maximum dimensions allowed are 2.

### Z Score Normalization

The unnormalized data varies between a greater range, which might hamper effective learning of the model since convergence is reached in a much more stable fashion when data is normalized. The normalization technique used for this model is known as **Z-Score Normalization**.

The normalization technique works using the following formula:
$$X_{\text{norm}} = \frac{X - \mu}{\sigma}$$

Where:
$$\mu = \frac{\sum_{i=1}^n x_i}{n}, \quad \sigma = \sqrt{\frac{\sum_{i=1}^n (x_i - \mu)^2}{n}}$$

### Logistic Regression

Logistic regression is one of the best algorithms when it comes to working on classification problems, especially if it is just a binary classification. The function used in logistic regression is the **sigmoid function**:

$$f(x) = \frac{1}{1 + e^{-x}}$$

Where:
- \( f(x) \) is the output of the sigmoid function.
- \( x \) is the input to the function.
- \( e \) is the base of the natural logarithm.

The logistic regression cost function (also known as the **log-loss** or **binary cross-entropy loss**) is given by:

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(f_w(x^{(i)})) + (1 - y^{(i)}) \log(1 - f_W(x^{(i)})) \right]$$

Where:
- **m** is the number of training examples.
- **y^(i)** is the true label for the i-th example.
- **f_w(x^(i))** is the predicted probability for the i-th example, computed using the sigmoid function.

### Scikit-Learn

Scikit-learn is used for evaluating the model's performance. The model outputs accuracy, confusion matrix, and classification metrics such as F1 Score:
1. **Accuracy**: 
$$Accuracy = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}$$
2. **Confusion Matrix**:

$$\text{Confusion Matrix} =\begin{bmatrix}\text{TP} & \text{FN} \\\ \text{FP} & \text{TN}\end{bmatrix}$$
- **TP** - True Positive
- **FN** - False Negative
- **FP** - False Positive
- **TN** - True Negative

3. **F1 Score**:
$$F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
Where:
- **Precision** = $$\frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}$$
- **Recall** = $$\frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}$$

### Testing
The final weights and bias are stored in **bias.txt** and **weights.txt** (the weights file contains 556 corresponding weights for 556 flattened features). Please refer to the code before using the weights and bias for testing.

### Possible Improvements
Hyperparameter tuning using techniques like **Bayesian Search** or **Grid Searches** could improve the model. Feature engineering is another area with potential for enhancing prediction accuracy.

### Author
**NovaPrime2077**
