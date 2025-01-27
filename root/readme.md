# Quarks & Gluons
Pythia is a program which specializes in generation of high energy collision effects that happen in Physics. In the current model we are going to focus on one of the type of collisions in which quarks and gluons are produced. The model aims to identify type of particles based on 4 different features which are namely -
1. Transverse Momentum of particle
2. Rapidity of the particle
3. Azimuthal Angle
4. PGD-id

for more details on these specific features I request you to refer to this link - https://doi.org/10.5281/zenodo.2658763

Source of dataset - https://zenodo.org/records/3164691/files/QG_jets.npz?download=1

## Files involved
1. main.ipynb - the main notebook through which the entire model runs
2. plot.py - python script mainly written using AI with minor tweaks for visualisation of certain aspects of the model
3. weights.txt - the final weights acquired after running the model [ **Please see testing section below before testing the weights and bias** ]
4. bias.txt - the final bias acquired after running the model [ **Please see testing section below before testing the weights and bias** ]
5. files in data folder - The dataset of the model (source already provided **see above**)

## Approach
 
1. **Organising Data**
2. **Z Score Normalisation**
3. **Logistic Regression using Scikit-Learn**


### Organising Data
   1. *Number of jets* - There are exactly data for 100K jets.
   2. *Max Multiplicity* - It signifies that all entries of jets with less than max multiplicity are padded with fictious entries with zero values to ensure that errors regarding dimensions aren't encountered when the model starts to train/test on them.
   3. *features* - 
   
   
   The idea is to **flatten these layers** to ensure that logistic regression works on the matrix without any errors since maximum dimensions allowed are 2.

### Z Score Normalisation

The unnormalised data varies between a greater range which might hamper effective learning of the model since convergence is reached in a much more stable fashion when data is normalised. 
The Normalisation technique used for this model is known as Z-Score Normalisation.

The normalisation technique works using the following: $$ X_{\text norm} = \frac{X - \mu}{\sigma} $$

where $$ \mu = \frac{\sum_{i=1}^n x_i}{n}, \quad \sigma = \sqrt{\frac{\sum_{i=1}^n (x_i - \mu)^2}{n}} $$

### Logistic Regression
The logistic regression is one of the best algorithms when it comes to working on classification problems especially if it is just a binary classification. 
Just like linear regression it works on minimizing the cost function (**logarithmic** instead of squared error one since square error cost function will **have saddle points**), in a logistic regression the function used is known as sigmoid function: The sigmoid function is given by:

$$
f(x) = \frac{1}{1 + e^{-x}}
$$


Where:
- f(x) is the output of the sigmoid function.
- x is the input to the function.
- e is the base of the natural logarithm.

of course this function is just the mathematical aspect of the sigmoid function which uses input features as vectors and weights as vectors too. 
The range of sigmoid is between **(0,1)** making it ideal for binary classification

The logistic regression cost function (also known as the log-loss or binary cross-entropy loss) is given by:

#### *Logistic Regression Cost Function*

The logistic regression cost function (also known as the log-loss or binary cross-entropy loss) is given by:

$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(f_w(x^{(i)})) + (1 - y^{(i)}) \log(1 - f_W(x^{(i)})) \right]
$$
Where:
- **m** is the number of training examples.
- **y^(i)** is the true label for the i-th example.
- **y_w(x^(i))** is the predicted probability for the i-th example, computed using the sigmoid function stated above

- **θ** is the vector of model parameters (weights).
- **x^(i)** is the feature vector for the i-th example.

### *Scikit-Learn*
   The Scikit learn model outputs Accuracy of the model, Confusion Matrix and Classification which includes F1 Score:
    1. Accuracy - 
        $$Accuracy = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}} $$
    2. Confusion matrix -  It basically gives a sense of how much false positives/negatives model is giving for the testing data.                
                                            $$ Confusion\;matrix = 
                        \begin{bmatrix}
                        \text{TP} & \text{FN} \\
                        \text{FP} & \text{TN}
                        \end{bmatrix}
                        $$ 
                        - TP - True Positive
                        - FN - False Negative
                        - FP - False Positive
                        - TN - True Negative
    3. F1 Score - 
            $$
            F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} $$
            where:
            - **Precision** = $\frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}$
            - **Recall** = $\frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}$
    4. Support - It ensures that every class in the data is as impactful as others.
        $$
        \text{Support} = \text{Number of Actual Occurrences of the Class}
        $$
        For each class $i$:
        $$
        \text{Support}_i = \sum_{j=1}^{N} \mathbb{I}(y_j = i)
        $$
        where:
        - $N$ is the total number of samples.
        - $y_j$ is the true label of the $j$-th sample.
        - $\mathbb{I}(y_j = i)$ is an indicator function that equals 1 if $y_j$ is equal to class $i$, and 0 otherwise.

### Testing
The final weights and bias that the logistic regression calculated are stored in the files bias.txt [link - ] and weights.txt (contains 556 corresponding weights for 556 flattened features, **NOTE - Taking a look at the code is recommended before using the weights and bias**) [link - ].

### Possible Improvements
The code can be imporved via hyperparameter tuning by using Bayesian Search or Grid Searches. The biggest scope of improvement lies if new features are engineered from the existing features that can lead the model to better predictions.

### Author
$$ *****NovaPrime2077***** $$


