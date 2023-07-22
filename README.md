# KNN Algorithm for Pattern Recognition

Step 01:
Regressor:
1. Import data set
2. Separate x(Gender, Height) and y (y=Weight)
3. Train = 70%, Test = 30%
4. Apply Linear Regression
5. Evaluate the Model (Testing and training Accuracy, MSE for testing)
6. Apply KNN Regressor: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
7. Evaluate the Model (Testing and training Accuracy, MSE for testing)
8. Compare KNN & Linear regression with the KNN model and Linear regression as well.


Step 02:
KNN Classifier:
1. Import data set
2. Separate x and (y=Gender)
3. Train = 70%, Test = 30%
4. Apply KNN Classifier 
5. Evaluate the Model by only Accuracy.
6. Apply KNN Classifier: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html


**KNeighborsClassifier:**

![image](https://github.com/TITHI-KHAN/KNN-Algorithm-for-Pattern-Recognition/assets/65033964/d49fd3c5-d6fb-4b6a-9b01-9f157629e3b4)


![image](https://github.com/TITHI-KHAN/KNN-Algorithm-for-Pattern-Recognition/assets/65033964/75617506-564b-488b-838e-7de477ae6cb2)


P=2  (L2 Norm) (Euclidean Distance)
metric = Minkowski (universal form of distance)

**Minkowski Distance:**

![image](https://github.com/TITHI-KHAN/KNN-Algorithm-for-Pattern-Recognition/assets/65033964/710b70b5-e3b0-4403-82e8-ebf0e925f916)

If P=1, then Manhattan Distance.
If P=2, then Euclidean Distance.

# Vector Norms

**What is a Norm?**

The distance between two vectors.

A norm is a way to measure the size of a vector, a matrix, or a tensor. In other words, norms are a class of functions that enable us to quantify the magnitude of a vector. For instance, the norm of a vector X drawn below is a measure of its length from origin.

![image](https://github.com/TITHI-KHAN/KNN-Algorithm-for-Pattern-Recognition/assets/65033964/a7cfdabf-3ad7-4b8d-8e68-73133c7056b2)

norm -> It can be L1, L2, or infinity by depending on the situation.

**The subject of norms comes up on many occasions in the context of machine learning:**


◉ When defining loss functions, i.e., the distance between the actual and predicted values.

◉ As a regularization method in machine learning, e.g., ridge and lasso regularization methods.

◉ Even algorithms like SVM use the concept of the norm to calculate the distance between the discriminant and each support-vector.


**How Do We Represent Norms?**

The norm of any vector X is denoted by a double bar around it and is written as follows:

![image](https://github.com/TITHI-KHAN/KNN-Algorithm-for-Pattern-Recognition/assets/65033964/ba7ad9b8-c417-4deb-a7a1-cc561669a1af)


**What Are the Properties of Norm?**

Consider two vectors X and Y, having the same size and a scalar- ?. A function is considered a norm if and only if it satisfies the following **properties**:

◉ Non-negativity: It should always be non-negative.

◉ Definiteness: It is zero if and only if the vector is zero, i.e., zero vector.

◉ Triangle inequality: The norm of a sum of two vectors is no more than the sum of their norms.

◉ Homogeneity: Multiplying a vector by a scalar multiplies the norm of the vector by the absolute value of the scalar.

Let’s see these **qualities represented mathematically**.

**NON-NEGATIVITY:**

![image](https://github.com/TITHI-KHAN/KNN-Algorithm-for-Pattern-Recognition/assets/65033964/79d9c9d4-e03a-4642-921d-d55c1d7c15d3)

**DEFINITENESS:**

![image](https://github.com/TITHI-KHAN/KNN-Algorithm-for-Pattern-Recognition/assets/65033964/dd395ee8-bd93-4d37-831a-8b92be840c0b)

**TRIANGE INEQUALITY:**

![image](https://github.com/TITHI-KHAN/KNN-Algorithm-for-Pattern-Recognition/assets/65033964/52c1a94b-b90f-414d-9cd9-286ba3e5b8af)

![image](https://github.com/TITHI-KHAN/KNN-Algorithm-for-Pattern-Recognition/assets/65033964/315a6d06-0b60-4b2e-bdf7-005afc5a6354)

**HOMOGENEITY:**

![image](https://github.com/TITHI-KHAN/KNN-Algorithm-for-Pattern-Recognition/assets/65033964/0320b5f9-88d3-4364-80d1-232587ae05a6)

Any real value function of a vector that satisfies the above four properties is called a norm.

**What Are Some Standard Norms?**

A lot of functions can be defined that satisfy the properties above. Consider a two-dimensional column vector X as follows,

![image](https://github.com/TITHI-KHAN/KNN-Algorithm-for-Pattern-Recognition/assets/65033964/e8ba55f5-479b-4f4b-aab9-7f8fce43d586)

**COMMON VECTOR NORMS IN MACHINE LEARNING:**

◉ L¹ / Manhattan Norm

◉ L² / Euclidian Norm

◉ L∞ Norm

◉ Lᵖ Norm


