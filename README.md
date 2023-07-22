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

![image](https://github.com/TITHI-KHAN/KNN-Algorithm-for-Pattern-Recognition/assets/65033964/65b1945b-9722-431e-9198-73d92f106737)

We can now calculate some standard norms for X, starting with the L¹ norm.


**COMMON VECTOR NORMS IN MACHINE LEARNING:**

◉ L¹ / Manhattan Norm

◉ L² / Euclidian Norm

◉ L∞ Norm

◉ Lᵖ Norm


# What Is the L¹ Norm / Manhattan Norm?

The L¹ norm is defined as the sum of the absolute values of the components of a given vector. Since we have a vector X with only two components, the L¹ norm of x can be written as:

![image](https://github.com/TITHI-KHAN/KNN-Algorithm-for-Pattern-Recognition/assets/65033964/c636a944-9fe6-4f2b-8783-dcbe85b4c9cd)

![image](https://github.com/TITHI-KHAN/KNN-Algorithm-for-Pattern-Recognition/assets/65033964/36cba5c6-293a-4232-8916-8f0eff20e142)

Notice the representation with one written as a subscript. This norm is also called the Manhattan or the taxicab norm, inspired by the burrough of Manhattan in New York. The L¹ norm is typically the distance a taxi will have to drive from the origin to point x.

**MATHEMATICAL NOTATION:**

The L¹ norm can be mathematically written as:

![image](https://github.com/TITHI-KHAN/KNN-Algorithm-for-Pattern-Recognition/assets/65033964/ee478227-9b49-4f55-b517-dee279fa7040)

![image](https://github.com/TITHI-KHAN/KNN-Algorithm-for-Pattern-Recognition/assets/65033964/0afeeea9-a2ce-4fd1-8612-b7afffa6155a)

**WHAT ARE THE PROPERTIES OF THE L1/MANHATTAN NORM?**

◉ The L¹ norm is used in situations when it is helpful to distinguish between zero and non-zero values.

◉ The L¹ norm increases linearly around the origin.

◉ It is used in Lasso (Least Absolute Shrinkage and Selection Operator) regression, which involves adding the L¹ norm of the coefficient as a penalty term to the loss function.

# What Is the L² Norm / Euclidean Norm?

L² is the most commonly used norm and the one most encountered in real life. The L² norm measures the shortest distance from the origin. It is defined as the root of the sum of the squares of the components of the vector. So, for our given vector X, the L² norm would be:

![image](https://github.com/TITHI-KHAN/KNN-Algorithm-for-Pattern-Recognition/assets/65033964/c0e7cdd1-8acf-4d43-973b-ac443a03aedb)

![image](https://github.com/TITHI-KHAN/KNN-Algorithm-for-Pattern-Recognition/assets/65033964/21394f17-8b4c-4a27-8032-f081b002a17a)

The L² norm is so common that it is sometimes also denoted without any subscript:

![image](https://github.com/TITHI-KHAN/KNN-Algorithm-for-Pattern-Recognition/assets/65033964/8ee24e7b-0668-4965-8e4d-622623519589)

The L² norm is also known as the Euclidean norm after the famous Greek mathematician, often referred to as the founder of geometry. The Euclidean norm essentially means we are referring to the Euclidean distance.

**MATHEMATICAL NOTATION:**

The L² norm can be mathematically written as:

![image](https://github.com/TITHI-KHAN/KNN-Algorithm-for-Pattern-Recognition/assets/65033964/ad7f03dc-9ffb-4c4d-9189-2a382177eeeb)

![image](https://github.com/TITHI-KHAN/KNN-Algorithm-for-Pattern-Recognition/assets/65033964/1aa51524-925d-4845-9312-22255dd697ad)

**WHAT ARE THE PROPERTIES OF THE L2 NORM?**

◉  The L² norm is the most commonly used one in machine learning

◉  Since it entails squaring of each component of the vector, it is not robust to outliers.

◉  The L² norm increases slowly near the origin, e.g., 0.¹² = 0.01

◉  It is used in ridge regression, which involves adding the coefficient of the L² norm as a penalty term to the loss function.

# What Is the L∞/Max Norm?

The L∞ norm is defined as the absolute value of the largest component of the vector. Therefore, it is also called the max norm. So, continuing with our example of a 2D vector X having two components, i.e., x₁ and x₂, where x₂ > x₁, the ∞ norm would simply be the absolute value of x₂.

![image](https://github.com/TITHI-KHAN/KNN-Algorithm-for-Pattern-Recognition/assets/65033964/3fdf7feb-73b3-464e-83cb-c3b405f9dbc9)

**MATHEMATICAL NOTATION:**

The L∞ norm can be mathematically written as:

![image](https://github.com/TITHI-KHAN/KNN-Algorithm-for-Pattern-Recognition/assets/65033964/1a9b761c-7642-41e7-979c-b6ce5897076e)

**WHAT ARE THE PROPERTIES OF THE MAX NORM?**

◉ The L∞ norm simplifies to the absolute value of the largest element in the vector.

# What Is the Lᵖ Norm?

We can now generalize to the idea of what is known as the p-norm. In a way, we can derive all other norms from the p-norm by varying the values of p. That is to say, if you substitute the value of p with one, two, and ∞ respectively in the formula below, you’ll obtain L¹, L², and L∞ norms.

**MATHEMATICAL NOTATION:**

The Lᵖ norm can be mathematically written as:

![image](https://github.com/TITHI-KHAN/KNN-Algorithm-for-Pattern-Recognition/assets/65033964/80521fe2-d4bd-42d5-aa74-1140d784550f)

An important point to remember here is that each of the norms above fulfills the properties of the norms mentioned in the beginning.

**What happens when p equals zero?**

When this happens, we might want to call the value L⁰ “norm.” It is not technically a norm, however, because it violates the homogeneous property of the norms as this excerpt Wikipedia makes clear. To put things in a clearer perspective, we can say that the L⁰ “norm” is useful when we want to know the number of non-zero components in a vector. This means sparsity can be modeled via the L⁰ “norm.”

![image](https://github.com/TITHI-KHAN/KNN-Algorithm-for-Pattern-Recognition/assets/65033964/190e1188-8c8f-4aba-809e-abb2e8e6c55d)

![image](https://github.com/TITHI-KHAN/KNN-Algorithm-for-Pattern-Recognition/assets/65033964/2905bf4d-0767-4b6b-8115-0a4ef8f77098)

Sparsity is an important concept in machine learning, as it helps to improve robustness and prevent overfitting. 


