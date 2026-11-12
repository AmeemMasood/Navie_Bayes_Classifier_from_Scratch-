# Navie_Bayes_Classifier_from_Scratch-

Gaussian Naive Bayes from Scratch

A simple implementation of the Gaussian Naive Bayes classifier using only NumPy, with no machine learning libraries. Built to understand how the algorithm works under the hood.

Features
Pure NumPy implementation
Works with continuous features and any number of classes
Uses log-probabilities to avoid numerical underflow
Small variance smoothing (1e-9) to avoid division by zero
Tested on a toy dataset and matches sklearn.naive_bayes.GaussianNB
How Naive Bayes works

For a sample x, we want the class c with the highest posterior probability. By Bayes' theorem:

P(c | x) ∝ P(c) · P(x | c)
P(c) is the prior: the fraction of training samples in class c.
P(x | c) is the likelihood: how typical x is for class c.

The "naive" assumption is that features are independent given the class:

P(x | c) = P(x₁ | c) · P(x₂ | c) · … · P(xₙ | c)

The "Gaussian" assumption is that each P(xᵢ | c) follows a normal distribution:

f(x) = exp( -(x - μ)² / (2σ²) ) / √(2πσ²)

So for every (class, feature) pair, we only need to learn a mean (μ) and a variance (σ²).

Working in log space

Multiplying many small probabilities underflows to zero, so we use logs, which turn products into sums:

log P(c | x) ∝ log P(c) + Σ log P(xᵢ | c)

The prediction is the class with the highest score. We skip dividing by P(x) because it is the same for every class and doesn't change the argmax.

Implementation walkthrough
fit(X, y)

Learns three things from the training data:

| Attribute | Shape | Meaning |
|-----------|-------|---------|
| `_mean` | `(n_classes, n_features)` | Mean of each feature for each class |
| `_var` | `(n_classes, n_features)` | Variance of each feature for each class |
| `_priors` | `(n_classes,)` | Fraction of samples in each class |



_predict(x)

For each class, adds the log prior to the sum of the per-feature log-likelihoods, then returns the class with the highest total.

predict(X)

Applies _predict to every row of X.
