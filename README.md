# Robust interquantile multilayer perceptron (IQ-MLP), version 1.0

A (robust) interquantile multilayer perceptron. The idea is to compute two selected nonlinear 
regression quantiles by means of multilayer perceptrons (MLPs) and the final regression fit is 
obtained as a standard MLP computed however only for such observations, which are between the 
two quantiles; to achieve robustness, the remaining observations are ignored completely.

Feel free to use or modify the code.

## Requirements

You need to install Python, its library NumPy, its math module, TensorFlow, and Keras  (which itself is an open-source library written in Python).

## Usage

* The usage of the code is straightforward. The training of the robust MLP is called in the same way as habitually
used calling of a standard (non-robust) MLP.

## Authors
  * Jan Tichavský, The Czech Academy of Sciences, Institute of Computer Science
  * Jan Kalina, The Czech Academy of Sciences, Institute of Computer Science

## Contact

Do not hesitate to contact us (tichavsk@seznam.cz) or write an Issue.

## How to cite

When refering to the IQ-MLP  method, please consider citing the following:

Kalina J, Vidnerová P (2020): On robust training of regression neural networks. In Aneiros G, Horová I, 
Hušková M, Vieu P (eds): Functional and High-Dimensional Statistics and Related Fields. IWFOS 2020, 
Contributions to Statistics. Springer, Cham, pp. 145-152.

## Acknowledgement

This work was supported by the Czech Science Foundation grant GA19-05704S.