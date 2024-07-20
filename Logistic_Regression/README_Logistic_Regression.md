This directory contains my implementation of logistic regression. You can view the model in logistic_regression_model.py. I applied my model to 4 toy datasets (generated using a fake model with some added noise). Lastly, I used it to predict whether a patient has diabetes based on several key features. 

This project started with me goofing around in a colab notebook, and turned out to be a very interesting lesson in how to achieve numerical stability when calculating sigmoid, cross entropy loss, and their derivatives. Who would've thought I'd be back to doing calculus by hand?

I used 4 algorithms as solvers: batch gradient descent, mini-batch gradient descent, stochastic gradient descent, and stochastic average gradient. 

My implementation achieves nearly identical loss and accuracy but is pretty slow compared to scikit-learn (about 200x), but hey, a model is a model :\)
