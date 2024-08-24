# Intro
Hello! This repo will be a collection of exercises in which I'll implement different classical machine learning algorithms from scratch, and then apply these models to both toy and real-world datasets.

# Why ?
My motivation for doing this is both to deepen my understanding of ML math theory as well as strengthening my python data science skillset. This also helps me gain a better intuition for how to translate academic papers into usable code.

# Practical? Nope.
Would I ever use these models in practice? Definitely not, as my implementations will lack a lot of optimization and extra functionality I could get from scikit-learn. Nevertheless, I feel much more comfortable using such tools when I understand how they work at a deeper level.

# No NumPy someday?
Because my models only use basic matrix operations via NumPy, I would like to eventually write my own low-level replacement (with an identical API) in either C or Rust which I can substitute in, so that this will truly be "from scratch". 

# Project Structure
You can see how I've applied each model to real and toy datasets along with their respective READMEs in 'notebooks_and_readmes'. The source code for models can be seen in 'models', and their corresponding unit tests can be found in 'tests'. 

# Testing
I'll also write unit tests for all models and utility functions using pytest. They will be contained in tests/ and will mirror the structure of the project.

