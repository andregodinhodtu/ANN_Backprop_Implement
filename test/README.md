# Testing

# Overview

Both implementations of the Artificial Neural Network — the base-Python version and the NumPy version — share the same underlying structure, methods, and algorithms.

# Key Difference in Testing

The NumPy version takes advantage of vectorization: methods that handle only a single sample in base Python can process an entire batch in NumPy. Most methods behave identically across both versions — the NumPy version simply extends them to also accept batches. Testing is built around this principle: the NumPy implementation should be able to do everything the base-Python one does (and more), but not the other way around.
