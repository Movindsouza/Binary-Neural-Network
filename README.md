# Binary-Neural-Network
DogVsCat image classification using BNN with the help of larq using python.

Larq is an open-source Python library for training neural networks with extremely low-precision weights and activations, such as Binarized Neural Networks (BNNs)1.

Existing deep neural networks use 32 bits, 16 bits or 8 bits to encode each weight and activation, making them large, slow and power-hungry. This prohibits many applications in resource-constrained environments. Larq is the first step towards solving this. The API of Larq is built on top of tf.keras and is designed to provide an easy to use, composable way to design and train BNNs (1 bit) and other types of Quantized Neural Networks (QNNs). It provides tools specifically designed to aid in BNN development, such as specialized optimizers, training metrics, and profiling tools. It is aimed at both researchers in the field of efficient deep learning and practitioners who want to explore BNNs for their applications. 

Source: https://larq.dev/
