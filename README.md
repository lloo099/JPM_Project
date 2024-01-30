# JPM Project

This implementation pertains to the JPM Project, focusing on Problem 2. In this paper, we introduce an adaptation of the classical Adam method tailored to operate in place, capitalizing on the adaptive and scalable nature of deep learning, particularly multi-layer perceptrons (MLPs). By leveraging the robust capabilities of neural networks to learn intricate patterns from data autonomously, we enhance the model's ability to discern complex relationships within the input space.

With a single 3090 GPU, our implementation efficiently trains a small MLP model. The iterative training process, facilitated by frameworks like TensorFlow, forms the cornerstone for uncovering intricate relationships within the input space. Addressing the optimization challenge posed by a four-variable function ƒ (x, y, u, v) under nuanced constraints, we introduce a novel application of convolutional neural networks (CNNs). Traditionally recognized for spatial hierarchy extraction in visual domains, CNNs exhibit versatility beyond image-related tasks. Here, we harness their hierarchical feature extraction capabilities to unveil latent patterns within the input space, optimizing ƒ while adhering to the specified constraints.

## Installation

Please install the latest versions of Tensorflow (`tensorflow` following [[https://tensorflow.org](https://pytorch.org](https://www.tensorflow.org/install/pip))).

## Prepare the data

We organize the datasets generated from NumPy random functions, and you can review the details in our code implementation.

See `main.py` for more options. For results in section 4.1 of our paper, we use the default option: we take several sampling numbers (100,1000,10000).

See `main_v2.py` for more options. For results in section 4.2 of our paper, we introduce noise as outlined in Part 2 of the problem and proceed to train the neural networks accordingly.

See `main_bayesian.py` for more options. For results in section 4.3 of our paper, we introduce bayesian optimization as outlined in Part 2 of the problem and proceed to train the neural networks accordingly.

## Usage

Use `main_bayesian.py` for all functions and refer to `main_v2.py` for all arguments.
```bash
python main_bayesian.py 
```
