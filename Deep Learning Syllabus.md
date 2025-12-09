# Deep Learning Syllabus (MINDS)

## Prerequisites
- Python: Data structures, NumPy, Pandas, Matplotlib
- Math: Basics of linear algebra (vectors, matrices), derivatives
- Machine Learning: Train–test split, metrics, overfitting, underfitting

---

# LEVEL–1: Foundations of Deep Learning & Feed-Forward Networks

## Class 1 — Introduction to Deep Learning
- Why do we need Deep Learning?
- What is a Neural Network?
- Biological inspiration (Brain → Neurons → Perceptron)
- Feed-Forward Neural Networks
- Forward Propagation
- Activation Functions
- Loss Functions
- Backward Propagation
- Gradient Descent & Optimization Basics
- Types of Neural Networks (FFNN, CNN, RNN, etc.)

**Project:** Implement a Neural Network from Scratch (NumPy)

---

## Class 2 — Optimization & PyTorch Basics
- What is Gradient Descent?
- Vanishing/Exploding Gradient Problem
- Improving accuracy:
  - Weight Initialization
  - Batch Normalization
  - Dropout & Regularization
- PyTorch Basics:
  - Tensors
  - Autograd
  - DataLoader Class
- Implementing an ANN in PyTorch
- CPU vs GPU Training
- Optimization algorithms: SGD, Adam, RMSProp

**Project:** Re-implement the previous Neural Network using PyTorch

---

## Class 3 — Convolutional Neural Networks
- CNNs and biological vision system
- Why CNNs?
- Forward Propagation in CNN
- Layers:
  - Convolution
  - Pooling
  - Flattening
  - Fully Connected Layers
- Backpropagation in CNN
- CNN Architectures:
  - AlexNet
  - GoogleNet
  - MobileNet
  - ResNet
- How to choose an architecture?
- Optimization tricks
- Data Augmentation
- Transfer Learning basics

**Project:** Implement ResNet in PyTorch for image classification

---

# LEVEL–2: Recurrent Neural Networks & Sequence Modeling

## Class 1 — Recurrent Neural Networks
- Why RNNs?
- RNN Basics
- Forward Propagation in RNN
- Backpropagation Through Time (BPTT)
- Implementing a basic RNN in PyTorch
- LSTM vs RNN
- Implementing LSTM in PyTorch
- What is GRU?

---

## Class 2 — Sequence Modeling
- Bidirectional RNN
- Encoder–Decoder / Seq2Seq Architecture
- Attention Mechanism
- Teacher Forcing
- Applications:
  - Machine Translation
  - Summarization
  - Speech Processing

**Project:** Build a Seq2Seq Machine Translation Model

---

## Class 3 — Advanced RNN Topics
- Stacked RNNs
- Handling long sequences & memory bottlenecks
- Scheduled Sampling
- Beam Search
- Applications of RNNs:
  - Time Series Forecasting
  - Text Generation
  - Speech Recognition
  - Video Analysis
- Attention Mechanisms:
  - Bahdanau Attention
  - Luong Attention
  - Self-Attention Basics

**Project:** Build an LSTM-based Time Series Forecasting Model

---

# LEVEL–3: Generative AI

## Class 1 — Foundations of Generative Models
- Generative vs Discriminative Models
- Autoencoders:
  - Vanilla Autoencoder
  - Variational Autoencoder (VAE)
- Propagation in Autoencoders
- Latent Space representation
- Autoencoders in PyTorch
- Generative Adversarial Networks (GANs):
  - Generator & Discriminator
  - Adversarial Loss
  - GAN Variants: DCGAN, StyleGAN, CycleGAN
- Mode Collapse & solutions
- Implementing GANs in PyTorch

**Project:** Build an Autoencoder for Image Reconstruction

---

## Class 2 — Diffusion Models
- What are Diffusion Models?
- DDPM: Denoising Diffusion Probabilistic Models
- Forward Diffusion Process
- Reverse Diffusion Process
- Training Diffusion Models
- Stable Diffusion Architecture
- Latent Diffusion Models
- Conditional Generation
- Comparison: GANs vs VAEs vs Diffusion

**Project:** Implement a simple diffusion model for image generation

---

## Class 3 — LLMs & Multimodal Generation
### Large Language Models
- Transformer Architecture
- Pretraining & Fine-tuning
- GPT Architecture
- Encoder-only Models (BERT, RoBERTa)
- Tokenization Strategies
- Prompt Engineering Basics

### Multimodal Generation
- Text-to-Image:
  - Stable Diffusion
  - DALL·E
- Image-to-Image:
  - ControlNet
- Text-to-Speech:
  - Audio diffusion models
  - Neural vocoders
- Multimodal Models:
  - CLIP
  - BLIP
  - SAM
  - Vision-Language Models

### Advanced Topics
- Few-shot & Zero-shot Learning
- RAG (Retrieval-Augmented Generation)
- Fine-tuning LLMs for downstream tasks

**Project:** Build a Text-to-Image Generator using Stable Diffusion
