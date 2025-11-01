# From Pixels to Perceptions: A Conceptual Image Vision Roadmap

> We are mainly interested in conceptual mapping with the focus on building decision-making intuition; rather than hands-on coding implementation. 

## Part 1: Foundations of Visual Understanding

### Day 1: Image as structured data

- Basics of image
    - How images differ from traditional data
    - Understanding pixels, color channels, image representation.
- Convolution operation
    - Introduction to convolution
    - Sliding window concept
- Filtering (kernels)
    - Different kernel types
    - Different filters applications (blur, sharpen, edge) using kernel
    - Intuition behind spatial filtering
- Edge detection
    - Exploring Sobel, Prewitt, and Canny detectors to extract meaningful structure from images

### Day 2: So... What IS CNN?

- CNN basics
    - Why ANN fails on image data
    - How CNNs extend the concept of convolution for feature extraction
- CNN Structure
    - Convolution
    - Pooling
- End-to-end basic CNN operation
    - Simple CNN for classification
    - Feature visualization using GradCAM
- Brief introduction to milestone models(ResNet, UNet, AlexNet, VGG etc)
    - Overview of key CNN architectures and innovations

### Day 3: Beyond classifications

- Data Augmentation
    - Different transformation operations(rotation, scaling, flipping)
- Image segmentation
    - Pixel-level classification, Mask generation
    - Implementation of UNet as a baseline model
- Object Detection
    - Localization concepts: Bounding boxes, IoU, Anchor boxes
    - Non-max suppression
    - YOLO vs R-CNN
- Applications
    - YOLO, FRCNN
    - Trade-offs between accuracy and speed
    - Comparison between a segmentor and a detector

## Part 2: Vision Transformers and Modern Architectures

### Day 1: ViT - The global attention
- Introduction to Vision Transformers
    - Concept of patch embeddings, position embeddings, and transformer encoder
- CNN vs Transformer
    - How ViT differs from CNN in feature extraction and representation
- ViT Architecture
    - Layer normalization, MLP head, class token, and training approach

### Day 2: Transfer learning and hybrid trends
- Transfer learning in ViT
    - Using pre-trained ViT models on small datasets
- Applications of ViT
    - Image classification, segmentation, and object detection
- Hybrid models
    - Combining CNN and Transformer backbones

### Day 3: 
- Modern ViT Variant
    - DeiT, Swin Transformer, ConvNeXt, and other efficient ViTs
- Introduction to VLMs
    - Overview of key architectures like CLIP, BLIP, and Flamingo that align image and text embeddings. Discussion on contrastive learning, image-text matching, and cross-attention mechanisms.

## Part 3: Generative Vision Models

### Day 1: Creating images
- VAE
- GAN
- Diffusion models

## Part 4: Motion & 3D Vision

### Day 1: Object in motion 

- TimeSformer
- Optical Flow
- Action Recognition

### Day 2: The 3D realm

- NeRF
- Point Cloud – Gaussian Splatting
- MonoCular depth