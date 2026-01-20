# Super-Resolution

## Introduction

This project implements deep learning models for image super-resolution, a technique that enhances low-resolution images to higher resolution with improved visual quality and detail. Super-resolution has practical applications in medical imaging, satellite imagery, video enhancement, and general image restoration tasks.

The project explores two neural network architectures, each incorporating enhanced techniques to improve reconstruction quality. We use a combination of reconstruction loss (L1) and perceptual loss to achieve sharp, visually pleasing results.

## Project Structure

### Notebooks

This project contains two complementary Jupyter notebooks that guide you through the exploration and implementation of super-resolution models:

#### 1. **0.0-exploratory.ipynb** - Data Exploration
This notebook provides an initial exploration of the dataset, including:
- Loading and visualizing high-resolution, downscale to get low-resolution image pairs
- Visualizing the data image statistics
- Preparing data for model training

#### 2. **0.1-modeling-and-training.ipynb** - Model Development and Training
This notebook implements the complete training pipeline, featuring:
- Dataset loading and preprocessing with custom PyTorch dataset classes
- Implementation of two super-resolution models with increasing complexity
- Training loops with loss computation and optimization
- Validation metrics and visual results comparison

This is where the core modeling work happens. Follow along to understand the training process and see the models in action.

## Models

### Model 1: Baseline Super-Resolution Model

The baseline model serves as a foundational architecture for image super-resolution. It consists of:
- **Convolutional feature extraction layers** to learn low-level image features
- **Residual blocks** for stable gradient flow during training
- **Upsampling layers** to increase spatial resolution from 32×32 to 128×128
- **Reconstruction layer** to produce the final super-resolved image

**Performance Characteristics:**
- Achieves approximately 23 dB PSNR after 50 epochs
- Tends to produce blurry results due to limited model capacity
- Serves as a baseline for comparison with more advanced approaches

### Model 2: Enhanced Residual Super-Resolution Model

The enhanced model builds upon the baseline with significant improvements:
- **Deeper architecture** with more residual connections for better feature learning
- **Residual learning framework** that explicitly learns the difference between high-resolution and upsampled low-resolution images
- **Perceptual loss integration** comparing feature representations rather than just pixel values
- **Combined loss function** using L1 reconstruction loss + weighted perceptual loss

**Key Improvements:**
- Generates visibly sharper and more detailed super-resolved images
- Better preservation of fine details and textures
- Improved visual quality despite modest PSNR gains (constrained by 128×128 resolution limit)
- More efficient training with residual learning approach

## Dataset

The dataset consists of 17354 paired high-resolution (128×128) and low-resolution (32×32) image patches stored in `.npz` format:
- `hr_images.npz` - High-resolution images
- `hr_lr_images.npz` - Paired high-resolution and low-resolution images for training


## Requirements

See `requirements.txt` for complete dependency list and versions.
We use virtual environments to manage dependencies and cuda for GPU acceleration

## Results

The enhanced residual model demonstrates superior visual quality compared to the baseline:
- **Sharper reconstructions** with better edge definition
- **Improved detail preservation** in textured regions
- **More natural appearance** compared to blurry baseline results

The improvement is particularly visible in visual comparisons, even though PSNR gains are modest due to the inherent constraints of small image sizes in the dataset.

## Future Improvements

Potential directions for further enhancement:
- Implement advanced architectures (SRGAN, SRResNet, ESPCN)
- Explore different loss functions (adversarial loss, style loss)
- Increase training dataset size and diversity
- Experiment with different upsampling techniques
