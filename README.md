<p align="center">
  <img src="logo.jpg" alt="Loopy logo" width="440">
</p>

# Hands-On Machine Learning with Scikit-Learn and PyTorch

> [!NOTE]
> **Book Started:** 09/11/2026  
> **Target Finish:** 01/05/2027  
> **Status:** 🟡 In Progress

## Table of Contents

- [Description](#description)
- [Study Plan](#study-plan)
- [Progress](#progress)
  - [Part I: The Fundamentals of Machine Learning](#part-i-the-fundamentals-of-machine-learning)
    - [Chapter 1: The Machine Learning Landscape](#chapter-1-the-machine-learning-landscape)
    - [Chapter 2: End-to-End Machine Learning Project](#chapter-2-end-to-end-machine-learning-project)
    - [Chapter 3: Classification](#chapter-3-classification)
    - [Chapter 4: Training Models](#chapter-4-training-models)
    - [Chapter 5: Decision Trees](#chapter-5-decision-trees)
    - [Chapter 6: Ensemble Learning and Random Forests](#chapter-6-ensemble-learning-and-random-forests)
    - [Chapter 7: Dimensionality Reduction](#chapter-7-dimensionality-reduction)
    - [Chapter 8: Unsupervised Learning Techniques](#chapter-8-unsupervised-learning-techniques)
  - [Part II: Neural Networks and Deep Learning](#part-ii-neural-networks-and-deep-learning)
    - [Chapter 9: Introduction to Artificial Neural Networks](#chapter-9-introduction-to-artificial-neural-networks)
    - [Chapter 10: Building Neural Networks with PyTorch](#chapter-10-building-neural-networks-with-pytorch)
    - [Chapter 11: Training Deep Neural Networks](#chapter-11-training-deep-neural-networks)
    - [Chapter 12: Deep Computer Vision Using Convolutional Neural Networks](#chapter-12-deep-computer-vision-using-convolutional-neural-networks)
    - [Chapter 13: Processing Sequences Using RNNs and CNNs](#chapter-13-processing-sequences-using-rnns-and-cnns)
    - [Chapter 14: Natural Language Processing with RNNs and Attention](#chapter-14-natural-language-processing-with-rnns-and-attention)
    - [Chapter 15: Transformers for Natural Language Processing and Chatbots](#chapter-15-transformers-for-natural-language-processing-and-chatbots)
    - [Chapter 16: Vision and Multimodal Transformers](#chapter-16-vision-and-multimodal-transformers)
    - [Chapter 17: Speeding Up Transformers](#chapter-17-speeding-up-transformers)
    - [Chapter 18: Autoencoders, GANs, and Diffusion Models](#chapter-18-autoencoders-gans-and-diffusion-models)
    - [Chapter 19: Reinforcement Learning](#chapter-19-reinforcement-learning)
  - [Appendices](#appendices)
    - [Appendix A: Autodiff](#appendix-a-autodiff)
    - [Appendix B: Mixed Precision and Quantization](#appendix-b-mixed-precision-and-quantization)
- [Study Log](#study-log)
- [Repository Structure](#repository-structure)
- [Screenshots](#screenshots)
- [Resources](#resources)
- [Learned](#learned)
- [Challenges & Open Questions](#challenges--open-questions)
- [Conclusions](#conclusions)

## Description

Here I keep track of all the exercises and contents I learn while reading the book *"Hands-On Machine Learning with Scikit-Learn and PyTorch"* by **Aurélien Géron**.

> [!IMPORTANT]
> No content of the book is available here. This repository is just a place where I leave my solutions to the exercises, my own notes, and some of the concepts I learn along the way.

## Study Plan

| Item | Details |
| ---- | ------- |
| **Goal** | Build a solid ML + deep learning foundation and apply it to my own projects|
| **Schedule** | 3 hours/day, Mon–Fri (Except Wed) |
| **Method** | Read → Take notes → Run the code in Colab → Do the exercises → Review |
| **Review cadence** | Every Sunday |
| **Book code** | [Official book repository](https://github.com/ageron/handson-mlp) |

## Progress

**Overall:** `0 / 19` chapters completed (`0%`)

| Legend | Meaning |
| ------ | ------- |
| ✔️ | Chapter completed (exercises included) |
| 👈 | Where I am right now |

Each chapter links to its own **notes** and **exercises**:

### Part I: The Fundamentals of Machine Learning

#### Chapter 1: The Machine Learning Landscape

📝 [Notes](PART%20I%20-%20The%20Fundamentals%20of%20Machine%20Learning/Chapter%2001%20-%20The%20Machine%20Learning%20Landscape/notes.md) · 💻 [Exercises](PART%20I%20-%20The%20Fundamentals%20of%20Machine%20Learning/Chapter%2001%20-%20The%20Machine%20Learning%20Landscape/exercises.ipynb) · 👈 Current chapter

- [ ] **What Is Machine Learning?**
- [ ] **Why Use Machine Learning?**
- [ ] **Examples of Applications**
- [ ] **Types of Machine Learning Systems**
  - [ ] Training Supervision
  - [ ] Batch Versus Online Learning
  - [ ] Instance-Based Versus Model-Based Learning
- [ ] **Main Challenges of Machine Learning**
  - [ ] Insufficient Quantity of Training Data
  - [ ] Nonrepresentative Training Data
  - [ ] Poor-Quality Data
  - [ ] Irrelevant Features
  - [ ] Overfitting the Training Data
  - [ ] Underfitting the Training Data
  - [ ] Deployment Issues
  - [ ] Stepping Back
- [ ] **Testing and Validating**
  - [ ] Hyperparameter Tuning and Model Selection
  - [ ] Data Mismatch
- [ ] **Exercises**
- [ ] **Notes written**

#### Chapter 2: End-to-End Machine Learning Project

📝 [Notes](PART%20I%20-%20The%20Fundamentals%20of%20Machine%20Learning/Chapter%2002%20-%20End-to-End%20Machine%20Learning%20Project/notes.md) · 💻 [Exercises](PART%20I%20-%20The%20Fundamentals%20of%20Machine%20Learning/Chapter%2002%20-%20End-to-End%20Machine%20Learning%20Project/exercises.ipynb)

- [ ] **Working with Real Data**
- [ ] **Look at the Big Picture**
  - [ ] Frame the Problem
  - [ ] Select a Performance Measure
  - [ ] Check the Assumptions
- [ ] **Get the Data**
  - [ ] Running the Code Examples Using Google Colab
  - [ ] Saving Your Code Changes and Your Data
  - [ ] The Power and Danger of Interactivity
  - [ ] Book Code Versus Notebook Code
  - [ ] Download the Data
  - [ ] Take a Quick Look at the Data Structure
  - [ ] Create a Test Set
- [ ] **Explore and Visualize the Data to Gain Insights**
  - [ ] Visualizing Geographical Data
  - [ ] Look for Correlations
  - [ ] Experiment with Attribute Combinations
- [ ] **Prepare the Data for Machine Learning Algorithms**
  - [ ] Clean the Data
  - [ ] Handling Text and Categorical Attributes
  - [ ] Feature Scaling and Transformation
  - [ ] Custom Transformers
  - [ ] Transformation Pipelines
- [ ] **Select and Train a Model**
  - [ ] Train and Evaluate on the Training Set
  - [ ] Better Evaluation Using Cross-Validation
- [ ] **Fine-Tune Your Model**
  - [ ] Grid Search
  - [ ] Randomized Search
  - [ ] Ensemble Methods
  - [ ] Analyzing the Best Models and Their Errors
  - [ ] Evaluate Your System on the Test Set
- [ ] **Launch, Monitor, and Maintain Your System**
- [ ] **Try It Out!**
- [ ] **Exercises**
- [ ] **Notes written**

#### Chapter 3: Classification

📝 [Notes](PART%20I%20-%20The%20Fundamentals%20of%20Machine%20Learning/Chapter%2003%20-%20Classification/notes.md) · 💻 [Exercises](PART%20I%20-%20The%20Fundamentals%20of%20Machine%20Learning/Chapter%2003%20-%20Classification/exercises.ipynb)

- [ ] **MNIST**
- [ ] **Training a Binary Classifier**
- [ ] **Performance Measures**
  - [ ] Measuring Accuracy Using Cross-Validation
  - [ ] Confusion Matrices
  - [ ] Precision and Recall
  - [ ] The Precision/Recall Trade-Off
  - [ ] The ROC Curve
- [ ] **Multiclass Classification**
- [ ] **Error Analysis**
- [ ] **Multilabel Classification**
- [ ] **Multioutput Classification**
- [ ] **Exercises**
- [ ] **Notes written**

#### Chapter 4: Training Models

📝 [Notes](PART%20I%20-%20The%20Fundamentals%20of%20Machine%20Learning/Chapter%2004%20-%20Training%20Models/notes.md) · 💻 [Exercises](PART%20I%20-%20The%20Fundamentals%20of%20Machine%20Learning/Chapter%2004%20-%20Training%20Models/exercises.ipynb)

- [ ] **Linear Regression**
  - [ ] The Normal Equation
  - [ ] Computational Complexity
- [ ] **Gradient Descent**
  - [ ] Batch Gradient Descent
  - [ ] Stochastic Gradient Descent
  - [ ] Mini-Batch Gradient Descent
- [ ] **Polynomial Regression**
- [ ] **Learning Curves**
- [ ] **Regularized Linear Models**
  - [ ] Ridge Regression
  - [ ] Lasso Regression
  - [ ] Elastic Net Regression
  - [ ] Early Stopping
- [ ] **Logistic Regression**
  - [ ] Estimating Probabilities
  - [ ] Training and Cost Function
  - [ ] Decision Boundaries
  - [ ] Softmax Regression
- [ ] **Exercises**
- [ ] **Notes written**

#### Chapter 5: Decision Trees

📝 [Notes](PART%20I%20-%20The%20Fundamentals%20of%20Machine%20Learning/Chapter%2005%20-%20Decision%20Trees/notes.md) · 💻 [Exercises](PART%20I%20-%20The%20Fundamentals%20of%20Machine%20Learning/Chapter%2005%20-%20Decision%20Trees/exercises.ipynb)

- [ ] **Training and Visualizing a Decision Tree**
- [ ] **Making Predictions**
- [ ] **Estimating Class Probabilities**
- [ ] **The CART Training Algorithm**
- [ ] **Computational Complexity**
- [ ] **Gini Impurity or Entropy?**
- [ ] **Regularization Hyperparameters**
- [ ] **Regression**
- [ ] **Sensitivity to Axis Orientation**
- [ ] **Decision Trees Have a High Variance**
- [ ] **Exercises**
- [ ] **Notes written**

#### Chapter 6: Ensemble Learning and Random Forests

📝 [Notes](PART%20I%20-%20The%20Fundamentals%20of%20Machine%20Learning/Chapter%2006%20-%20Ensemble%20Learning%20and%20Random%20Forests/notes.md) · 💻 [Exercises](PART%20I%20-%20The%20Fundamentals%20of%20Machine%20Learning/Chapter%2006%20-%20Ensemble%20Learning%20and%20Random%20Forests/exercises.ipynb)

- [ ] **Voting Classifiers**
- [ ] **Bagging and Pasting**
  - [ ] Bagging and Pasting in Scikit-Learn
  - [ ] Out-of-Bag Evaluation
  - [ ] Random Patches and Random Subspaces
- [ ] **Random Forests**
  - [ ] Extra-Trees
  - [ ] Feature Importance
- [ ] **Boosting**
  - [ ] AdaBoost
  - [ ] Gradient Boosting
  - [ ] Histogram-Based Gradient Boosting
- [ ] **Stacking**
- [ ] **Exercises**
- [ ] **Notes written**

#### Chapter 7: Dimensionality Reduction

📝 [Notes](PART%20I%20-%20The%20Fundamentals%20of%20Machine%20Learning/Chapter%2007%20-%20Dimensionality%20Reduction/notes.md) · 💻 [Exercises](PART%20I%20-%20The%20Fundamentals%20of%20Machine%20Learning/Chapter%2007%20-%20Dimensionality%20Reduction/exercises.ipynb)

- [ ] **The Curse of Dimensionality**
- [ ] **Main Approaches for Dimensionality Reduction**
  - [ ] Projection
  - [ ] Manifold Learning
- [ ] **PCA**
  - [ ] Preserving the Variance
  - [ ] Principal Components
  - [ ] Projecting Down to d Dimensions
  - [ ] Using Scikit-Learn
  - [ ] Explained Variance Ratio
  - [ ] Choosing the Right Number of Dimensions
  - [ ] PCA for Compression
  - [ ] Randomized PCA
  - [ ] Incremental PCA
- [ ] **Random Projection**
- [ ] **LLE**
- [ ] **Other Dimensionality Reduction Techniques**
- [ ] **Exercises**
- [ ] **Notes written**

#### Chapter 8: Unsupervised Learning Techniques

📝 [Notes](PART%20I%20-%20The%20Fundamentals%20of%20Machine%20Learning/Chapter%2008%20-%20Unsupervised%20Learning%20Techniques/notes.md) · 💻 [Exercises](PART%20I%20-%20The%20Fundamentals%20of%20Machine%20Learning/Chapter%2008%20-%20Unsupervised%20Learning%20Techniques/exercises.ipynb)

- [ ] **Clustering Algorithms: k-means and DBSCAN**
  - [ ] k-Means Clustering
  - [ ] Limits of k-Means
  - [ ] Using Clustering for Image Segmentation
  - [ ] Using Clustering for Semi-Supervised Learning
  - [ ] DBSCAN
  - [ ] Other Clustering Algorithms
- [ ] **Gaussian Mixtures**
  - [ ] Using Gaussian Mixtures for Anomaly Detection
  - [ ] Selecting the Number of Clusters
  - [ ] Bayesian Gaussian Mixture Models
  - [ ] Other Algorithms for Anomaly and Novelty Detection
- [ ] **Exercises**
- [ ] **Notes written**

### Part II: Neural Networks and Deep Learning

#### Chapter 9: Introduction to Artificial Neural Networks

📝 [Notes](PART%20II%20-%20Neural%20Networks%20and%20Deep%20Learning/Chapter%2009%20-%20Introduction%20to%20Artificial%20Neural%20Networks/notes.md) · 💻 [Exercises](PART%20II%20-%20Neural%20Networks%20and%20Deep%20Learning/Chapter%2009%20-%20Introduction%20to%20Artificial%20Neural%20Networks/exercises.ipynb)

- [ ] **From Biological to Artificial Neurons**
  - [ ] Biological Neurons
  - [ ] Logical Computations with Neurons
  - [ ] The Perceptron
  - [ ] The Multilayer Perceptron and Backpropagation
- [ ] **Building and Training MLPs with Scikit-Learn**
  - [ ] Regression MLPs
  - [ ] Classification MLPs
- [ ] **Hyperparameter Tuning Guidelines**
  - [ ] Number of Hidden Layers
  - [ ] Number of Neurons per Hidden Layer
  - [ ] Learning Rate
  - [ ] Batch Size
  - [ ] Other Hyperparameters
- [ ] **Exercises**
- [ ] **Notes written**

#### Chapter 10: Building Neural Networks with PyTorch

📝 [Notes](PART%20II%20-%20Neural%20Networks%20and%20Deep%20Learning/Chapter%2010%20-%20Building%20Neural%20Networks%20with%20PyTorch/notes.md) · 💻 [Exercises](PART%20II%20-%20Neural%20Networks%20and%20Deep%20Learning/Chapter%2010%20-%20Building%20Neural%20Networks%20with%20PyTorch/exercises.ipynb)

- [ ] **PyTorch Fundamentals**
  - [ ] PyTorch Tensors
  - [ ] Hardware Acceleration
  - [ ] Autograd
- [ ] **Implementing Linear Regression**
  - [ ] Linear Regression Using Tensors and Autograd
  - [ ] Linear Regression Using PyTorch's High-Level API
- [ ] **Implementing a Regression MLP**
- [ ] **Implementing Mini-Batch Gradient Descent Using DataLoaders**
- [ ] **Model Evaluation**
- [ ] **Building Nonsequential Models Using Custom Modules**
  - [ ] Building Models with Multiple Inputs
  - [ ] Building Models with Multiple Outputs
- [ ] **Building an Image Classifier with PyTorch**
  - [ ] Using TorchVision to Load the Dataset
  - [ ] Building the Classifier
- [ ] **Fine-Tuning Neural Network Hyperparameters with Optuna**
- [ ] **Saving and Loading PyTorch Models**
- [ ] **Compiling and Optimizing a PyTorch Model**
- [ ] **Exercises**
- [ ] **Notes written**

#### Chapter 11: Training Deep Neural Networks

📝 [Notes](PART%20II%20-%20Neural%20Networks%20and%20Deep%20Learning/Chapter%2011%20-%20Training%20Deep%20Neural%20Networks/notes.md) · 💻 [Exercises](PART%20II%20-%20Neural%20Networks%20and%20Deep%20Learning/Chapter%2011%20-%20Training%20Deep%20Neural%20Networks/exercises.ipynb)

- [ ] **The Vanishing/Exploding Gradients Problems**
  - [ ] Glorot Initialization and He Initialization
  - [ ] Better Activation Functions
  - [ ] Batch Normalization
  - [ ] Layer Normalization
  - [ ] Gradient Clipping
- [ ] **Reusing Pretrained Layers**
  - [ ] Transfer Learning with PyTorch
  - [ ] Unsupervised Pretraining
  - [ ] Pretraining on an Auxiliary Task
- [ ] **Faster Optimizers**
  - [ ] Momentum
  - [ ] Nesterov Accelerated Gradient
  - [ ] AdaGrad
  - [ ] RMSProp
  - [ ] Adam
  - [ ] AdaMax
  - [ ] NAdam
  - [ ] AdamW
- [ ] **Learning Rate Scheduling**
  - [ ] Exponential Scheduling
  - [ ] Cosine Annealing
  - [ ] Performance Scheduling
  - [ ] Warming Up the Learning Rate
  - [ ] Cosine Annealing with Warm Restarts
  - [ ] 1cycle Scheduling
- [ ] **Avoiding Overfitting Through Regularization**
  - [ ] ℓ1 and ℓ2 Regularization
  - [ ] Dropout
  - [ ] Monte Carlo Dropout
  - [ ] Max-Norm Regularization
- [ ] **Practical Guidelines**
- [ ] **Exercises**
- [ ] **Notes written**

#### Chapter 12: Deep Computer Vision Using Convolutional Neural Networks

📝 [Notes](PART%20II%20-%20Neural%20Networks%20and%20Deep%20Learning/Chapter%2012%20-%20Deep%20Computer%20Vision%20Using%20Convolutional%20Neural%20Networks/notes.md) · 💻 [Exercises](PART%20II%20-%20Neural%20Networks%20and%20Deep%20Learning/Chapter%2012%20-%20Deep%20Computer%20Vision%20Using%20Convolutional%20Neural%20Networks/exercises.ipynb)

- [ ] **The Architecture of the Visual Cortex**
- [ ] **Convolutional Layers**
  - [ ] Filters
  - [ ] Stacking Multiple Feature Maps
  - [ ] Implementing Convolutional Layers with PyTorch
- [ ] **Pooling Layers**
- [ ] **Implementing Pooling Layers with PyTorch**
- [ ] **CNN Architectures**
  - [ ] LeNet-5
  - [ ] AlexNet
  - [ ] GoogLeNet
  - [ ] ResNet
  - [ ] Xception
  - [ ] SENet
  - [ ] Other Noteworthy Architectures
  - [ ] Choosing the Right CNN Architecture
  - [ ] GPU RAM Requirements: Inference Versus Training
  - [ ] Reversible Residual Networks (RevNets)
- [ ] **Implementing a ResNet-34 CNN Using PyTorch**
- [ ] **Using TorchVision's Pretrained Models**
- [ ] **Pretrained Models for Transfer Learning**
- [ ] **Classification and Localization**
- [ ] **Object Detection**
  - [ ] Fully Convolutional Networks
  - [ ] You Only Look Once
- [ ] **Object Tracking**
- [ ] **Semantic Segmentation**
- [ ] **Exercises**
- [ ] **Notes written**

#### Chapter 13: Processing Sequences Using RNNs and CNNs

📝 [Notes](PART%20II%20-%20Neural%20Networks%20and%20Deep%20Learning/Chapter%2013%20-%20Processing%20Sequences%20Using%20RNNs%20and%20CNNs/notes.md) · 💻 [Exercises](PART%20II%20-%20Neural%20Networks%20and%20Deep%20Learning/Chapter%2013%20-%20Processing%20Sequences%20Using%20RNNs%20and%20CNNs/exercises.ipynb)

- [ ] **Recurrent Neurons and Layers**
  - [ ] Memory Cells
  - [ ] Input and Output Sequences
- [ ] **Training RNNs**
- [ ] **Forecasting a Time Series**
  - [ ] The ARMA Model Family
  - [ ] Preparing the Data for Machine Learning Models
  - [ ] Forecasting Using a Linear Model
  - [ ] Forecasting Using a Simple RNN
  - [ ] Forecasting Using a Deep RNN
  - [ ] Forecasting Multivariate Time Series
  - [ ] Forecasting Several Time Steps Ahead
  - [ ] Forecasting Using a Sequence-to-Sequence Model
- [ ] **Handling Long Sequences**
  - [ ] Fighting the Unstable Gradients Problem
  - [ ] Tackling the Short-Term Memory Problem
- [ ] **Exercises**
- [ ] **Notes written**

#### Chapter 14: Natural Language Processing with RNNs and Attention

📝 [Notes](PART%20II%20-%20Neural%20Networks%20and%20Deep%20Learning/Chapter%2014%20-%20Natural%20Language%20Processing%20with%20RNNs%20and%20Attention/notes.md) · 💻 [Exercises](PART%20II%20-%20Neural%20Networks%20and%20Deep%20Learning/Chapter%2014%20-%20Natural%20Language%20Processing%20with%20RNNs%20and%20Attention/exercises.ipynb)

- [ ] **Generating Shakespearean Text Using a Character RNN**
  - [ ] Creating the Training Dataset
  - [ ] Embeddings
  - [ ] Building and Training the Char-RNN Model
  - [ ] Generating Fake Shakespearean Text
- [ ] **Sentiment Analysis Using Hugging Face Libraries**
  - [ ] Tokenization Using the Hugging Face Tokenizers Library
  - [ ] Reusing Pretrained Tokenizers
  - [ ] Building and Training a Sentiment Analysis Model
  - [ ] Bidirectional RNNs
  - [ ] Reusing Pretrained Embeddings and Language Models
  - [ ] Task-Specific Classes
  - [ ] The Trainer API
  - [ ] Hugging Face Pipelines
- [ ] **An Encoder-Decoder Network for Neural Machine Translation**
- [ ] **Beam Search**
- [ ] **Attention Mechanisms**
- [ ] **Exercises**
- [ ] **Notes written**

#### Chapter 15: Transformers for Natural Language Processing and Chatbots

📝 [Notes](PART%20II%20-%20Neural%20Networks%20and%20Deep%20Learning/Chapter%2015%20-%20Transformers%20for%20Natural%20Language%20Processing%20and%20Chatbots/notes.md) · 💻 [Exercises](PART%20II%20-%20Neural%20Networks%20and%20Deep%20Learning/Chapter%2015%20-%20Transformers%20for%20Natural%20Language%20Processing%20and%20Chatbots/exercises.ipynb)

- [ ] **Attention Is All You Need: The Original Transformer Architecture**
  - [ ] Positional Encodings
  - [ ] Multi-Head Attention
  - [ ] Building the Rest of the Transformer
- [ ] **Building an English-to-Spanish Transformer**
- [ ] **Encoder-Only Transformers for Natural Language Understanding**
  - [ ] BERT's Architecture
  - [ ] BERT Pretraining
  - [ ] BERT Fine-Tuning
  - [ ] Other Encoder-Only Models
- [ ] **Decoder-Only Transformers**
  - [ ] GPT-1 Architecture and Generative Pretraining
  - [ ] GPT-2 and Zero-Shot Learning
  - [ ] GPT-3, In-Context Learning, One-Shot Learning, and Few-Shot Learning
  - [ ] Using GPT-2 to Generate Text
  - [ ] Using GPT-2 for Question Answering
  - [ ] Downloading and Running an Even Larger Model: Mistral-7B
- [ ] **Turning a Large Language Model into a Chatbot**
  - [ ] Fine-Tuning a Model for Chatting and Following Instructions Using SFT and RLHF
  - [ ] Direct Preference Optimization (DPO)
  - [ ] Fine-Tuning a Model Using the TRL Library
  - [ ] From a Chatbot Model to a Full Chatbot System
  - [ ] Model Context Protocol
  - [ ] Libraries and Tools
- [ ] **Encoder-Decoder Models**
- [ ] **Exercises**
- [ ] **Notes written**

#### Chapter 16: Vision and Multimodal Transformers

📝 [Notes](PART%20II%20-%20Neural%20Networks%20and%20Deep%20Learning/Chapter%2016%20-%20Vision%20and%20Multimodal%20Transformers/notes.md) · 💻 [Exercises](PART%20II%20-%20Neural%20Networks%20and%20Deep%20Learning/Chapter%2016%20-%20Vision%20and%20Multimodal%20Transformers/exercises.ipynb)

- [ ] **Vision Transformers**
  - [ ] RNNs with Visual Attention
  - [ ] DETR: A CNN-Transformer Hybrid for Object Detection
  - [ ] The Original ViT
  - [ ] Data-Efficient Image Transformer
  - [ ] Pyramid Vision Transformer for Dense Prediction Tasks
  - [ ] The Swin Transformer: A Fast and Versatile ViT
  - [ ] DINO: Self-Supervised Visual Representation Learning
  - [ ] Other Major Vision Models and Techniques
- [ ] **Multimodal Transformers**
  - [ ] VideoBERT: A BERT Variant for Text plus Video
  - [ ] ViLBERT: A Dual-Stream Transformer for Text plus Image
  - [ ] CLIP: A Dual-Encoder Text plus Image Model Trained with Contrastive Pretraining
  - [ ] DALL·E: Generating Images from Text Prompts
  - [ ] Perceiver: Bridging High-Resolution Modalities with Latent Spaces
  - [ ] Perceiver IO: A Flexible Output Mechanism for the Perceiver
  - [ ] Flamingo: Open-Ended Visual Dialogue
  - [ ] BLIP and BLIP-2
- [ ] **Other Multimodal Models**
- [ ] **Exercises**
- [ ] **Notes written**

#### Chapter 17: Speeding Up Transformers

📝 [Notes](PART%20II%20-%20Neural%20Networks%20and%20Deep%20Learning/Chapter%2017%20-%20Speeding%20Up%20Transformers/notes.md) · 💻 [Exercises](PART%20II%20-%20Neural%20Networks%20and%20Deep%20Learning/Chapter%2017%20-%20Speeding%20Up%20Transformers/exercises.ipynb)

- [ ] **(Add sections as you reach this chapter)**
- [ ] **Exercises**
- [ ] **Notes written**

#### Chapter 18: Autoencoders, GANs, and Diffusion Models

📝 [Notes](PART%20II%20-%20Neural%20Networks%20and%20Deep%20Learning/Chapter%2018%20-%20Autoencoders%2C%20GANs%2C%20and%20Diffusion%20Models/notes.md) · 💻 [Exercises](PART%20II%20-%20Neural%20Networks%20and%20Deep%20Learning/Chapter%2018%20-%20Autoencoders%2C%20GANs%2C%20and%20Diffusion%20Models/exercises.ipynb)

- [ ] **Efficient Data Representations**
- [ ] **Performing PCA with an Undercomplete Linear Autoencoder**
- [ ] **Stacked Autoencoders**
  - [ ] Implementing a Stacked Autoencoder Using PyTorch
  - [ ] Visualizing the Reconstructions
  - [ ] Anomaly Detection Using Autoencoders
  - [ ] Visualizing the Fashion MNIST Dataset
  - [ ] Unsupervised Pretraining Using Stacked Autoencoders
  - [ ] Tying Weights
  - [ ] Training One Autoencoder at a Time
- [ ] **Convolutional Autoencoders**
- [ ] **Denoising Autoencoders**
- [ ] **Sparse Autoencoders**
- [ ] **Variational Autoencoders**
- [ ] **Generating Fashion MNIST Images**
  - [ ] Discrete Variational Autoencoders
- [ ] **Generative Adversarial Networks**
  - [ ] The Difficulties of Training GANs
- [ ] **Diffusion Models**
- [ ] **Exercises**
- [ ] **Notes written**

#### Chapter 19: Reinforcement Learning

📝 [Notes](PART%20II%20-%20Neural%20Networks%20and%20Deep%20Learning/Chapter%2019%20-%20Reinforcement%20Learning/notes.md) · 💻 [Exercises](PART%20II%20-%20Neural%20Networks%20and%20Deep%20Learning/Chapter%2019%20-%20Reinforcement%20Learning/exercises.ipynb)

- [ ] **What Is Reinforcement Learning?**
- [ ] **Policy Gradients**
  - [ ] Introduction to the Gymnasium Library
  - [ ] Neural Network Policies
  - [ ] Evaluating Actions: The Credit Assignment Problem
  - [ ] Solving the CartPole Using Policy Gradients
- [ ] **Value-Based Methods**
  - [ ] Markov Decision Processes
  - [ ] Temporal Difference Learning
  - [ ] Q-Learning
  - [ ] Exploration Policies
  - [ ] Approximate Q-Learning and Deep Q-Learning
  - [ ] Implementing Deep Q-Learning
  - [ ] DQN Improvements
- [ ] **Actor-Critic Algorithms**
- [ ] **Mastering Atari Breakout Using the Stable-Baselines3 PPO Implementation**
- [ ] **Overview of Some Popular RL Algorithms**
- [ ] **Exercises**
- [ ] **Notes written**

### Appendices

#### Appendix A: Autodiff

📝 [Notes](Appendices/Appendix%20A%20-%20Autodiff/notes.md)

- [ ] **Manual Differentiation**
- [ ] **Finite Difference Approximation**
- [ ] **Forward-Mode Autodiff**
- [ ] **Reverse-Mode Autodiff**

#### Appendix B: Mixed Precision and Quantization

📝 [Notes](Appendices/Appendix%20B%20-%20Mixed%20Precision%20and%20Quantization/notes.md)

- [ ] **Common Number Representations**
- [ ] **Reduced Precision Models**
- [ ] **Mixed-Precision Training**
- [ ] **Quantization**
  - [ ] Linear Quantization
  - [ ] Post-Training Quantization Using torch.ao.quantization
  - [ ] Quantization-Aware Training (QAT)
  - [ ] Quantizing LLMs Using the bitsandbytes Library
- [ ] **Using Pre-Quantized Models**

## Study Log

| Date | Time Spent | What I Covered | Notes / Blockers |
| ---- | ---------- | -------------- | ---------------- |
| 09/11/2026 | 2h | Chapter 1 – What Is Machine Learning? | — |
| | | | |

## Repository Structure

```
.
├── README.md
├── logo.jpg
├── PART I - The Fundamentals of Machine Learning/
│   ├── Chapter 01 - The Machine Learning Landscape/
│   │   ├── notes.md
│   │   ├── exercises.ipynb
│   │   └── images/
│   ├── Chapter 02 - End-to-End Machine Learning Project/
│   │   ├── notes.md
│   │   ├── exercises.ipynb
│   │   └── images/
│   └── ... (Chapters 03–08, same layout)
├── PART II - Neural Networks and Deep Learning/
│   ├── Chapter 09 - Introduction to Artificial Neural Networks/
│   │   ├── notes.md
│   │   ├── exercises.ipynb
│   │   └── images/
│   └── ... (Chapters 10–19, same layout)
├── Appendices/
│   ├── Appendix A - Autodiff/
│   │   ├── notes.md
│   │   └── images/
│   └── Appendix B - Mixed Precision and Quantization/
│       ├── notes.md
│       └── images/
├── Projects/        # mini-projects applying what I learned
│   └── README.md
└── Resources/       # cheat sheets and references
    └── README.md
```

## Screenshots

> [!TIP]
> Here I post images, plots, and outputs that I find relevant.

<!-- ![Description](path/to/image.png) -->

## Resources

- [Aurélien Géron – Book Repository](https://github.com/ageron/handson-mlp)
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/)
- [Google Colab](https://colab.research.google.com/)

## Learned

**Chapters 1–3**

- **[Concept]** — [one-line explanation in my own words]

**Chapters 4–8**

- _Coming soon…_

**Chapters 9–19**

- _Coming soon…_

## Challenges & Open Questions

- [ ] [Something I didn't fully understand — revisit later]
- [x] [Resolved question → answer or link]

## Conclusions

Here I will leave my conclusions once I finish the book.

---

<p align="center">Last updated: 09/11/2026</p>
