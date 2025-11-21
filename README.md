#  Attack a Self-Driving Car with an Adversarial Stop Sign and Noise

**Course:** 24-784 Special Topics: Trustworthy AI Autonomy

**Model Focus**: *ResNet-18 and AlexNet*

Imagine a self-driving car has a camera that reads road signs. It uses an AI "brain" (a Neural Network) to look at a picture and say, "*That is a Stop Sign*."

This project explores three ways to test—and trick—that brain:

1. **Adversarial Attacks (Tricking the AI)**: We take a normal picture of a stop sign and change the pixels just slightly. To a human, it still looks exactly like a stop sign. However, we change it in a specific mathematical way that confuses the AI, making it think the stop sign is actually a "street car" or something else entirely. This shows how fragile AI vision can be.

2. **Visualization (Reading the Mind)**: We use special tools to "look inside" the AI's brain. We generate images that show us what specific layers of the AI are looking for (like edges, textures, or shapes). This helps us understand why it makes certain decisions.

3. **Robustness (Stress Testing)**: We add random "static" (noise) to the image thousands of times to see how often the AI gets confused. It's like checking if the car can still see the stop sign during a heavy rainstorm or with a dirty camera lens.

## 🚀 Project Description

This project assesses AI robustness through three main tasks:

1. Adversarial Example Generation: Implementing FGM (Fast Gradient Method) and PGD (Projected Gradient Descent) attacks on a pretrained ResNet-18 model.

2. Neural Network Visualization: Using FlashTorch for maximal activation visualization and MapExtract for saliency maps to interpret CNN layers (AlexNet).

3. Probabilistic Robustness: Using Monte Carlo (MC) sampling to estimate the misclassification rate of a network under Gaussian noise.

## 🛠️ Installation & Requirements

This project relies on Python and several specific deep learning libraries. It is recommended to run this in a Google Colab environment or a virtual environment.

Dependencies:

```
Python 3.x
PyTorch / TorchVision
Numpy
Matplotlib
```
**Specific Security/Visualization Libraries:**

```bash
pip install scratchai-nightly
pip install torchvision
pip install flashtorch
pip install mapextrackt
```
## 📂 Project Structure

```bash
├── images/
│   └── stop_sign.jpg       # The base image used for attacks
├── notebooks/
│   └── Project_1.ipynb     # Main execution notebook (Colab compatible)
├── outputs/                # Generated adversarial images and plots
└── README.md
```
### 📝 Exercises & Usage

#### Exercise 1: Adversarial Example Generation

Target Model: ResNet-18

Baseline: Load the stop_sign.jpg and check the model's confidence.

Untargeted Attack (Noise): Add uniform random noise to the image and observe if classification breaks.

Untargeted Attack (FGM): Use the Fast Gradient Method to create a perturbation that forces a misclassification.

Targeted Attack (PGD): Use Projected Gradient Descent to force the model to classify the Stop Sign specifically as a "Street Car" (Class ID: 829).

#### Exercise 2: Neural Network Visualization

Target Model: AlexNet

Maximal Activation: Visualize what specific filters (5, 10, 15, 20) "want to see" in Layer 0 and Layer 10.

Saliency Maps: Generate a heat map showing which pixels in the stop sign image are most important for the AI's decision.

#### Exercise 3: Evaluation of Probabilistic Robustness

Target Model: ResNet-18

Monte Carlo Estimation: Estimate the misclassification rate ($\mu$) when the image is subjected to Gaussian noise ($\sigma^2=0.2$).

Convergence Plot: Plot the estimated mean $\hat{\mu}_n$ against the number of samples ($n$) ranging from 50 to 500.

Relative Error: Calculate and plot the relative error of the estimator.

Adversarial Robustness: Repeat the estimation using the adversarial example generated in Exercise 1 as the center point, rather than the original clean image.


📜 License & Integrity

This project is for the 24-784 course at CMU. Please abide by the Academic Integrity Policy.

