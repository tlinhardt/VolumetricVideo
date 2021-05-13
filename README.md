# VolumetricVideo
AML Project 2021

## Requirements:  
> Python 2.8+  
Tensorflow 2.x  
TensorBoard  
PIL  
SimpleITK  
numpy  
matplotlib  

## Setup Environment
Data is the [Automated Cardiac Diagnosis Challenge (ACDC)​ MICCAI challenge 2017​ data set](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html)

The data should be unzipped and placed in a directory named ImageData a level lower than the repository location. A third directory named TrainingData will be created by the code to store the generated datasets. The directory logs will be created in the repository directory. All gif outputs and training weights are saved to the repository directory

Folder structure:
```
VolumetricVideo/  
|  {all code is here}  
|  *.gif  
|  *.h5
+--logs/*   
ImageData/
+ --training/  
|  +--patient*/*  
+--testing/  
|  +--patient*/*  
TrainingData/  
|  *.npz  
```

## Run the Code
To run the final results of the project run the entirety of [train_chain.ipynb](train_chain.ipynb)
This will initialize the GPU and:
1. generate all of the training data, 
2. Build the Feedback Loop AutoEncoder
3. Train the Auto Encoder
4. Save sample output gifs
5. Save the Auto Encoder weights
6. Build the GAN and initialize the Generator of the GAN to the saved weights of the Auto Encoder
7. Train the GAN
8. Save the GAN weights
9. Save sample output gifs
10. Start a TensorBoard session to show the tracked metrics from each network that was trained.

Generated outputs of the AE will be saved from unseen data, predicting the next 5 frames as well as the next 50 frames  
Generated outputs of the GAN will also be saved, also of the next 5 and 50.

# What the Code Is
The codebase consists of a base Generator class [(SimpleUgen)](SimpleUGEN.py) modeled after the common UNET architecture, which serves as the encoder/decoder system for an Auto Encoder as well as the generator for a GAN.

The base SimpleUGEN is used in both the [FeedbackGenerator](FeedbackGenerator.py) and the [FeedbackGAN](FeedbackGAN.py) classes, which respectively implement feedback-loop style training of the SimpleUGEN network in the Auto Encoder and GAN cases.

The discriminator for the GAN is implemented in Discriminator.py.

NoiseLayer.py contains a definition of a custom layer that produces random Gaussian noise matching the shape of an input tensor, allowing much easier calling of the inbuilt keras model.fit() function. Thus, there is no need to insert random noise tensors for each training batch.


Data processing functions are defined in setup_data.py:
>setup_feedback_data() generates the dataset for the feedback loop auto-encoder
setup_data() generates the dataset for the feedback loop GAN.

[slicing.ipynb](slicing.ipynb) served to test the data generation and generate the sets for other notebooks since the data does not necessarily need to be generated for each run.

The steps to follow for training the feedback loop Auto-Encoder are in [train_AE.ipynb](train_AE.ipynb). The data set must be generated prior.  

The steps to follow for training the feedback loop GAN are in [train_GAN.ipynb](train_GAN.ipynb). This does not produce good results and the data must be generated prior.