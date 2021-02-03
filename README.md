# StyleTransfer_CycleGan

# Using CycleGan architecture for style trasfer from T1 weighted MRI images to T2 weighted and vice versa.

## Introduction

Misdiagnosis in the medical field is a very serious issue but it’s also uncomfortably common to occur. Imaging procedures in the medical field requires an expert radiologist’s opinion since interpreting them is not a simple binary process (Normal or Abnormal). Even so, one radiologist may see something that another does not. This can lead to conflicting reports and makes it difficult to effectively recommend treatment options to the patient.
One of the complicated tasks in medical imaging is to diagnose MRI (Magnetic Resonance Imaging). Sometimes to interpret the scan, the radiologist needs different variations of the imaging which can drastically enhance the accuracy of diagnosis by providing practitioners with a more comprehensive understanding. But to have access to different imaging is difficult & expensive.  
With the help of Deep learning, we can use style transfer to generate artificial MRI images of different contrast levels from existing MRI scans. This will help to provide a better diagnosis with the help of an additional image. Using the generator models we can create T2 weighted images from T1 weighted MRI image and vice-versa.

![TR1](https://user-images.githubusercontent.com/62643813/106710754-8156d100-661c-11eb-932a-e22bd03b9ca3.png)
TR1 Weighted Image
![Tr2](https://user-images.githubusercontent.com/62643813/106710765-8582ee80-661c-11eb-9a71-d6edd0ce2197.png)
TR2 Weighted Image

## Deep Convolutional General Adversarial Networks (https://www.tensorflow.org/tutorials/generative/dcgan)

DCGAN or Deep Convolutional based General Adversarial networks are one of the interesting innovations in machine learning.  They are kind of generative models by which we can create new data points which can resemble the original data at such a level that it could fool even the human eyes.  The models are built using Convolutional Neural Networks (CNN) and reach this level by training two neural networks simultaneously in an adversarial manner. On one side we have a generator network and on the other a discriminator also called as a classifier. 
The model pair are trained using two sets of images in two different domains. The Generator network tries to produce a fake image in domain _2 corresponding to a real image input in domain_1 and the discriminator tries to distinguish the fake Image in domain_2 from the real image in domain_2. The training is carried out simultaneously, the generator network is trained to fool the discriminator into believing the fake image as real, at the same time discriminator is trained to identify fake as fake. It can be seen that both are trained in an adversarial manner hence the name GAN. Sufficient training makes the Generator produce fake images corresponding to inputs in one domain, which are indistinguishable from the real images in the other domain. This type of translation works for paired images in two domains and depends upon availability of paired images in the two domains. A more generalized generator for unpaired data needs other types training method i.e. DiscoGan,DualGan and CycleGan.

## CycleGan (https://www.tensorflow.org/tutorials/generative/cyclegan)(https://hardikbansal.github.io/CycleGANBlog/)

As mentioned above, for creating a generalized style transfer model using unpaired data a different type of training method is required. CycleGan is one of those methods and has been used in this work to develop generator networks that could translate MRI images in one domain into another. 
![CG_Network_Architecture](https://user-images.githubusercontent.com/62643813/106709796-07721800-661b-11eb-9f28-36ade2cc05c7.jpg)
![CG_Network_Architecture_1](https://user-images.githubusercontent.com/62643813/106709864-1fe23280-661b-11eb-8b97-1c88efd469d1.jpg)
 
The four networks in the above picture and corresponding one in the work are listed below:
#### generator_g - Generator A2B – A Unet network to convert T1 weighted image into T2
#### generator_f – GeneratorB2A – A Unet network to convert T2 weighted image into T1
#### discriminator_x – Discriminator A - CNN based classifier to classify Real T1 and Generated T1 images
#### discriminator_y – Discriminator B - CNN based classifier to classify Real T2 and Generated T2 images
Generator A2B takes in Input_A(real) and produces Generated_B(fake), which is fed to the Discriminator B. 

### Generator_loss

To train the Generator A2B model the loss obtained by comparing the output of Discriminator B for Generated_B images with 1 is used, i.e. it is trained to produce fake images resembling real images in domain B. This is called as Generator_A2B_loss.

### Discriminator_loss

The Discriminator B is also trained with Input_B i.e. real images in domain B. For its training by minimizing the loss calculated by comparing the output for (Input_B(real) with 1 and Generated_B(fake) with 0) i.e. it is trained to classify real and fake images in domain B. Which is adverserial to the training given to Generator A2B.
The same process is followed for Generator B2A and Discriminator A with respective input/real and generated/fake images.

In addition to the Generator_loss as mentioned above, the Generators are further trained by using cycle consistency and identity losses. 

### Cycle_consistency_loss

The Generated_B image is fed to Generator B2A to reproduce the image in domain A called as cyclic_A. The cycle_consistency_loss is calculated by comparing the cyclic_A image with Input_A i.e. the real image. This is ensuring that we are able to get the image back using another generator, thus the difference between the input_A and cyclic_A should be as low as possible. This loss is given a higher weightage than the Generator_loss by multiplying is with a positive integral factor lambda, when calculating the total generator loss.

### Identity_loss

The Generator A2B is trained to produce an image in domain B for an input of real image in domain A. Identity_loss ensures that if Generator A2B is given a real image of domain B as input it produces a fake image in domain B itself. The difference between the real and generated images thus produced is called as the identity_loss.

## Model building

### Generator Network (https://machinelearningmastery.com/how-to-implement-pix2pix-gan-models-from-scratch-with-keras/)

The Unet Architecture is used to build the generator models. The image shows the basic architecture of the Unet network.
<img width="251" alt="Unet Architecture" src="https://user-images.githubusercontent.com/62643813/106710062-5ae46600-661b-11eb-8720-60430ff8a0f6.png">
  
The U-Net model is an encoder-decoder model for image translation, it also uses skip connections that connect layers in the encoder with corresponding layers in the decoder with the same sized feature maps. The connections concatenate the channels of the feature map in the downsampling layer with the feature map in the upsampling layer.
The encoder part of the model is comprised of convolutional layers that use a 3×3 filter to down sample the input source image down to a bottleneck layer. The decoder part of the model reads the bottleneck output and uses transpose convolutional layers to upsample to the required output image size.
The batch normalization is used both during training and while making predictions and is called as instance normalization. Since keras batch normalization function behaves differently during training and inference a separate instance normalization class is defined.
The output of the generator will be of same dimensions as the input and would be fed to the respective discriminator and the other generator to calculate the generator and cyclic loss. 

### Discriminator 

The Discriminator used is simply a CNN classifier. The number of features in the layers can be experimented with to get best results. At the end of the network a convolution layer is added with just one feature to produce a one dimensional output (bs, 4, 4, 1). The respective fake and real image are given as input to the discriminator and a corresponding output with shape (bs, 4, 4, 1) is received. 
This output is used both for calculating discriminator loss and generator loss as described below.

## Loss Calculations

### Generator loss:

The output of the discriminator_y corresponding to the output of the generator_g (fake_y) along with an array of ones of the same size i.e.(bs,4,4,1), fed to the binarycrossentropy function to calculate the gen_g_loss. Same way gen_f_loss is also calculated.

### Discriminator loss:

The outputs of the discriminator_y, corresponding to the output of the generator_g (fake_y) is used along with an array of zeros of the same size i.e.(bs,4,4,1) and the one corresponding to the real_y with array of ones, and the average of the two is taken to calculate the discriminator_y_loss. Same way discriminator_x loss is also calculated.

### Cyclic loss

The absolute difference of the image cycled through the two generator functions (cycled_x) and the original image (real_x) is condensed to a single value by the tf.reduce_mean function. The value is multiplied by a factor of lambda (15) to be used to calculate the total generator loss.

### Identity loss

The absolute difference of the, same_x image produced by giving real_x as input to generator_y and the original image (real_x) is condensed to a single value by the tf.reduce_mean function. The value is multiplied by a factor of lambda*0.25 to be used to calculate the total generator loss.

## Training

The number of images of each type available for training is only 43. The images were augmented by changing the contrast and flipping horizontally thereby increasing the dataset four times. The model was trained in batches of 8x4 images at a time. Since the training needs to be sequenced manually the tf.GradientTape function was used along with tape.gradient and optimizer.apply_gradients for calculating the gradients, updating the weights and optimizing the learning rate.

## Results

The model was able to predict the image in the other domain with a high resemblance as can be seen in the image below.
![MRI Predictions Tr1-Tr2](https://user-images.githubusercontent.com/62643813/106710439-f4137c80-661b-11eb-8a9c-fae7b7947533.png)
![MRI Predictions Tr2-Tr1](https://user-images.githubusercontent.com/62643813/106710454-f970c700-661b-11eb-9875-00947032761a.png)

Initial generator  models were found to predict images resembling the input rather than the images in the other domain. The model was improved by changing the number of features in the discriminator layers, the value of lambda and data augmentation by changing contrast and horizontal flipping. 

 
