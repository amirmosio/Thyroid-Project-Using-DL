# Thyroid Carcinoma and Analysis of Cytology Slides Using Deep Networks
ChatGPT generated

## Description
Thyroid carcinoma is a type of cancer that begins in the thyroid gland, located at the base of the neck. It may cause symptoms like a neck lump, voice changes, and difficulty swallowing. There are several types, including the common Papillary thyroid carcinoma. Most thyroid cancers grow slowly and can be cured with treatment. However, some can be aggressive.
There are different methods to detect this disease:
 - Physical Exam
 - Ultrasound Imaging
 - Fine-Needle Aspiration Biopsy
 - ...

Here I tried to use the biopsy data to train an image model for the detection of this disease. A biopsy is a medical procedure where a small sample of tissue is taken from the body for closer examination. This is typically done when an initial test suggests an area of tissue in the body isn't normal. The sample can be taken from almost anywhere on or in your body, including the skin, organs, and other structures. The extracted sample is then examined under a microscope to determine the presence or extent of a disease.

## Stages
 
#### Dataset Collection
 - [National cancer institute tcga-thca project](https://portal.gdc.cancer.gov/projects/TCGA-THCA)

 - [Papanicolaou society of cytopathology](https://www.papsociety.org/image-atlas/)

 - [The stanford tissue microarray database](https://tma.im/cgi-bin/home.pl)

#### Data Preparation
In this stage, we are dealing with huge cytology slides which volume size can take up to 10GB.
Here I tried to split each slide into multiple patches with size 512*512 which can be given to an image model.
One more step in this section is to ignore empty fragments.

This was previously done by different methods.
One is to try to filter pixels with a color threshold.
The disadvantage of this method is that there are some scanning methods in which they use different chemicals and slides would get different colors based on the chemical that was used.
So for each coloring technique, we have to set a color filter.
One other method for this purpose is to use image models like U-Net and train on a dataset to segment informative data and obviously, this is computationally costly.
Here I tried a different method which is computationally efficient and is a more general way of doing this.
I used the variance of the Laplacian of image and found on threshold to filter patches below.
Laplacian of an image is somehow the second derivative of that image.
In this image lines and borders of shapes are more bold.
Using the variance of a patch I can tell how much each patch is informative and does contain shapes and complex patterns.

#### Data Augmentation
 - Fourier Domain Adaptation
 - GaussNoise
 - Color Jitter
 - Mixup
 - Flip, Rotate, Random Scale

#### Training
 - Resnet 101

#### Device
 - Quadro RTX 8000
