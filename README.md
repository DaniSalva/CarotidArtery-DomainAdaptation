# CarotidArtery-DomainAdaptation
Final Project for the Artificial Intelligence Research Master from the International University Menendez Pelayo (UIMP) and the Spanish AI Society (AEPIA).

### Environment Semantic Segmentation

It contains the dockerfiles for building the images needed for running the semantic segmentation model based on Tiramisu architecture (DenseNet) implemented by the authors: https://github.com/beareme/keras_semantic_segmentation. A README file is provided with the instructions.

### Domain Adaptation.
It is divided in three folders:
- CycleGANs: Code for training cycleGAN architecture. Using the implementation provided by the authors: https://github.com/eriklindernoren/Keras-GAN
- USGeneralization: Code for training the adversarial model proposed in https://github.com/xy0806/congeneric_renderer.
- Evaluation: code with metrics for evaluate the CIMT segmentation.
- Image transformations: Auxiliar codes for image manipulation (crop, resize, split, encode to binary labels...).
