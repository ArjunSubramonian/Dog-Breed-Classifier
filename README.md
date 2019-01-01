# Dog Breed Classifier
Python and Keras implementations of 3 different deep learning classifiers:

- Simple logistic regression unit
- Multi-layer fully-connected neural network
- 50-layer residual convolutional neural network (ResNet50)

that, given an image of a dog, can determine the dog’s breed. The ResNet50 was initialized during training with pre-trained ImageNet weights and only trained partially due to a lack of computational power.

# Data prep
Download the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/).\
\
Run:\
`python ./Data\ Python\ Scripts/dataset_prep.py`\
`python ./Data\ Python\ Scripts/data_prep.py`

# Simple logistic regression unit
To train the model, run:\
`python ./Logistic\ Regression\ Python\ Scripts/dog_trainer.py`\
\
To predict the breed of the dog in image.jpg, run:\
`python ./Logistic\ Regression\ Python\ Scripts/dog_classifier.py`

# Multi-layer fully-connected neural network
To train the model, run:\
`python ./Deep\ FC\ NN\ Python\ Scripts/dnn_trainer.py`\
\
To predict the breed of the dog in image.jpg, run:\
`python ./Deep\ FC\ NN\ Python\ Scripts/dnn_classifier.py`

# 50-layer residual convolutional neural network (ResNet50)
To prepare the data, run:\
`python ./ResNet50\ Python\ Scripts/resnet50_prep.py`\
\
To train the model, run:\
`python ./ResNet50\ Python\ Scripts/resnet50_train.py`\
\
To predict the breed of the dog in image.jpg, run:\
`python ./ResNet50\ Python\ Scripts/resnet50_classify.py`

