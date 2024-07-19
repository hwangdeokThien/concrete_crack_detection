## Concrete Crack Detection project

This project dive in detecting crack concrete through collected images using PyTorch library, this is also my first project using PyTorch, I've learnt using this library for a while, and this project is the result for the effort.

The project covers many stages of an AI/DS project, containing data stage, modelling stage, evaluation.

-   The dataset used in this project is [Concrete Crack Images for Classification](https://data.mendeley.com/datasets/429vzbgmbx/1), which include 40000 images, 20000 images for each class.
    ![y=1](https://ibb.co/CWFs1hW "positive sample")
    ![y=0](https://ibb.co/0mH2DL3 "negative sample")

-   In the modelling stage, I use the pretrained [ResNet18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) provided by PyTorch, and modify it with a fully connected layer with 2 outputs (2 classes).
