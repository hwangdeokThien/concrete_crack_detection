## Concrete Crack Detection project

This project dive in detecting crack concrete through collected images using PyTorch library, this is also my first project using PyTorch, I've learnt using this library for a while, and this project is the result for the effort.

The project covers many stages of an AI/DS project, containing data stage, modelling stage, evaluation.

-   The dataset used in this project is [Concrete Crack Images for Classification](https://data.mendeley.com/datasets/429vzbgmbx/1), which include 40000 images, 20000 images for each class.

    ![Positive sample (y=1)](https://github.com/user-attachments/assets/cbe0fc43-10c3-4301-ba0b-454618ee596e)
    
    ![Negative sample (y=0)](https://github.com/user-attachments/assets/1bbb103a-5443-475a-bc0d-c89571be4954)


-   In the modelling stage, I use the pretrained [ResNet18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) provided by PyTorch, and modify it with a fully connected layer with 2 outputs (2 classes).
