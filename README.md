## Concrete Crack Detection project

This project focuses on detecting cracks in concrete through collected images using the PyTorch library. It marks my first venture into using PyTorch, representing the culmination of my learning efforts with this powerful tool.

![demo](https://github.com/user-attachments/assets/60e755dd-9453-4062-94a7-e945e062eaa9)


The project encompasses several stages typical of an AI/DS project, including data preprocessing, model training, and evaluation. Additionally, a simple UI application is deployed to allow for easy testing of the model.

### Details

-   **Dataset**: The dataset used in this project is [Concrete Crack Images for Classification](https://data.mendeley.com/datasets/429vzbgmbx/1), which include 40000 images, 20000 images for each class.

    ![Positive sample (y=1)](https://github.com/user-attachments/assets/cbe0fc43-10c3-4301-ba0b-454618ee596e)

    ![Negative sample (y=0)](https://github.com/user-attachments/assets/1bbb103a-5443-475a-bc0d-c89571be4954)

-   **Modelling** The pretrained [ResNet18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) provided by PyTorchwas used, modified with a fully connected layer with 2 outputs (for binary classification).

### Training the Model

To train the model, you can use the `notebook.ipynb` for both data exploration and training. Alternatively, a `train.py` script is provided for training and saving the model, though its evaluation capabilities are limited. Both methods will export the model as `model.pth`.

You can use the following command the run the file

```bash
python train.py
```

### Running the Application

A simple UI application is deployed using Streamlit. To run it locally, use the following command:

```bash
streamlit app.py
```

or using the provided link to my deployment.
