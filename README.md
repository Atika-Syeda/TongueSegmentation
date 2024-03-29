# Segmentation model for soft-body object(tongue)

### Introduction
This project focuses on image segmentation of soft-body objects using various convolution neural networks. Specifically, the project uses data from a high-speed imaging camera to perform image segmentation to identify and track a soft-body object, namely the tongue, from static images of mouse faces recorded from a bottom view camera.

### Approach
To perform image segmentation, the project compares different convolution neural networks, including UNet, UNet++, and DeepLabV3, to find a model with the best generalization and test accuracy for the image segmentation task. 

### Results
The performance of different convolutional neural networks (CNNs), including UNet, UNet++, and DeepLabV3, was evaluated on the image segmentation task. The models were trained on the labeled dataset, and their performance was evaluated on a held-out test set using a set of evaluation metrics. The metrics used for comparison included Mask IoU, Mask Edges IoU, Mask Dice Coefficient, inference time (ms) and number of parameters. The quantitative comparison of the models is presented in the table below:

<div style="display: flex; justify-content: center;">
  <img src="figs/results_table.png" width="800" />
</div>

The project successfully identifies and tracks the tongue from the static images of mouse faces using different convolution neural networks. Following is an example visualization of results using UNet - Small model on test data: 

<div style="display: flex; justify-content: center;">
  <img src="figs/out_raw.gif" width="400" />
  <img src="figs/out_maskpred.gif" width="400" />
</div>

The output images show the raw image on the left and the predicted mask overlaid on the raw image on the right.

### Project report 
The project report can be found [here](https://github.com/Atika-Syeda/TongueSegmentation/blob/main/syeda-final_report.pdf). 

### Conclusions
In conclusion, this project demonstrates the effectiveness of the best selected convolutional neural network in performing image segmentation for soft-body object.

# Project setup

Use the following to clone the package:
```
git clone https://github.com/Atika-Syeda/DLCV_final_project.git
```
After cloning, the project structure will be as follows:

```
├── notebooks
├── NestedUNet.py
├── README.md
├── dataset.py
├── environment.yml
├── evaluate.py
├── main.py
├── model.py
├── utils.py
```

Next, install anaconda and create a virtual environment as follows:
```
conda env create -f environment.yml
```
To activate the environment for running the package use:
```
conda activate DLCV
```

### I. Download dataset

Please download the dataset from this [link](https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/asyeda1_jh_edu/Esa5boTlaL5Bn869GK80GLsBCsDTT6dUfEl-8W7-BcxVig?e=4Qbo28) and extract the files in the same directory as the code.
After downloading and extracting the dataset the project structure should be as follows:
```
├── data
    ├── bottom
        ├── train
        ├── test
├── notebooks
├── README.md
├── dataset.py
├── environment.yml
├── evaluate.py
├── main.py
├── model.py
├── utils.py
```

### II. Model training

To train the model, run the following command:
```
python3 main.py
```
Other optional arguments:
```
  -h, --help          show this help message and exit
  --batch-size N      input batch size for training (default: 8)
  --epochs N          number of epochs to train (default: 150)
  --lr LR             learning rate (default: 0.001)
  --weight-decay WD   weight decay (default: 0.0)
  --seed S            random seed (default: 0)
  --verbose V         verbose (default: True)
  --output-dir OP     Output directory (default: output)
  --view V            Camera view (default: bottom)
  --model-name MN     Which model to use, options include [FMnet, UNet,
                      UNet++, DeepLabv3_ResNet50, DeepLabv3_ResNet101, and
                      DeepLabv3_MobileNet] (Default: FMnet)
  --model-weights MW  Model weights to be used if resuming training (default:
                      None)
```

### III. Evaluation

To evaluate the trained model, run the following command:

```
python3 evaluate.py
```
Other optional arguments:
```
  -h, --help         show this help message and exit
  --seed S           random seed (default: 1)
  --verbose V        verbose (default: True)
  --output-dir OP    Output directory (default: output)
  --model-folder MF  Model path containing model_best.pth file (default:
                     trained_models)
  --model-name MW    Which model was used during training, options include
                     [FMnet, UNet, UNet++, DeepLabv3_ResNet50,
                     DeepLabv3_ResNet101, and DeepLabv3_MobileNet] (Default:
                     FMnet)
  --view V           Camera view (default: bottom)
```

### IV. Predict
Support added to perform image segmentation on a video (*.mp4, *.avi, *.mov). To run any video use the following command:
```
python3 predict.py --movie <filepath> --model-name <str> --model-weights <filepath>
```
Other optional arguments:
```
  -h, --help          show this help message and exit
  --movie M           Filepath to the movie to be used
  --model-weights MW  Filepath to the model weights to be used
  --model-name MN     Which model to use, options include [FMnet, UNet,
                      UNet++, and DeepLabv3] (Default: FMnet)
  --output-dir OP     Output directory (Default: None, saved in current
                      directory subfolder "output")
  --output-name O     Output file name (Default: None, video name is used).
  --output-type T     Output file type (Default: .mp4).
  --fps F             Frame rate in number of frames per second for output
                      video (Default: 60).
```
## References
Please see acknowledgements and reference section in the project report for details.
