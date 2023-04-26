# DLCV final project
Deep Learning for Computer Vision final project 

## Summary 
Image segmentation of soft-body object using different convolution neural networks. This project uses data from high-speed imaging camera and performs image segmentation to identify/track a soft-body object (tongue) from static images of mouse face recorded from a bottom view camera. The project compared various segmentation neural networks (UNet, UNet++, DeepLabV3) to find a model with the best generalization and test accuracy for the image segmentation task.

## Project setup

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

## Usage

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
To modify any default parameters, use the following command:
```
python3 main.py --batch_size <int> --epochs <int> --lr <float> --weight-decay <float> --seed <float> --verbose <bool> --output-dir <filepath> --data-augmentation <bool> --model-weights <filepath> --model-name <str>
```
Use --help to get mpre details for the different tags. The default command `python3 main.py` will save the output of the model/script in a folder called `output` by default. 


### III. Evaluation

To evaluate the trained model, run the following command:

```
python3 evaluate.py
```

The file uses the best model state saved for evaluation. To modify any default parameters, use the following tags:
```
python3 evaluate.py --seed <int> --verbose <bool> --output-dir <filepath> --model-folder <filename> --model-name <str>
```

### IV. Predict
Support added to perform image segmentation on a video (*.mp4, *.avi, *.mov). To run any video use the following command:
```
python3 predict.py --movie <filepath> --model-name <str> --model-state <filepath>
```
Some other features supported by the following tags:
```
python3 predict.py --movie <filepath> --model-name <str> --model-state <filepath> --output-type <str> --fps <int>
```
Use --help to get mpre details for the different tags. The default command will save the output video in the currect folder. 

## References
Please see acknowledgements and reference section in the project report for details.
