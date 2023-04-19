# DLCV_final_project
Deep Learning for Computer Vision final project 

## Summary 


## Project setup

Use the following to clone the package:
```
git clone https://github.com/Atika-Syeda/DLCV_final_project.git
```
After cloning, the project structure will be as follows:

```
├── environment.yml
├── evaluate.py
├── main.py
├── model.py
├── README.md
├── utils.py
```

Next, install anaconda and create the virtual environment as follows:
```
conda env create -f environment.yml
```
To activate the environment for running the package use:
```
conda activate DLCV
```

## Usage

### Download dataset

Please download the dataset from ... and extract the files in the same directory as the code.
After downloading and extracting the dataset the project structure should look as follows:
```
├── data
├── environment.yml
├── evaluate.py
├── main.py
├── model.py
├── README.md
├── utils.py
```

### Training

To train the model, run the following command:
```
python3 main.py
```
To modify any default parameters, use the following command:
```
python3 main.py --batch_size <batch_size> --epochs <epochs> --lr <learning_rate> --seed <seed> --verbose <bool> --output-dir <output_dir> --data-augmentation <data_augmentation>
```
Use --help to get mpre details for the different tags. The default command `python3 main.py` will save the output of the model/script in a folder called output by default. 


### Evaluation

To evaluate the trained model, run the following command:

```
python3 evaluate.py
```

The file by default uses the last saved epoch for evaluation. To modify any default parameters including the model file saved in output/trained_models, use the following command:
```
python3 evaluate.py --seed <seed> --verbose <bool> --output-dir <output_dir> --model-folder <model_folder>
```

## References
Please see acknowledgements and reference section in the attached report for details.
