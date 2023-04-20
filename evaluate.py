import argparse
from tqdm import tqdm
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from glob import glob
from model import FMnet
import utils

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Training settings ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
parser = argparse.ArgumentParser(description='PyTorch GTSRB')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--verbose', type=bool, default=True, metavar='V',
                    help='verbose (default: True)')   
parser.add_argument(('--output-dir'), type=str, default='output', metavar='OP',
                    help='Output directory (default: output)')
parser.add_argument('--model-folder', type=str, default='trained_models', metavar='MF',
                    help='Models path (default: trained_models)')
args = parser.parse_args()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Setup ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    use_gpu = True
    print("Using GPU")
else:
	use_gpu = False
	print("Using CPU")

# Create output directory if it does not exist
output_path = os.path.join(os.getcwd(), args.output_dir)
if not os.path.exists(output_path):
    os.makedirs(output_path)

model_path = os.path.join(os.getcwd(), args.output_dir, args.model_folder)
if not os.path.exists(model_path):
    raise Exception("Model path does not exist")
# Use last epoch model
models = glob(os.path.join(model_path, '*.pth'))
models = [i.split("_")[-1].split(".")[0] for i in models]
models = [int(i) for i in models]
model_file = os.path.join(model_path, f'model_{max(models)}.pth')
if args.verbose:
    print(f"Using model: {model_file}")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Model setup ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
state_dict = torch.load(model_file)
model = GTSRNet(n_classes=43)
model.load_state_dict(state_dict)
model.eval();

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluate ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
test_dir = os.path.join(os.getcwd(), 'GTSRB/Final_Test/Images')
output_file = open(os.path.join(output_path, args.model_folder+'_pred.csv'), "w")
output_file.write("Filename,ClassId\n")
for f in tqdm(sorted(glob(os.path.join(test_dir, "*.ppm"))), disable=(not args.verbose)):
    output = torch.zeros([1, 43], dtype=torch.float32)
    with torch.no_grad():
        data = utils.transform(utils.pil_loader(f))
        data = data.view(1, data.size(0), data.size(1), data.size(2))
        data = Variable(data)
        output = output.add(model(data))
        pred = output.data.max(1, keepdim=True)[1]
        file_id = f[0:5]
        output_file.write("%s,%d\n" % (file_id, pred))
output_file.close()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Compute test accuracy ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
gt_file = os.path.join(os.getcwd(), 'GTSRB/GT-final_test.csv')
gt = pd.read_csv(gt_file, sep=';')
pred_file = os.path.join(output_path, args.model_folder+'_pred.csv')
pred = pd.read_csv(pred_file, sep=',')

if args.verbose:
    print("Accuracy: ", (gt['ClassId']==pred['ClassId']).sum()/len(gt)*100, " %")
# Write accuracy to file
with open(os.path.join(output_path, args.model_folder+'_accuracy.txt'), 'w') as f:
    f.write(str((gt['ClassId']==pred['ClassId']).sum()/len(gt)*100))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Plot restuls ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


