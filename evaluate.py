import argparse, os, torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from glob import glob
from model import FMnet
import utils
from torch.utils import data

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
parser.add_argument('--view', type=str, default='bottom', metavar='V',
                    help='View (default: bottom)')
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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Load data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
try:
    test_dataset_files = glob(os.path.join(os.getcwd(), 'data', args.view, 'test', '*'))
except Exception as e:
    raise Exception("Dataset view name not recognized: {}".format(args.view))

# Load data
print("Loading data...")
test_dataset = utils.get_dataset(test_dataset_files, args.view, train=False)
test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
print("Done loading data")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Model setup ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
state_dict = torch.load(model_file)
model = FMnet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device);
model.load_state_dict(state_dict)
model.eval();

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluate ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# compute the intersection over union for each mask
iou_masks, iou_mask_edges = [], []
for batch_data in tqdm(test_loader):
    inputs, masks, mask_edges = batch_data['image'], batch_data['mask'], batch_data['mask_edges']
    pred_masks, pred_edges, _ = utils.predict(model, inputs, sigmoid=True, threshold=0.5)
    iou_masks.append(utils.iou(pred_masks, masks.numpy()))
    iou_mask_edges.append(utils.iou(pred_edges, mask_edges.numpy()))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Compute test accuracy ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if args.verbose:
    print("Mean IoU for masks: ", np.nanmean(iou_masks))
    print("Mean IoU for mask edges: ", np.nanmean(iou_mask_edges))
# Write accuracy to file
with open(os.path.join(output_path, f'model_{max(models)}'+'_accuracy.txt'), 'w') as f:
    f.write("mask IoU: "+str(np.nanmean(iou_masks))+"\n")
    f.write("mask edges IoU: "+str(np.nanmean(iou_mask_edges)))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Plot restuls ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


