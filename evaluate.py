import argparse, os, torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from glob import glob
import utils
from torch.utils import data
from matplotlib import animation
from IPython.display import HTML
import time

from model import FMnet, UNet
from NestedUNet import NestedUNet
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenet_v3_large

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Training settings ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
parser = argparse.ArgumentParser(description='PyTorch DLCV')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--verbose', type=bool, default=True, metavar='V',
                    help='verbose (default: True)')   
parser.add_argument(('--output-dir'), type=str, default='output', metavar='OP',
                    help='Output directory (default: output)')
parser.add_argument('--model-folder', type=str, default='trained_models', metavar='MF',
                    help='Model path containing model_best.pth file (default: trained_models)')
parser.add_argument('--model-name', type=str, default='FMnet', metavar='MW',
                    help='Which model was used during training, options include [FMnet, UNet, UNet++, DeepLabv3_ResNet50, DeepLabv3_ResNet101, and DeepLabv3_MobileNet] (Default: FMnet)')
parser.add_argument('--view', type=str, default='bottom', metavar='V',
                    help='Camera view (default: bottom)')
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
# Use best model
model_file = os.path.join(model_path, 'model_best.pth')
if args.verbose:
    print(f"Using model: {model_file}")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Load data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
try:
    test_dataset_files = glob(os.path.join(os.getcwd(), 'data', args.view, 'test', '*'))
except Exception as e:
    raise Exception("Dataset view name not recognized: {}".format(args.view))

# Load data
if args.verbose:
    print("Loading data...")
test_dataset = utils.get_dataset(test_dataset_files, args.view, train=False)
test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=16)
if args.verbose:
    print("Done loading data")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Model setup ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
state_dict = torch.load(model_file)
print(f"Using model: {args.model_name}")
if args.model_name == 'FMnet':
    model = FMnet() 
elif args.model_name == 'UNet':
    model = UNet()
elif args.model_name == 'UNet++':
    model = NestedUNet(num_classes=3, input_channels=1)
elif args.model_name == 'DeepLabv3_ResNet50':
    model = deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=3)
elif args.model_name == 'DeepLabv3_ResNet101':
    model = deeplabv3_resnet101(weights=None, weights_backbone=None, num_classes=3)
elif args.model_name == 'DeepLabv3_MobileNet':
    model = deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=3)
else:
    raise Exception("Model name not recognized: {}".format(args.model_name))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device);
model.load_state_dict(state_dict)
model.eval();

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluate ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# compute the intersection over union and dice coefficient for each mask
iou_masks, iou_mask_edges, dice_masks, dice_mask_edges, inference_times = [], [], [], [], []
for batch_data in tqdm(test_loader):
    inputs, masks, mask_edges = batch_data['image'], batch_data['mask'], batch_data['mask_edges']
    start_time = time.time()
    pred_masks, pred_edges, _ = utils.predict(model, inputs, sigmoid=True, threshold=0.5, model_name=args.model_name)
    inference_times.append(time.time() - start_time)
    iou_masks.append(utils.iou(pred_masks, masks.numpy()))
    iou_mask_edges.append(utils.iou(pred_edges, mask_edges.numpy()))
    dice_masks.append(utils.dice(pred_masks, masks.numpy()))
    dice_mask_edges.append(utils.dice(pred_edges, mask_edges.numpy()))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Compute test accuracy ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if args.verbose:
    print("Mean IoU for masks: ", np.nanmean(iou_masks))
    print("Mean IoU for mask edges: ", np.nanmean(iou_mask_edges))
    print("Mean Dice Coefficient for masks: ", np.nanmean(dice_masks))
    print("Mean Dice Coefficient for mask edges: ", np.nanmean(dice_mask_edges))
    print("Mean Inference Time (ms): ", np.nanmean(inference_times) * 1000)
    print("Model parameters: ", sum(param.numel() for param in model.parameters()))
# Write accuracy to file
#with open(os.path.join(output_path, f'model_{max(models)}'+'_accuracy.txt'), 'w') as f:
with open(os.path.join(output_path, 'model_best_accuracy.txt'), 'w') as f:
    f.write("mask IoU: " + str(np.nanmean(iou_masks)) + "\n")
    f.write("mask edges IoU: " + str(np.nanmean(iou_mask_edges)) + "\n")
    f.write("mask Dice Coefficient: " + str(np.nanmean(dice_masks)) + "\n")
    f.write("mask edges Dice Coefficient: " + str(np.nanmean(dice_mask_edges)) + "\n")
    f.write("Mean Inference Time (ms): " + str(np.nanmean(inference_times) * 1000) + "\n")
    f.write("Model parameters: " + str(sum(param.numel() for param in model.parameters())) + "\n")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Plot restuls ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create an animation of video and model predictions
fig, ax = plt.subplots(2, 3, figsize=(8, 5), dpi=300)

num_frames = test_dataset.__len__() 
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=16)
iterator = iter(test_loader)

start_idx = 0
batch_data = next(iterator)
imgs, masks, edges = batch_data['image'], batch_data['mask'], batch_data['mask_edges']
# threshold masks and edges
masks = (masks > 0.5).float()
edges = (edges > 0.5).float()
pred_masks, pred_edges, _ = utils.predict(model, imgs, model_name=args.model_name)
# Plot the original image
img_plot = ax[0,0].imshow(imgs[0].squeeze(), cmap='gray')
ax[0,0].axis("off")
ax[0,0].set_title("Frame: " + str(start_idx))
mask_plot = ax[0,1].imshow(masks[0].squeeze(), cmap='Greens', alpha=1, vmin=0, vmax=1)
ax[0,1].axis("off")
ax[0,1].set_title("Mask: " + str(start_idx))
mask_edge_plot = ax[0,2].imshow(edges[0].squeeze(), cmap='Reds', alpha=.4, vmin=0, vmax=1)
ax[0,2].axis("off")
ax[0,2].set_title("Edges: " + str(start_idx))

# Plot the predictions
frame_plot = ax[1,0].imshow(imgs[0].squeeze(), cmap='gray')
ax[1,0].axis("off")
ax[1,0].set_title("Frame: " + str(start_idx))
pred_mask_plot = ax[1,1].imshow(pred_masks[0].squeeze(), cmap='Greens', alpha=1)
ax[1,1].axis("off")
ax[1,1].set_title("Predicted mask: " + str(start_idx))
pred_mask_edge_plot = ax[1,2].imshow(pred_edges[0].squeeze(), cmap='Reds', alpha=.4)
ax[1,2].axis("off")
ax[1,2].set_title("Predicted edges: " + str(start_idx))

def animate(i):
    batch_data = next(iterator)
    imgs, masks, edges = batch_data['image'], batch_data['mask'], batch_data['mask_edges']
    masks = (masks > 0.5).float()
    edges = (edges > 0.5).float()
    pred_masks, pred_edges, _ = utils.predict(model, imgs)
    img_plot.set_data(imgs[0].squeeze())
    ax[0,0].set_title("Frame: " + str(i))
    frame_plot.set_data(imgs[0].squeeze())
    ax[1,0].set_title("Frame: " + str(i))
    mask_plot.set_data(masks[0].squeeze())
    ax[0,1].set_title("Mask: " + str(i))
    pred_mask_plot.set_data(pred_masks[0].squeeze())
    ax[1,1].set_title("Predicted mask: " + str(i))
    mask_edge_plot.set_data(edges[0].squeeze())
    ax[0,2].set_title("Edges: " + str(i))
    pred_mask_edge_plot.set_data(pred_edges[0].squeeze())
    ax[1,2].set_title("Predicted edges: " + str(i))
    return (frame_plot, mask_plot, mask_edge_plot, img_plot, pred_mask_plot, pred_mask_edge_plot)

if args.verbose:
    print("Creating animation...")
anim = animation.FuncAnimation(fig, animate, frames=num_frames-5, interval=100, repeat=False, blit=True)
# HTML(anim.to_html5_video())
# save to mp4 using ffmpeg writer
writervideo = animation.FFMpegWriter(fps=60)
iterator = iter(test_loader)
anim.save(os.path.join(output_path, 'model_best_pred.mp4'), writer=writervideo)
if args.verbose:
    print("Saved animation to file: ", os.path.join(output_path, 'model_best_pred.mp4'))
plt.close()
