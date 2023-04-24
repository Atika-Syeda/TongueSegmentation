import argparse, os, torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from glob import glob
from model import FMnet
import utils
from torch.utils import data
from matplotlib import animation
from IPython.display import HTML

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Training settings ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
parser = argparse.ArgumentParser(description='PyTorch DLCV')
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
# Use best model
#models = glob(os.path.join(model_path, '*.pth'))
#models = [i.split("_")[-1].split(".")[0] for i in models]
#models = [int(i) for i in models]
#model_file = os.path.join(model_path, f'model_{max(models)}.pth')
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
#with open(os.path.join(output_path, f'model_{max(models)}'+'_accuracy.txt'), 'w') as f:
with open(os.path.join(output_path, 'model_best_accuracy.txt'), 'w') as f:
    f.write("mask IoU: "+str(np.nanmean(iou_masks))+"\n")
    f.write("mask edges IoU: "+str(np.nanmean(iou_mask_edges)))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Plot restuls ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create an animation of video and model predictions
fig, ax = plt.subplots(1, 3, figsize=(10, 5), dpi=100)

num_frames = test_dataset.__len__() 
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=16)
iterator = iter(test_loader)

start_idx = 0
batch_data = next(iterator)
imgs, masks, edges = batch_data['image'], batch_data['mask'], batch_data['mask_edges']
pred_masks, pred_edges, _ = utils.predict(model, imgs)
# Plot the first frame
frame_plot = ax[0].imshow(imgs[0].squeeze(), cmap='gray')
ax[0].axis("off")
ax[0].set_title("Frame: " + str(start_idx))
mask_plot = ax[1].imshow(pred_masks[0].squeeze(), cmap='Greens', alpha=1)
ax[1].axis("off")
ax[1].set_title("Predicted mask: " + str(start_idx))
mask_edge_plot = ax[2].imshow(pred_edges[0].squeeze(), cmap='Reds', alpha=.4)
ax[2].axis("off")
ax[2].set_title("Predicted edges: " + str(start_idx))

def animate(i):
    batch_data = next(iterator)
    imgs, masks, edges = batch_data['image'], batch_data['mask'], batch_data['mask_edges']
    pred_masks, pred_edges, _ = utils.predict(model, imgs)
    frame_plot.set_data(imgs[0].squeeze())
    ax[0].set_title("Frame: " + str(i))
    mask_plot.set_data(pred_masks[0].squeeze())
    ax[1].set_title("Predicted mask: " + str(i))
    mask_edge_plot.set_data(pred_edges[0].squeeze())
    ax[2].set_title("Predicted edges: " + str(i))
    return (frame_plot, mask_plot, mask_edge_plot)

if args.verbose:
    print("Creating animation...")
anim = animation.FuncAnimation(fig, animate, frames=num_frames-5, interval=100, repeat=False, blit=True)
HTML(anim.to_html5_video())
# save to mp4 using ffmpeg writer
writervideo = animation.FFMpegWriter(fps=60)
iterator = iter(test_loader)
#anim.save(os.path.join(output_path, f'model_{max(models)}'+'_pred.mp4'), writer=writervideo)
anim.save(os.path.join(output_path, 'model_best_pred.mp4'), writer=writervideo)
if args.verbose:
    #print("Saved animation to file: ", os.path.join(output_path, f'model_{max(models)}'+'_pred.mp4'))
    print("Saved animation to file: ", os.path.join(output_path, 'model_best_pred.mp4'))
plt.close()

