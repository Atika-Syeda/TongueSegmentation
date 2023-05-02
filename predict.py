#  TODO: test this script

import argparse, os, torch
from tqdm import tqdm
from model import FMnet, UNet
from NestedUNet import NestedUNet
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenet_v3_large
import utils

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Training settings ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
parser = argparse.ArgumentParser(description='PyTorch DLCV - predict')
parser.add_argument('--movie', type=str, metavar='M',
                    help='Filepath to the movie to be used')
parser.add_argument('--model-weights', type=str, metavar='MW',
                    help='Filepath to the model weights to be used (Default: None)')
parser.add_argument('--model-name', type=str, default='FMnet', metavar='MN',
                    help='Which model to use, options include [FMnet, UNet, UNet++, and DeepLabv3] (Default: FMnet)')
parser.add_argument('--output-dir', type=str, default=None, metavar='OP',
                    help='Output directory (Default: None)')
parser.add_argument('--output-name', type=str, default=None, metavar='O',
                    help='Output file name (Default is None so video name is used).')
parser.add_argument('--output-type', type=str, default='.mp4', metavar='T',
                    help='Output file type (Default: .mp4).')
parser.add_argument('--fps', type=int, default=60, metavar='F',
                    help='Frames per second for output video (Default: 60).')
args = parser.parse_args()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GPU/CPU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if torch.cuda.is_available():
    use_gpu = True
    print("GPU available and will be used for training")
else:
	use_gpu = False
	print("GPU not available, using CPU instead")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Directory setup ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Create output directory if it does not exist
if args.output_dir is None:
    args.output_dir = os.path.join(os.getcwd(), 'output')
os.makedirs(args.output_dir, exist_ok=True)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Load model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
print("Loading model weights from {}".format(args.model_weights))
model.load_state_dict(torch.load(args.model_weights))
model.eval();

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Load movie ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print(f"Loading movie: {args.movie}")
frames = utils.load_movie(args.movie)
frames = utils.preprocess_imgs(frames, resize_shape=(256, 256))
print(f"Movie loaded, {len(frames)} frames found")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Predict ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Predicting masks")
pred_masks, pred_edges, pred_frames = [], [], []
for frame in tqdm(frames):
    frame = frame.unsqueeze(0).to(device)
    pred_mask, pred_edge, _ = utils.predict(model, frame, sigmoid=True, threshold=0.5, model_name=args.model_name)
    pred_masks.append(pred_mask)
    pred_edges.append(pred_edge)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Save ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Saving output videos...")
if args.output_name is None:
    args.output_name = os.path.splitext(os.path.basename(args.movie))[0]
mask_output_path = os.path.join(args.output_dir, args.output_name +"_maskpred" + args.output_type)
utils.save_video_with_mask(pred_masks, frames, mask_output_path, fps=args.fps)
raw_output_path = os.path.join(args.output_dir, args.output_name +"_raw" + args.output_type)
utils.save_video(frames, raw_output_path, fps=args.fps)
print(f"Videos saved to {raw_output_path}")

