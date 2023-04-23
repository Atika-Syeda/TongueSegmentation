import numpy as np
import argparse, os, utils 
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils import data
from glob import glob
from model import FMnet

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Training settings ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
parser = argparse.ArgumentParser(description='PyTorch DLCV')
parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 8)')
parser.add_argument('--epochs', type=int, default=150, metavar='N',
                    help='number of epochs to train (default: 150)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--weight-decay', type=float, default=0.001, metavar='WD',
                    help='weight decay (default: 0.9)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--verbose', type=bool, default=True, metavar='V',
                    help='verbose (default: True)')   
parser.add_argument(('--output-dir'), type=str, default='output', metavar='OP',
                    help='Output directory (default: output)')
parser.add_argument('--data-augmentation', type=bool, default=False, metavar='DA',
                    help='Data augmentation (default: False)')
parser.add_argument('--view', type=str, default='bottom', metavar='V',
                    help='View (default: bottom)')
parser.add_argument('--model-weights', type=str, default=None, metavar='MW',
                    help='Model weights (default: None)')
args = parser.parse_args()

torch.manual_seed(args.seed)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Setup ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if torch.cuda.is_available():
    use_gpu = True
    print("GPU available and will be used for training")
else:
	use_gpu = False
	print("GPU not available, using CPU instead")

# Create output directory if it does not exist
output_path = os.path.join(os.getcwd(), args.output_dir)
if not os.path.exists(output_path):
    os.makedirs(output_path)
# Create trained models directory if it does not exist
trained_models_path = os.path.join(output_path, 'trained_models')
if not os.path.exists(trained_models_path):
    os.makedirs(trained_models_path)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Data loaders ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
try:
    train_dataset_files = glob(os.path.join(os.getcwd(), 'data', args.view, 'train', '*'))
except Exception as e:
    raise Exception("Dataset view name not recognized: {}".format(args.view))

# Load data
print("Loading data...")
train_dataset = utils.get_dataset(train_dataset_files, args.view, train=True)
print("Done loading data")

# Divide data into training and validation set
train_ratio = 0.9
n_train_examples = int(len(train_dataset) * train_ratio)
n_val_examples = len(train_dataset) - n_train_examples
train_data, val_data = data.random_split(train_dataset, [n_train_examples, n_val_examples])
if args.verbose:
    print(f"Number of training samples = {len(train_data)}")
    print(f"Number of validation samples = {len(val_data)}")

# Create data loader for training and validation
train_loader = data.DataLoader(train_data, shuffle=True, batch_size=args.batch_size, num_workers=16)
val_loader = data.DataLoader(val_data, shuffle=False, batch_size=args.batch_size, num_workers=16)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Model settings ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
model = FMnet() 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device);
if args.model_weights is not None:
    print("Loading model weights from {}".format(args.model_weights))
    model.load_state_dict(torch.load(args.model_weights))

# Set optimizer and learning rate scheduler
learning_rate = args.lr
change_lr_every = 30
change_lr_epoch = 60
if args.epochs > change_lr_epoch:
    LR = np.ones(change_lr_epoch)*learning_rate
    for i in range(int(np.ceil((args.epochs-change_lr_epoch)/change_lr_every))):
        LR = np.append(LR, LR[-1]/2 * np.ones(change_lr_every))
else:
    LR = np.ones(args.epochs)*learning_rate
    LR[-6:-3] = LR[-1]/10
    LR[-3:] = LR[-1]/25

# Plot the learning rate schedule
fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)
ax.plot(LR)
ax.set_xlabel('Epochs')
ax.set_ylabel('Learning Rate')
ax.set_title('Learning Rate Scheduler')
# hide the spines between ax and ax2
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.savefig(os.path.join(output_path, 'LR_scheduler.png'))

optimizer = optim.Adam(model.parameters(), lr=LR[0], weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

# Loss functions    
loss_fn = nn.BCEWithLogitsLoss() 
dist_loss = nn.MSELoss()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Training ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def train():
    model.train()
    train_loss = 0
    train_acc = []
    n_batches = 0
    for train_batch in tqdm(train_loader, desc="Train Loop"):
        images = train_batch["image"].to(device, dtype=torch.float32)
        mask = train_batch["mask"].to(device, dtype=torch.float32)
        mask_edges = train_batch["mask_edges"].to(device, dtype=torch.float32)
        mask_dist_to_boundary = train_batch["mask_dist_to_boundary"].to(device, dtype=torch.float32)

        mask_pred, mask_edges_pred, mask_dist_to_boundary_pred = model(images)

        # Compute loss
        loss = loss_fn(mask_pred, mask) + loss_fn(mask_edges_pred, mask_edges) + 0.1*dist_loss(mask_dist_to_boundary_pred*mask, mask_dist_to_boundary*mask)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mask_pred[mask_pred > 0.5] = 1
        mask_pred[mask_pred <= 0.5] = 0
        train_acc.append(utils.iou(mask_pred.detach().cpu().numpy(), mask.cpu().numpy()).item())
        n_batches += 1

    train_loss /= n_batches
    train_acc = np.nanmean(train_acc) 

    return train_loss, train_acc

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Validation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def validation():
    model.eval()
    validation_loss = 0
    n_batches = 0
    validation_acc = [] 
    
    for val_batch in tqdm(val_loader, desc="Validation Loop"):
        images = val_batch["image"].to(device, dtype=torch.float32)
        mask = val_batch["mask"].to(device, dtype=torch.float32)
        mask_edges = val_batch["mask_edges"].to(device, dtype=torch.float32)
        mask_dist_to_boundary = val_batch["mask_dist_to_boundary"].to(device, dtype=torch.float32)

        mask_pred, mask_edges_pred, mask_dist_to_boundary_pred = model(images)

        # Compute loss and accuracy
        loss = loss_fn(mask_pred, mask) + loss_fn(mask_edges_pred, mask_edges) + 0.1*dist_loss(mask_dist_to_boundary_pred*mask, mask_dist_to_boundary*mask)
        validation_loss += loss.item()

        mask_pred[mask_pred > 0.5] = 1
        mask_pred[mask_pred <= 0.5] = 0
        validation_acc.append(utils.iou(mask_pred.detach().cpu().numpy(), mask.cpu().numpy()).item())
        n_batches += 1

    validation_loss /= n_batches
    scheduler.step(np.around(validation_loss,2))
    validation_acc = np.nanmean(validation_acc)

    return validation_loss, validation_acc

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Training ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
epoch_train_loss, epoch_train_acc = [], []
epoch_val_loss, epoch_val_acc = [], []

if args.verbose:
    print("Training started...")
best_val_loss = float('inf')
early_stopping_counter = 0
pbar = tqdm(range(args.epochs), disable=not(args.verbose), desc="Epoch Loop")
for epoch in pbar:
    if early_stopping_counter >= 30:
        break
    # Set learning rate
    #for param_group in optimizer.param_groups:
    #    param_group["lr"] = LR[epoch]
    utils.set_seed(epoch)
    avg_train_loss, avg_train_acc = train()
    avg_val_loss, avg_val_acc = validation()
    epoch_train_loss.append(avg_train_loss)
    epoch_train_acc.append(avg_train_acc)
    epoch_val_loss.append(avg_val_loss)
    epoch_val_acc.append(avg_val_acc)
    model_file = os.path.join(trained_models_path, 'model_best.pth')
    if avg_val_loss < best_val_loss:
        torch.save(model.state_dict(), model_file)
        best_val_loss = avg_val_loss
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
    pbar.set_postfix({'val_loss': avg_val_loss, 'best_val_loss': best_val_loss, 'early_stopping_counter': early_stopping_counter})

if args.verbose:
    print("Training completed!")
    print("Model saved to {}".format(model_file))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Plot results ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot training and validation loss
fig, ax = plt.subplots(1, 2, figsize=(12, 4), dpi=100)
ax[0].plot(epoch_train_loss, label='train', lw=2)
ax[0].plot(epoch_val_loss, label='val', lw=2)
ax[0].set_title('Loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].legend()
#ax[0].set_ylim([0, 0.1])
# remove right and top spines
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
ax[1].plot(epoch_train_acc, label='train', lw=2)
ax[1].plot(epoch_val_acc, label='val', lw=2)
ax[1].set_title('Accuracy')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy (%) - IoU')
ax[1].legend()
# remove right and top spines
ax[1].spines['right'].set_visible(False)    
ax[1].spines['top'].set_visible(False)
#ax[1].set_ylim([95, 100])
# Save figure
fig.savefig(os.path.join(output_path, 'loss_acc.png'))
if args.verbose:
    print("Loss and accuracy plots saved to {}".format(os.path.join(output_path, 'loss_acc.png')))

# References:

