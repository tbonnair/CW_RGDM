#%%
import json
import numpy as np
import torch
from tqdm import tqdm
from ml_collections import ConfigDict
import matplotlib as mpl

from RGDM_diffusion import RGDiffusionSampler
from model import UNet
from dataset import get_dataset

from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

plt.rcParams['figure.figsize'] = [6,6]
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight']= 'normal'
# plt.style.use('seaborn-whitegrid')
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['savefig.dpi'] = 300       # Number of dpi of saved figures
mpl.rcParams['font.size'] = 22
mpl.rcParams['axes.formatter.limits']=(-6, 6)
mpl.rcParams['axes.formatter.use_mathtext']=True

#mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.top'] = True

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img[0].detach()
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    return fig

# %%

in_chan = 1     # To be put in flags
log_path = 'logs/RGDM_CW32_T500/'
ckpt_path = '{:s}/ckpt/ckpt_24.pt'.format(log_path)
num_images = 100

# read logdir/saved_config.json
config_path = log_path + '/saved_config.json'
with open(config_path, 'r') as f:
    config_data = json.load(f)    
FLAGS = ConfigDict(config_data)

# device
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# %% Load training data
dataset = get_dataset(FLAGS.dataset_key)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=len(dataset), shuffle=True,
    num_workers=FLAGS.num_workers, drop_last=True)

images, _ = next(iter(dataloader))
images = images*2           # Detransform the data

# %% Load model and sampler

# Load model
model = UNet(
    T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
    num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout, in_chan=in_chan)

# Load checkpoint (for models trained with dataParallel)
state_dict = torch.load(ckpt_path, map_location='cpu')['ema_model']
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('module.'):
        k = k[7:] # Remove 'module.'
    new_state_dict[k] = v
model.load_state_dict(new_state_dict)
model.eval()

# Define sampler
sampler = RGDiffusionSampler(
    model=model, T=FLAGS.T, L=FLAGS.img_size,
    beta_1=FLAGS.beta_1, beta_2=FLAGS.beta_T,
    m0=FLAGS.m0, mL=FLAGS.mL, k_cutoff=FLAGS.k_cutoff, in_chan=in_chan).to(device)

# %% Example sampling of num_images data

batch_size = 100
n_iter = num_images // batch_size
with torch.no_grad():
    x_gen = torch.zeros(num_images, in_chan, FLAGS.img_size, FLAGS.img_size)
    for i in tqdm(range(n_iter)):
        x0 = sampler(batch_size, device).detach().cpu()
        x0 *= 2             # Detransform data
        x_gen[i*batch_size:(i+1)*batch_size] = x0

# %% Plot some generated images

# Show and save a batch of input images
grid = make_grid(images[0:16], 4, padding=1)
fig = show(grid)

# Show and save a batch of generated images
grid = make_grid(torch.clip(x_gen, images.min(), images.max())[0:16], 4, padding=1)
fig = show(grid)
