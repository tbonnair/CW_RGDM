# %%
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch_dct as dct
from tqdm import tqdm
from ml_collections import ConfigDict
from scipy.integrate import cumulative_trapezoid

import RGDM_diffusion as RG_Diff
from RGDM_diffusion import RGDiffusionTrainer, RGDiffusionSampler
from model import UNet
from dataset import get_dataset

import matplotlib.pyplot as plt


# %% Define a model to load

in_chan = 1             # Todo: put it in flags
log_path = 'logs/RGDM_PHI4_L128_T500/'
ckpt_path = '{:s}ckpt/ckpt_7.pt'.format(log_path)      # 4 is the last model

# read logdir/saved_config.json
config_path = log_path + 'saved_config.json' #'RGDM_configs/RGDM_PHI4_L128_T500.json' #info_args.config
with open(config_path, 'r') as f:
    config_data = json.load(f)
FLAGS = ConfigDict(config_data)

# device
if torch.cuda.is_available():
    device = torch.device('cuda:1')
else:
    device = torch.device('cpu')

# %% Load model

model = UNet(
    T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
    num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout, in_chan=in_chan)

# For models trained with dataParallel
state_dict = torch.load(ckpt_path, map_location='cpu')['ema_model']
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('module.'):
        k = k[7:] # Remove 'module.'
    new_state_dict[k] = v
model.load_state_dict(new_state_dict)

# I had to change the following line because I'm using a CPU and not a GPU
# model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=True)['ema_model'])  # ema, not net_model
model.eval()

# Also need trainer to noise the data
trainer = RGDiffusionTrainer(
    model, FLAGS.T, FLAGS.img_size,
    FLAGS.beta_1, FLAGS.beta_T,
    FLAGS.m0, FLAGS.mL, FLAGS.k_cutoff).to(device)
# %% Evaluate the score

time_step = 0     # Diffusion time step in [0, T]
phi_const = -1.0     # Homogenous phi value of the field
phi = torch.ones(1 , 1, 32, 32).to(device) * phi_const
t = torch.ones([phi.shape[0], ], dtype=torch.long).to(device) * time_step


# Get the value of sqrt_betas_bar (which corresponds to sq_bb(t)) for this timestep
sqrt_betas_bar = trainer.sqrt_betas_bar  # This is the buffer holding sqrt(betas) over time
sqrt_alphas_bar = trainer.sqrt_alphas_bar

sq_bb_t = torch.index_select(sqrt_betas_bar, dim=0, index=t)  # Extract value for time_step t





# the score is given by evaluating the model on phi noised at time t
#phi_t, res_noise, cutoff = trainer.make_xt(phi, t)
#noise_t = model(phi_t, t)



def compute_score(noise, sq_bb):
    """
    Compute the score as:
        score = -IDCT2D( DCT2D(noise) / b_tk )                  in the non rescaled case
        score = -IDCT2D( DCT2D(noise) / b_tk*sqrt(K_t(k)) )     in the rescaled case
    where:
      - noise: predicted noise epsilon_theta(x,t)
    """
    # Transform predicted noise to the DCT domain.
    noise_dct = dct.dct_2d(noise)
    # Divide elementwise by sq_bb.
    noise_dct_transformed = - noise_dct / sq_bb
    # Transform back.
    score = dct.idct_2d(noise_dct_transformed)
    return score


# I compute scores for some values of phi (constant) and at different times

tt = [0, 50, 100, 150, 200, 250, 300, 400]

m = [-1.+(2*i/50) for i in range(51)]
V_s = []


with torch.no_grad():
    for ts in tqdm(tt):
        dV = []
        time = torch.tensor([ts], dtype=torch.long).to(device)

        sq_bb_t = torch.index_select(sqrt_betas_bar, dim=0, index=time) # I extract b_kt
        sq_K_t = torch.index_select(sqrt_alphas_bar, dim=0, index=time) # and K_t(k)
        M = torch.sqrt(sq_K_t) * sq_bb_t

        for x in m:
            phi = torch.ones(1, 1, 128, 128).to(device) * x # create constant field

            phi_t, res_noise, cutoff = trainer.make_xt(phi, time)
            noise_t = model(phi_t, time)
            
            # Test TB 280325: Add RG projection
            noise_t = dct.dct_2d(noise_t)
            noise_t = dct.idct_2d(cutoff * noise_t)

            # score in the non-rescaled version
            # s = compute_score(noise_t, sq_bb_t)

            # rescaled version
            s = compute_score(noise_t, M)

            #np.save(f'score/score_rescaled_t{ts}.npy', s.detach().numpy())

            # dV/dphi = - sum_i(score_i)
            dV.append(-s.mean().item())


        # I have the derivative dV, let's find V
        V = cumulative_trapezoid(dV, m, initial=0)
        V_s.append(V)



# %% plot V(\phi)

plt.figure(figsize=(6, 6))
for i in range(len(tt)):
    plt.plot(m, V_s[i], label='t = ' +str(tt[i]))
plt.xlabel(r'$\phi$')
plt.ylabel(r'$V(\phi)$')
plt.title(r'Plot of $V(\phi)$')
plt.legend(frameon=False, fontsize=10)
plt.grid()
plt.savefig(log_path + "plot_V_rescaled_ckpt7.png", bbox_inches='tight')
plt.show()




# From there, we need to go from noise to score
# Usually, it is just - noise_t / std_t (Tweedie formula)
# But we should be careful because here the noise is scale dependant
