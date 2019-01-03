import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torchvision.utils as vutils
import time

from draw_model import DRAWModel

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', required=True, help='Checkpoint to load path from')
parser.add_argument('-num_output', default=36, help='Number of generated outputs')
parser.add_argument('-t', default=None, help='Number of glimpses.')
args = parser.parse_args()

# Load the checkpoint file.
state_dict = torch.load(args.load_path)

# Set the device to run on: GPU or CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
# Get the 'params' dictionary from the loaded state_dict.
params = state_dict['params']

# Set the number of glimpses.
# Best to just use the same value which was used for training.
params['T'] = int(args.t) if(args.t) else params['T']

# Load the model
model = DRAWModel(params).to(device)
# Load the trained parameters.
model.load_state_dict(state_dict['model'])
print('\n')
print(model)

start_time = time.time()
print('*'*25)
print("Generating Image...")
# Generate images.
with torch.no_grad():
    x = model.generate(int(args.num_output))

time_elapsed = time.time() - start_time
print('\nDONE!')
print('Time taken to generate image: %.2fs' % (time_elapsed))

print('\nSaving generated image...')
fig = plt.figure(figsize=(int(np.sqrt(int(args.num_output)))*2, int(np.sqrt(int(args.num_output)))*2))
plt.axis("off")
plt.imshow(np.transpose(vutils.make_grid(
    x[-1], nrow=int(np.sqrt(int(args.num_output))), padding=1, normalize=True, pad_value=1).cpu(), (1, 2, 0)))
plt.savefig("Generated_Image")
plt.close('all')

# Create animation for the generation.
fig = plt.figure(figsize=(int(np.sqrt(int(args.num_output)))*2, int(np.sqrt(int(args.num_output)))*2))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in x]
anim = animation.ArtistAnimation(fig, ims, interval=200, repeat_delay=2000, blit=True)
anim.save('draw_generate.gif', dpi=100, writer='imagemagick')
print('DONE!')
print('-'*50)
plt.show()