import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import time
import torchvision.utils as vutils


from torchvision import datasets, transforms
from draw_model import DRAWModel
from dataloader import get_data

def generate_image(count):
    x = model.generate(36)
    fig = plt.figure(figsize=(12, 12))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in x]
    anim = animation.ArtistAnimation(fig, ims, interval=200, repeat_delay=1000, blit=True)
    anim.save('draw_{}.gif'.format(count), dpi=100, writer='imagemagick')
    plt.close('all')

params = {
    'T' : 64,
    'batch_size': 64,
    'A' : 32,
    'B': 32,
    'z_size' :100,
    'read_N' : 2,
    'write_N' : 5,
    'dec_size': 256,
    'enc_size' :256,
    'epoch_num': 10,
    'learning_rate': 2e-4,
    'beta1': 0.5,
    'clip': 5.0,
    'save_epoch' : 5,
    'channel' : None}

# Use GPU is available else use CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")

params['device'] = device


train_loader = get_data(params)
"""
train_loader = torch.utils.data.DataLoader(
    datasets.SVHN('data/', split='train', download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()])),
    batch_size=params['batch_size'], shuffle=True)
"""
params['channel'] = 1

# Plot the training images.
sample_batch = next(iter(train_loader))
plt.figure(figsize=(6, 6))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(
    sample_batch[0].to(device)[ : 36], nrow=6, padding=2, normalize=True).cpu(), (1, 2, 0)))
plt.savefig("Training_Data")

model = DRAWModel(params).to(device)
optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], betas=(params['beta1'], 0.999))

losses = []
iters = 0
avg_loss = 0
print("-"*25)
print("Starting Training Loop...")
print("-"*25)

start_time = time.time()

for epoch in range(params['epoch_num']):
    epoch_start_time = time.time()
    for i, (data, _ ) in enumerate(train_loader, 0):
        bs = data.size(0)
        data = data.view(bs, -1).to(device)
        optimizer.zero_grad()
        loss = model.loss(data)
        loss_val = loss.cpu().data.numpy()
        avg_loss += loss_val
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), params['clip'])
        optimizer.step()

        # Check progress of training.
        if i != 0 and i%100 == 0:
            print('[%d/%d][%d/%d]\tLoss: %.4f'
                  % (epoch, params['epoch_num'], i, len(train_loader), avg_loss / 100))

            avg_loss = 0
        
        losses.append(loss_val)
        iters += 1

    avg_loss = 0
    epoch_time = time.time() - epoch_start_time
    print("Time Taken for Epoch %d: %.2fm" %(epoch, epoch_time / 60))
    # Save checkpoint and generate test output.
    if epoch % params['save_epoch'] == 0:
        torch.save({
            'model' : model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'params' : params
            }, 'checkpoint/model_epoch_{}'.format(epoch))
        
        with torch.no_grad():
            generate_image(epoch)

training_time = time.time() - start_time
print("-"*50)
print('Training finished!\nTotal Time for Training: %.2fm' %(training_time / 60))
print("-"*50)
torch.save({
    'model' : model.state_dict(),
    'optimizer' : optimizer.state_dict(),
    'params' : params
    }, 'checkpoint/model_final'.format(epoch))

with torch.no_grad():
    generate_image(params['epoch_num'])

# Plot the training losses.
plt.figure(figsize=(10,5))
plt.title("Training Loss")
plt.plot(losses)
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.savefig("Loss_Curve")