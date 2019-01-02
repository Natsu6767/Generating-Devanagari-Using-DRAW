import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

from torchvision import datasets, transforms
from draw_model import DRAWModel

def generate_image(count):
    x = model.generate(64)
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in x]
    anim = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    anim.save('draw_{}.gif'.format(count), dpi=100, writer='imagemagick')

params = {
    'T' : 10,
    'batch_size': 64,
    'A' : 28,
    'B': 28,
    'z_size' :10,
    'N' : 5,
    'dec_size': 256,
    'enc_size' :256,
    'epoch_num': 5,
    'learning_rate': 1e-3,
    'beta1': 0.5,
    'clip': 5.0,
    'save_epoch' : 2}

# Use GPU is available else use CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")

params['device'] = device


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data/', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor()])),
    batch_size=params['batch_size'], shuffle=False)

model = DRAWModel(params).to(device)
optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], betas=(params['beta1'], 0.999))

losses = []
iters = 0
avg_loss = 0
print("Starting Training Loop...")
print("-"*25)

for epoch in range(params['epoch_num']):
    for i, (data, _) in enumerate(train_loader, 0):
        bs = data.size(0)
        data = data.view(bs, -1).to(device)
        optimizer.zero_grad()
        loss = model.loss(data)
        loss_val = loss.cpu().data.numpy()
        avg_loss += loss_val
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), params['clip'])
        optimizer.step()
        iters += 1
        
        if iters % 100 == 0:
            print('[%d/%d][%d/%d]\tLoss: %.4f' % (epoch, params['epoch_num'], i, len(train_loader), avg_loss/100))
            avg_loss = 0

        losses.append(loss_val)

    with torch.no_grad():
        generate_image(epoch+1)

    if epoch % params['save_epoch'] == 0:
        torch.save({
            'model' : model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'params' : params
            }, 'checkpoint/model_epoch_{}'.format(epoch))

torch.save({
    'model' : model.state_dict(),
    'optimizer' : optimizer.state_dict(),
    'params' : params
    }, 'checkpoint/model_final'.format(epoch))

# Plot the training losses.
plt.figure(figsize=(10,5))
plt.title("Training Loss")
plt.plot(losses)
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.savefig("Loss_Curve")