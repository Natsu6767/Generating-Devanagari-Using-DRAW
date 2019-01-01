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
    'N' : 28,
    'dec_size': 256,
    'enc_size' :256,
    'epoch_num': 20,
    'learning_rate': 1e-3,
    'beta1': 0.5,
    'clip': 5.0}

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

avg_loss = 0
count = 0

for epoch in range(params['epoch_num']):
    for data, _ in train_loader:
        bs = data.size(0)
        data = data.view(bs, -1).to(device)
        optimizer.zero_grad()
        loss = model.loss(data)
        avg_loss += loss.cpu().data.numpy()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), params['clip'])
        optimizer.step()
        count += 1
        
        if count % 100 == 0:
            print("Epoch-{}; Count-{}; loss: {};".format(epoch, count, avg_loss / 100))

            if count % 3000 == 0:
                with torch.no_grad():
                    generate_image(count)

            avg_loss = 0

generate_image(count)