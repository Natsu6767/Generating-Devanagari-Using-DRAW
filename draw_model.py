import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils

class DRAWModel(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.T = params['T']
        self.A = params['A']
        self.B = params['B']
        self.z_size = params['z_size']
        self.N = params['N']
        self.enc_size = params['enc_size']
        self.dec_size = params['dec_size']

        self.cs = [0] * self.T
        
        self.logsigmas = [0] * self.T
        self.sigmas = [0] * self.T
        self.mus = [0] * self.T

        self.encoder = nn.LSTMCell(2*self.N*self.N + self.dec_size, self.enc_size)

        self.fc_mu = nn.Linear(self.enc_size, self.z_size)
        self.fc_sigma = nn.Linear(self.enc_size, self.z_size)

        self.decoder = nn.LSTMCell(self.z_size, self.dec_size)

        self.fc_write = nn.Linear(self.dec_size, self.N*self.N)

    def forward(self, x):
        self.batch_size = x.size(0)

        h_enc_prev = torch.zeros(self.batch_size, self.enc_size, require_grads=True)
        h_dec_prev = torch.zeros(self.batch_size, self.dec_size, require_grads=True)

        enc_state = torch.zeros(self.batch_size, self.enc_size, require_grads=True)
        dec_state = torch.zeros(self.batch_size, self.dec_size, require_grads=True)

        for t in range(T):
            c_prev = torch.zeros(self.batch_size, self.B*self.A, require_grads=True) if t == 0 else self.cs[t-1]
            x_hat = x - F.sigmoid(c_prev)

            r_t = self.read(x, x_hat, h_dec_prev)

            h_enc, enc_state = self.encoder(torch.cat((r_t, h_dec_prev), dim=1), (h_enc_prev, enc_state))

            z, self.mus[t], self.logsigmas[t], self.sigmas[t] = self.sampleQ(h_enc)

            h_dec, dec_state = self.decoder(z, (h_dec_prev, dec_state))

            self.cs[t] = c_prev + self.write(h_dec)

            h_enc_prev = h_enc
            h_dec_prev = h_dec

    def read(self, x, x_hat, h_dec_prev):
        # No attention
        return torch.cat((x, x_hat), dim=1)

    def write(self, h_dec):
        # No attention
        return self.write(h_dec)

    def sampleQ(self, h_enc):
        e = torch.randn(self.batch_size, self.z_size)

        mu = self.fc_mu(h_enc)
        log_sigma = self.fc_sigma(h_enc)

        sigma = torch.exp(log_sigma)
        z = mu + e * sigma

        return z, mu, log_sigma, sigma

    def loss(self, x):
        self.forward(x)

        criterion = nn.BCELoss()
        x_recon = F.sigmoid(self.cs[-1])
        Lx = criterion(x_recon, x)

        Lz = 0

        for t in range(self.T):
            mu_2 = self.mus[t] * self.mus[t]
            sigma_2 = self.sigmas[t] * self.sigmas[t]
            logsigma = self.logsigmas[t]

            kl_loss = 0.5*torch.sum(mu_2 + sigma_2 - 2*logsigma, 1) - 0.5*self.T
            Lz += kl_loss

        Lz = torch.mean(Lz)
        net_loss = Lx + Lz

        return net_loss

    def generate(self, num_output):
        h_dec_prev = torch.zeros(self.batch_size, self.dec_size)
        dec_state = torch.zeros(self.batch_size, self.dec_size)

        for t in range(self.T):
            c_prev = torch.zeros(self.batch_size, self.B*self.A) if t == 0 else self.cs[t-1]
            z = torch.randn(self.batch_size, self.z_size)
            h_dec, dec_state = torch.decoder(z, (h_dec_prev, dec_state))
            self.cs[t] = c_prev + self.write(h_dec)
            h_dec_prev = h_dec

        imgs = []
        for img in self.cs:
            imgs.append(vutils.make_grid(img, nrow=int(np.sqrt(int(num_output))), padding=2, normalize=True))

        return imgs