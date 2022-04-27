# This is a first cut - training on a single GPU. 
# 1. Drop this file in the DALLE2-pytorch-main/dalle2_pytorch/ directory - I've changed the import paths to refer to local directory. 
# 2. Loss - In the LucidRains code - he defaults to "l1" - I've let it be as is. 
# 3. As mentioned in Section 2.2 of the paper - they're better off training on a L2 loss - but the code takes "l1" - I've let it be for now. 
# 4. Line 49 was changed for testing purposes, to see if the forward() of the DiffusionPrior was getting called ok, looks good. 
# Change it to line 50, for the full run, with the entire embeddings set. 

from dalle2_pytorch import DiffusionPrior
from embedding_reader import EmbeddingReader
from dalle2_pytorch import DiffusionPriorNetwork
import numpy as np
import math
from tqdm import tqdm
from inspect import isfunction
from functools import partial
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from einops_exts import rearrange_many, repeat_many, check_shape
from einops_exts.torch import EinopsToAndFrom

from kornia.filters import gaussian_blur2d

from tokenizer import tokenizer # all point to local directory here. 
from vqgan_vae import NullVQGanVAE, VQGanVAE
from attention import QueryAttnUpsample

# DiffusionPriorNetwork - dim needs to be 768, same as image_embed_dim
prior_network = DiffusionPriorNetwork( dim = 768, depth = 6, dim_head = 64, heads = 8).cuda()
# DiffusionPrior with text embeddings and image embeddings pre-computed
diffusion_prior = DiffusionPrior( net = prior_network, clip = None, image_embed_dim = 768, timesteps = 100, cond_drop_prob = 0.2, condition_on_text_encodings = False  ).cuda()
# Get image and text embeddings from the servers
ei = EmbeddingReader(embeddings_folder="https://mystic.the-eye.eu/public/AI/cah/laion5b/embeddings/laion2B-en/img_emb/", file_format="npy")
et = EmbeddingReader(embeddings_folder="s3://laion-us-east-1/embeddings/vit-l-14/laion2B-en/text_emb/", file_format="npy")

### Training code
optimizer = torch.optim.SGD(diffusion_prior.parameters(), lr = 0.01)
epochs = 5
min_valid_loss = np.inf
for e in range(epochs):
    train_loss = 0.0
    print("Training loop - epoch number ",e)
    # for testing purposes
    for embi,embt in zip(ei(batch_size=10 ** 3, start=0, end=10000),et(batch_size=10 ** 3, start=0, end=10000)):
#    for embi,embt in zip(ei(batch_size=10 ** 3, start=0, end=ei.count),et(batch_size=10 ** 3, start=0, end=et.count)):
        embi = list(embi)
        embt = list(embt)
        print(embi[0].shape,embt[0].shape)
        if torch.cuda.is_available():
            embi[0] = torch.tensor(embi[0][:(int(0.8*embi[0].shape[0]))]).cuda()
            embt[0] = torch.tensor(embt[0][:(int(0.8*embt[0].shape[0]))]).cuda()
        optimizer.zero_grad()
        # taking 80% for training - 20% for validation
        loss = diffusion_prior(text_embed = embt[0],image_embed = embi[0])
        loss.backward()
        optimizer.step()
        print("Training loss = ",loss.item())
        train_loss+=loss.item()

    print("Validation loop - epoch number ",e)
    valid_loss = 0.0
    for embi,embt in zip(ei(batch_size=10 ** 6, start=0, end=ei.count),et(batch_size=10 ** 6, start=0, end=et.count)):
        embi = list(embi)
        embt = list(embt)
        if torch.cuda.is_available():
            embi[0] = torch.tensor(embi[0][(int(0.8*embi[0].shape[0])):]).cuda()
            embt[0] = torch.tensor(embt[0][(int(0.8*embt[0].shape[0])):]).cuda()
        loss = diffusion_prior(text_embed = embt[0],image_embed = embi[0])
        print("Validation loss = ",loss.item())
        valid_loss+=loss.item()
        
        print(f'Epoch {e+1} \t\t Training Loss: { train_loss / len(trainloader)} \t\t Validation Loss: { valid_loss / len(validloader)}')

    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f\
        }--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss

        # Saving State Dict
        torch.save(model.state_dict(), 'saved_model.pth')


