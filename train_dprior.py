import argparse
import os
from dalle2_pytorch import DiffusionPrior
from embedding_reader import EmbeddingReader
from dalle2_pytorch import DiffusionPriorNetwork
import numpy as np
import math
from tqdm import tqdm
import torch
from torch import nn

import wandb
os.environ["WANDB_SILENT"] = "true"

import wandb
os.environ["WANDB_SILENT"] = "true"

def train(image_embed_dim,image_embed_url,text_embed_url,batch_size,train_percent,val_percent,device,learning_rate=0.01):
    # DiffusionPriorNetwork 
    prior_network = DiffusionPriorNetwork( dim = image_embed_dim, depth = 6, dim_head = 64, heads = 8).to(device)
    
    # DiffusionPrior with text embeddings and image embeddings pre-computed
    diffusion_prior = DiffusionPrior( net = prior_network, clip = None, image_embed_dim = image_embed_dim, 
                                     timesteps = 100, cond_drop_prob = 0.2, 
                                     condition_on_text_encodings = False).to(device)
    # Get image and text embeddings from the servers
    print("==============Downloading embeddings - image and text====================")
    ei = EmbeddingReader(embeddings_folder=image_embed_url, file_format="npy")
    et = EmbeddingReader(embeddings_folder=text_embed_url, file_format="npy")

    ### Training code
    optimizer = torch.optim.SGD(diffusion_prior.parameters(), lr = 0.01)
    epochs = 5
    min_valid_loss = np.inf
    for e in range(epochs):
        train_loss = 0.0
        print("Training loop - epoch number ",e)
        train_set_size = int(train_percent*ei.count)
        for embi,embt in zip(ei(batch_size=batch_size, start=0, end=train_set_size),et(batch_size=batch_size, start=0, end=train_set_size)):
            embi = list(embi)
            embt = list(embt)
            print(embi[0].shape,embt[0].shape)
            if torch.cuda.is_available():
                embi[0] = torch.tensor(embi[0]).to(device)
                embt[0] = torch.tensor(embt[0]).to(device)
            optimizer.zero_grad()
            # taking 80% for training - 20% for validation
            loss = diffusion_prior(text_embed = embt[0],image_embed = embi[0])
            loss.backward()
            # Log to wandb
            wandb.log({"Training Loss": loss})
            optimizer.step()
            print("Training loss = ",loss.item())
            train_loss+=loss.item()

        print("Validation loop - epoch number ",e)
        valid_loss = 0.0
        val_set_size = int(val_percent*ei.count)
        for embi,embt in zip(ei(batch_size=batch_size, start=ei.count-train_set_size, end=ei.count),et(batch_size=batch_size, start=ei.count-train_set_size, end=et.count)):
            embi = list(embi)
            embt = list(embt)
            if torch.cuda.is_available():
                embi[0] = torch.tensor(embi[0])
                embt[0] = torch.tensor(embt[0])    
            loss = diffusion_prior(text_embed = embt[0],image_embed = embi[0])
            
            # Log to wandb
            wandb.log({"Validation Loss ": loss})
            valid_loss+=loss.item()

            # Saving State Dict
        torch.save(model.state_dict(), 'saved_model.pth')

def main():
    parser = argparse.ArgumentParser()
    # Logging
    parser.add_argument("--wandb-entity", type=str, default="laion")
    parser.add_argument("--wandb-project", type=str, default="diffusion-prior")
    # URLs for embeddings 
    parser.add_argument("--image-embed-url", type=str, default="https://mystic.the-eye.eu/public/AI/cah/laion5b/embeddings/laion2B-en/img_emb/")
    parser.add_argument("--text-embed-url", type=str, default="s3://laion-us-east-1/embeddings/vit-l-14/laion2B-en/text_emb/")
    # Hyperparameters
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=10**6)
    # Image embed dimension
    parser.add_argument("--image-embed-dim", type=int, default=768)
    # Train-val split
    parser.add_argument("--train-percent", type=float, default=0.8)
    parser.add_argument("--val-percent", type=float, default=0.2)


    args = parser.parse_args()
    print("Setting up wandb logging... Please wait...")
    wandb.init(
      entity=args.wandb_entity,
      project=args.wandb_project,
      name=f"laion-dprior",
      config={
      "learning_rate": args.learning_rate,
      "architecture": "DiffusionPrior",
      "dataset": "LAION-5B",
      "epochs": 10,
      })
    print("wandb logging setup done!")
       # Obtain the utilized device.
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        has_cuda = True
    else:
        device = torch.device("cpu")
        has_cuda = False
      # Training loop
    train(args.image_embed_dim,args.image_embed_url,args.text_embed_url,args.batch_size,args.train_percent,args.val_percent,device,args.learning_rate)

if __name__ == "__main__":
  main()


