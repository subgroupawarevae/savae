import torch
import torch.optim as optim
import pandas as pd
from modules import (
    auxiliary as aux,
    beam_search_implementation_tensor as beam_search_T,
    autoencoder_models as models
)
from numpy import sqrt
import numpy as np
import os
from torchvision.utils import save_image
import re
import argparse
import sys
import os
from tqdm import tqdm


torch.manual_seed(0)

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Setting the model we are using
    model = models.DFCVAE(latent_size=tconst.LATENT_SIZE).to(device)

    # Load the model we are going analyse
    model.load_last_model(tconst.MODEL_PATH)

    # Setting the mode (hopefully this persists through function calls?)
    model.eval()

    aux.create_folder(tconst.SGD_VIS_PATH)
    
    # Read in latents, convert to tensor and remove the index column that was saved
    print ("Reading the latents...")
    latents_df = pd.read_csv(tconst.CSV_PATH)
    latents_tensor = torch.Tensor(latents_df.values)[:, 1:]
    latents_tensor = latents_tensor.to(device)
    print ("Binning the latents...")
    # Bin the latents and then run beam search to find the best subgroups
    binned_latents_tensor = aux.bin_dataset(latents_tensor)

    features = latents_tensor[:,:-1]
    labels = latents_tensor[:,-1]
    
    features_mean = features.mean(dim=0)
    image = torch.squeeze(aux.decode_latent(features_mean, model))
    save_image(image, os.path.join(tconst.SGD_VIS_PATH, 'avg_whole_population.png'))


    df=pd.read_csv(tconst.SUBGROUPS_PATH)
    avg_images = []
    for counter, sg in enumerate(df["subgroup"][2:]):
        print ("evaluating sg:")
        print (sg)
        pairs = re.findall("\d+==-?\d+.\d+", sg)
        res = []
        for pair in pairs:
            tokens = re.findall("(?:-?\d+.\d+|\d+)", pair)
            res.append(tokens)
        sg_idx = []
        for i in tqdm(range(len(features))):
            conj=True
            for pair in res:
                if features[i, int(pair[0])]!=float(pair[1]):
                    conj = False
            if conj:
                sg_idx.append(i)
        print ("matching ids:")
        print (sg_idx)
        if len(sg_idx)==0:
            continue
        average_latents = torch.mean(features[sg_idx] , dim=0)
        concat_imgs = []
        concat_imgs_org = []
        #for idx in sg_idx:
            #image = torch.squeeze(aux.decode_latent(features[idx], model))
            #org_image = aux.get_ims(['{:06d}.jpg'.format(idx+1)], tconst.IMAGES_PATH)[0]
            #concat_imgs.append(image)
            #concat_imgs_org.append(org_image)
        image = torch.squeeze(aux.decode_latent(average_latents, model))
        avg_images.append(image)
        save_image(image, os.path.join(tconst.SGD_VIS_PATH, 'sgd_avg_{}'.format(counter) + '.png'))
        #save_image(concat_imgs, os.path.join(tconst.SGD_VIS_PATH, 'sgd_cluster_{}'.format(counter) + '.png'))
        #save_image(concat_imgs_org, os.path.join(tconst.SGD_VIS_PATH, 'sgd_cluster_org_{}'.format(counter) + '.png'))

    save_image(
    avg_images,
    os.path.join(tconst.TRAVERSAL_PATH,'avg_subgroups.png'),
    padding=5,
    nrow=5) 

def parse_args():
    parser = argparse.ArgumentParser(sys.argv[1:])
    parser.add_argument("--config_id", type=int, default=0)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    import_command = "from configs import config_{} as tconst".format(args.config_id)
    exec (import_command)
    main()





