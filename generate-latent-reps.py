import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from modules import (
    autoencoder_models as models,
    auxiliary as aux,
)
import argparse
import sys
import os

torch.manual_seed(0)




def main():
    aux.create_folder(os.path.dirname(tconst.CSV_PATH))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Setting the model we are using
    model = models.DFCVAE(latent_size=tconst.LATENT_SIZE).to(device)
    # Load the model we are going analyse
    start_epoch = model.load_last_model(tconst.MODEL_PATH)
    print('Results saved at:', tconst.CSV_PATH)
  
    all_image_ids = ['{:06d}.jpg'.format(x) for x in range(1, tconst.NO_IMAGES_COMPUTE_OVER + 1)]
    upper_limit_no_images = len(all_image_ids)
    
    # Keeps a persistent tally as we traverse all images
    all_image_counter = 0

    # Iterating over batches of images
    for i in range(0, tconst.NO_IMAGES_COMPUTE_OVER, tconst.INCREMENT):
        
        # On that last batch we have to make sure we dont overextend
        if(i + tconst.INCREMENT) > len(all_image_ids):
            image_ids_for_batch = all_image_ids[i:upper_limit_no_images + 1]
        else:
            image_ids_for_batch = all_image_ids[i:i + tconst.INCREMENT]


        print('From: ', image_ids_for_batch[0], ' - ', image_ids_for_batch[-1])

        images_tensor = aux.get_images_by_id(image_ids_for_batch, tconst.IMAGES_PATH)
        images_tensor = images_tensor.to(device)
        latents_tensor = aux.get_latents(model, images_tensor, image_ids_for_batch, device, tconst.ATTR_CSV_PATH, tconst.CLASS)
        table_for_csv = pd.DataFrame(latents_tensor).astype('float')


        if i == 0:
            table_for_csv.to_csv(tconst.CSV_PATH)
        else:
            table_for_csv.to_csv(tconst.CSV_PATH, mode = 'a', header = False)


    print('Results Saved at: ', tconst.CSV_PATH)
        

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
