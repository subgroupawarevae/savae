import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from modules import (
    autoencoder_models as models,
    auxiliary as aux,
)
import os
import argparse
import sys
import os
import re
from torchvision.utils import save_image

torch.manual_seed(0)



def generate_latent_reps(model, NO_TESTING_IMAGES, LATENT_SIZE, device):

    images_tensor, image_ids = aux.get_images(NO_TESTING_IMAGES, tconst.IMAGES_PATH)
    images_tensor = images_tensor.to(device)
    latents_tensor = aux.get_latents(model, images_tensor, image_ids, device, tconst.ATTR_CSV_PATH, tconst.CLASS)
    latents_tensor = latents_tensor [:, :-1]
    latents_df = pd.DataFrame(latents_tensor, columns=[i for i in range(0, LATENT_SIZE)]).astype("float")
    #print(latents_df)

    return latents_df

def traverse_latent_reps(model, latents_df, LATENT_SIZE, device, latents_to_visualize):

    # Taking the first row
    first_entry = latents_df.iloc[0]
    first_entry = first_entry.to_frame()
    first_entry = first_entry.transpose()
    
    # Converting to a tensor that has just one dimension
    first_entry_tensor = torch.squeeze(torch.tensor((first_entry.values.astype(np.float32))))

    #print('The latent vector', first_entry_tensor)
    #print(' ')

    list_of_lists = []

    for i in latents_to_visualize:
 
        latent_max = latents_df[i].max()
        latent_min = latents_df[i].min()
        increment_to_extremes = (latent_max - latent_min)/5
               
        latent_max += 2*increment_to_extremes
        latent_min -= 2*increment_to_extremes 

        # Must redefine as this tensor has values changed each iteration
        first_entry_tensor = torch.squeeze(torch.tensor((first_entry.values.astype(np.float32))))

        list_of_reconstructions = []

        values_to_examine = list(np.linspace(latent_min, latent_max, num=10))

        for value in values_to_examine:
            first_entry_tensor[i] = value
            first_entry_tensor = first_entry_tensor.to(device)
            image = torch.squeeze(aux.decode_latent(first_entry_tensor, model))
            list_of_reconstructions.append(image)
        
        list_of_lists.append(list_of_reconstructions)

    presentable_save(list_of_lists)

def presentable_save(list_of_lists):
    print ("size of images")
    print (len(list_of_lists))
    # Now saving all the images into a neat grid 5 images per row
    INCREMENT = 15
    for start in range(0, len(list_of_lists), INCREMENT):
 
        concatenated_reconstructions = []

        # Goes till start + increment - 1  
        for index in range(start, min(start + INCREMENT, len(list_of_lists))):
            concatenated_reconstructions += list_of_lists[index] 
        
        # Have increment - 1 in the name because the range does not cover the last value
        save_image(
            concatenated_reconstructions, 
            os.path.join(tconst.TRAVERSAL_PATH,'latents_' + str(start) + '-' + str(start + (INCREMENT-1)) + '.png'),
            padding=0, 
            nrow=len(list_of_lists[0])
            )
        print (os.path.join(tconst.TRAVERSAL_PATH,'latents_' + str(start) + '-' + str(start + (INCREMENT-1)) + '.png'))

def get_recovered_latents():
    df=pd.read_csv(tconst.SUBGROUPS_PATH)
    res = set()
    for counter, sg in enumerate(df["subgroup"][2:]):
        print ("evaluating sg:")
        print (sg)
        pairs = re.findall("\d+==-?\d+.\d+", sg)
        for pair in pairs:
            tokens = re.findall("(?:-?\d+.\d+|\d+)", pair)
            res.add(int(tokens[0]))
    return res


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Setting the model we are using
    model = models.DFCVAE(latent_size=tconst.LATENT_SIZE).to(device)

    # Load the model we are going analyse
    model.load_last_model(tconst.MODEL_PATH)

    # Setting the mode (hopefully this persists through function calls?)
    model.eval()

    #print('latent size expected of the loaded model:', tconst.LATENT_SIZE)

    latents_df = generate_latent_reps(model, tconst.NO_TESTING_IMAGES, tconst.LATENT_SIZE, device)

    aux.create_folder(tconst.TRAVERSAL_PATH)

    latents_df.to_csv(os.path.join(tconst.TRAVERSAL_PATH, 'latents.csv'))
    latents_to_visualize = get_recovered_latents()
    latents_to_visualize = list(latents_to_visualize)
    latents_to_visualize.sort()
    print (latents_to_visualize)
    traverse_latent_reps(model, latents_df, tconst.LATENT_SIZE, device, latents_to_visualize)



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


