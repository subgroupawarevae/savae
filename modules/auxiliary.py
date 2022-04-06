import csv
from PIL import Image
from numpy import append
import torch
from torchvision import transforms
from torchvision.transforms.functional import crop
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import torch.nn.functional as F
import os
from modules import training_constants as tconst
import pandas as pd
import numpy as np

#IMAGE_PATH = '../data/img_align_celeba/img_align_celeba/'
#CLASS = 'attractive' 
NO_COLUMNS = 50



############################################ Getting the tensor of images ######################################



# Define a function where: input is the image id and the attribute
# output the attribute value

im_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
])

class ImageDiskLoader(torch.utils.data.Dataset):

    def __init__(self, im_ids, images_path):
        self.transform = im_transform
        self.im_ids = im_ids
        self.images_path = images_path

    def __len__(self):
        return len(self.im_ids)

    def __getitem__(self, idx):
        im_path = os.path.join(self.images_path, self.im_ids[idx])
        im = Image.open(im_path)
        if "celeb" in self.images_path:
            im = crop(im, 30, 0, 178, 178) #TODO
        data = self.transform(im)
        id = self.im_ids[idx]

        return data, id

# returns pytorch tensor of images, defined by the list of image names put in
def get_ims(im_ids, images_path):
    ims = []
    for im_id in im_ids:
        im_path = os.path.join(images_path, im_id)
        im = Image.open(im_path)
        if "celeb" in images_path:
            im = crop(im, 30, 0, 178, 178)  #TODO
        ims.append(im_transform(im))
    return ims


# This function loads in the list of attribute values and the id of the image for each image
def get_attributes(attr_csv_path):

    id_attr = {}
    with open(attr_csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        attr = next(reader)[1:]
        attributes = {descrb.lower(): idx for idx, descrb in enumerate(attr)}

        for row in reader:
            idx = row[0]
            attr_arr = [int(i) for i in row[1:]]
            id_attr[idx] = attr_arr

    return attributes, id_attr


def get_images(no_images, images_path):

    image_ids = ['{:06d}.jpg'.format(x) for x in range(1, no_images + 1)]
    print('num train_images:', len(image_ids))
    images = get_ims(image_ids, images_path)
    shape = (len(images), 3, 64, 64)
    images_tensor = torch.zeros(shape)
    for i in range(0, len(images)):
        images_tensor[i] = images[i]

    return images_tensor, image_ids
    

def get_images_by_id(image_ids, images_path):

    images = get_ims(image_ids, images_path)
    shape = (len(images), 3, 64, 64)
    images_tensor = torch.zeros(shape)
    for i in range(0, len(images)):
        images_tensor[i] = images[i]

    return images_tensor


    ######################### Getting the latents and binning them ##############################


def get_z(im, model, device):

    # model.eval()

    im = torch.unsqueeze(im, dim=0).to(device)

    # Get the distribution of encoded vectors for this image
    mu, logvar = model.encode(im)

    # Sample from that distribution once to get the latent space representation for this image
    z = model.sample(mu, logvar)

    return z


def get_z_minibatch_wise(tensor_of_images, model):

    # Get the distribution of encoded vectors for this image
    mu, logvar = model.encode(tensor_of_images)
    
    # Sample from that distribution once to get the latent space representation for this image
    z = model.sample(mu, logvar)

    return z


def get_z_minibatch_wise_eval(tensor_of_images, model):

    model.eval()

    with torch.no_grad():

        # Get the distribution of encoded vectors for this image
        mu, logvar = model.encode(tensor_of_images)
        
        # Sample from that distribution once to get the latent space representation for this image
        z = model.sample(mu, logvar)

    return z


def go_through_encoder(tensor_of_images, model):

    # Get the distribution of encoded vectors for this image
    mu, logvar = model.encode(tensor_of_images)
    
    # Sample from that distribution once to get the latent space representation for this image
    z = model.sample(mu, logvar)

    return z, mu, logvar


#def get_latents(model, images, image_ids, device):
#
#    # Getting the images and their attribute values
#    attr_map, id_attr_map = get_attributes()
#    class_index = attr_map[CLASS]
#
#
#    # Keeps a persistent tally as we traverse all images
#    image_counter = 0
#
#    for image in images: 
#        
#        latent_rep = get_z(image, model, device).cpu()
#        
#        # Clipping the latent rep no of columns so that the synthetic dataset is small enough to easily view 
#        latent_rep = latent_rep[:, 0:NO_COLUMNS]
#        
#        
#        # Concatenate the class value of this image to the latent rep
#        current_image_id = image_ids[image_counter]
#        class_val = id_attr_map[current_image_id][class_index]
#        class_val_tensor = torch.tensor([class_val])
#
#        # Adding a dimension with value one as the first dimension so we go from size = 1 to size = 1,1 
#        class_val_tensor_2d = class_val_tensor.unsqueeze(0).cpu()
#        latent_rep = torch.cat([latent_rep, class_val_tensor_2d], dim = 1)
#    
#        if image_counter == 0:
#            twod_matrix_of_reps = latent_rep 
#
#        else:
#            # Concatenating all subsequent entries to the existing tensor
#            twod_matrix_of_reps = torch.cat([twod_matrix_of_reps, latent_rep], dim = 0)
#
#        image_counter += 1
#
#    return twod_matrix_of_reps


def get_latents(model, images_tensor, image_ids, device, attr_csv_path, class_label):

    # Setup the tensor that will hold the class values
    no_images = images_tensor.size()[0]
    shape = (no_images, 1)
    classes_tensor = torch.zeros(shape, device=device)

    # Get the latents
    latents = get_z_minibatch_wise_eval(images_tensor, model)

    # Initialise counter and values that we will need to get the class values of each image
    image_counter = 0
    attr_map, id_attr_map = get_attributes(attr_csv_path)
    class_index = attr_map[class_label]

    # Get classes for each image
    for id in image_ids:
        class_val = id_attr_map[id][class_index]
        classes_tensor[image_counter] = class_val
        image_counter += 1

    # Concatenate latents and classes
    twod_matrix_of_reps = torch.cat([latents, classes_tensor], dim = 1)

    return twod_matrix_of_reps


def get_binned_val(lb, ub, val):

    if (val < lb):
        return float(-1)
    elif (lb <= val < ub):
        return float(0)
    else:
        return float(1)


def bin_dataset(latents_tensor):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   #TODO: pass device as a arg to the function
    # No rows and no columns
    no_columns = len(latents_tensor[1,:])
    no_rows = len(latents_tensor[:,1])
    

    # Iterate over the columns
    for i in range(0, no_columns - 1):

        mean_of_attr = torch.mean(latents_tensor[:,i])
        std_of_attr = torch.std(latents_tensor[:,i])

        # Defining cutoffs for the bins of this column. By default each bin is (]
        ub = mean_of_attr + std_of_attr
        lb = mean_of_attr - std_of_attr

        # Going down each column
        #for j in range(0, no_rows):
        #    latents_tensor[j,i] = get_binned_val(lb, ub, latents_tensor[j,i])
        #latents_tensor[:,i] = np.digitize(latents_tensor[:, i], tensor([lb,ub], device=device))
        print (latents_tensor.device)
        print (torch.tensor([lb,ub], device=device).device)
        latents_tensor[:,i] = torch.bucketize(latents_tensor[:, i], torch.tensor([lb,ub], device=device))-1

    return latents_tensor




################################# Generating Graphs ################################


def plot_loss_per_epch_1(train_losses, filepath):

    epchs_tuple, losses_master_tuple = zip(*train_losses)
    correlation_loss_list = []
    sgd_loss_list = []

    for losses_tuple_per_epch in losses_master_tuple:
        correlation_loss_list.append(losses_tuple_per_epch[0])
        sgd_loss_list.append(losses_tuple_per_epch[2])

    
    epchs_list = list(epchs_tuple)

    
    plt.figure()
    plt.title('Train Loss Across the Epochs')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(epchs_list, correlation_loss_list, 'b', label='correlation_loss', marker = 'o')
    plt.plot(epchs_list, sgd_loss_list,  'r', label='sgd_loss', marker = 'o')
    plt.legend()
    plt.savefig(filepath + '-epch')


def plot_loss_per_epch_2(train_losses, filepath):

    epchs_tuple, losses_master_tuple = zip(*train_losses)
    sgd_loss_list = []

    for losses_tuple_per_epch in losses_master_tuple:
        sgd_loss_list.append(losses_tuple_per_epch[1])

    
    epchs_list = list(epchs_tuple)

    plt.figure()
    plt.title('Train Loss Across the Epochs')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(epchs_list, sgd_loss_list,  'r', label='sgd_loss', marker = 'o')
    plt.legend()
    plt.savefig(filepath + '-epch')




def plot_loss_per_epch_3(train_losses, filepath):

    epchs_tuple, losses_master_tuple = zip(*train_losses)
    correlation_loss_list = []
    sgd_loss_list = []
    original_loss_list = []

    for losses_tuple_per_epch in losses_master_tuple:
        correlation_loss_list.append(losses_tuple_per_epch[2])
        sgd_loss_list.append(losses_tuple_per_epch[3])
        original_loss_list.append(losses_tuple_per_epch[1])

    
    epchs_list = list(epchs_tuple)

    
    plt.figure()
    plt.title('SGD related Loss Across the Epochs')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(epchs_list, correlation_loss_list, 'b', label='correlation_loss', marker = 'o')
    plt.plot(epchs_list, sgd_loss_list,  'r', label='sgd_loss', marker = 'o')
    plt.legend()
    plt.savefig(filepath + '-SGD')


    plt.figure()
    plt.title('Original Loss Across the Epochs')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(epchs_list, original_loss_list, 'g', label='original_loss', marker = 'o')
    plt.legend()
    plt.savefig(filepath + '-Origin')


def plot_loss_per_epch_4(train_losses, filepath):

    epchs_tuple, losses_master_tuple = zip(*train_losses)
    normalized_correlation_loss_list = []
    correlation_loss_list = []
    sgd_loss_list = []
    original_loss_list = []

    for losses_tuple_per_epch in losses_master_tuple:
        correlation_loss_list.append(losses_tuple_per_epch[0])
        normalized_correlation_loss_list.append(losses_tuple_per_epch[1])
        sgd_loss_list.append(losses_tuple_per_epch[2])
        original_loss_list.append(losses_tuple_per_epch[3])

    
    epchs_list = list(epchs_tuple)

    
    plt.figure()
    plt.title('SGD Loss + Norm_Corr_Loss Across the Epochs')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(epchs_list, normalized_correlation_loss_list, 'b', label='norm_corr_loss', marker = 'o')
    plt.plot(epchs_list, sgd_loss_list,  'r', label='sgd_loss', marker = 'o')
    plt.legend()
    plt.savefig(filepath + '-1')


    plt.figure()
    plt.title('Original Loss + Corr_loss Across the Epochs')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(epchs_list, original_loss_list, 'g', label='original_loss', marker = 'o')
    plt.plot(epchs_list, correlation_loss_list, 'y', label='corr_loss', marker = 'o')
    plt.legend()
    plt.savefig(filepath + '-2')



def plot_loss_per_epch_5(train_losses, filepath):

    epchs_tuple, losses_master_tuple = zip(*train_losses)
    normalized_correlation_loss_list = []
    correlation_loss_list = []
    sgd_loss_list = []
    training_original_loss_list = []
    test_original_loss_list = []

    for losses_tuple_per_epch in losses_master_tuple:
        training_original_loss_list.append(losses_tuple_per_epch[1])
        correlation_loss_list.append(losses_tuple_per_epch[2])
        normalized_correlation_loss_list.append(losses_tuple_per_epch[3])
        sgd_loss_list.append(losses_tuple_per_epch[4])
        test_original_loss_list.append(losses_tuple_per_epch[5])

    
    epchs_list = list(epchs_tuple)

    
    plt.figure()
    plt.title('Corr_Loss Across the Minibatches')
    plt.xlabel('Minibatches')
    plt.ylabel('loss')
    plt.plot(epchs_list, correlation_loss_list, 'b', label='corr_loss')
    plt.legend()
    plt.savefig(filepath + '-1')

    plt.figure()
    plt.title('Vae Loss Across the Minibatches')
    plt.xlabel('Minibatches')
    plt.ylabel('loss')
    plt.plot(epchs_list, test_original_loss_list, 'g', label='vae loss')
    plt.legend()
    plt.savefig(filepath + '-2')


    plt.figure()
    plt.title('Normalized corr_Loss Across the Minibatches')
    plt.xlabel('Minibatches')
    plt.ylabel('loss')
    plt.plot(epchs_list, normalized_correlation_loss_list, 'purple', label='normalized_corr_loss')
    plt.legend()
    plt.savefig(filepath + '-3')

    plt.figure()
    plt.title(' SGD_loss Across the Minibatches')
    plt.xlabel('Minibatches')
    plt.ylabel('loss')
    plt.plot(epchs_list, sgd_loss_list, 'pink', label='sgd loss')
    plt.legend()
    plt.savefig(filepath + '-4')


    

######################## Generating Comparisons ###################################

def generate_comparisons_1(model, test_images):
    
    model.eval()

    with torch.no_grad():
        output, mu, logvar = model(test_images)

    return output



def generate_comparisons_2(model, images_path):

    image_ids = ['{:06d}.jpg'.format(x) for x in range(1, 5 + 1)]
    images_list = get_ims(image_ids, images_path)
    shape = (len(images_list), 3, 64, 64)
    test_images_tensor = torch.zeros(shape)
    for i in range(0, len(test_images_tensor)):
        test_images_tensor[i] = images_list[i]


    model.eval()

    with torch.no_grad():
        output, mu, logvar = model(test_images_tensor)

    return test_images_tensor, output


################################### Overall testing functions ######################################

def compute_original_loss(output, test_images_tensor, mu, logvar):

    beta = 1
    recon_loss = F.binary_cross_entropy(output, test_images_tensor, reduction='sum')
    kl_diverge = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    original_loss = (recon_loss + beta * kl_diverge) / 5

    return original_loss


def test_reconstructions(model, device, images_path):

    image_ids = ['{:06d}.jpg'.format(x) for x in range(1, 5 + 1)]
    images_list = get_ims(image_ids, images_path)
    if "celeb" in images_path:
        shape = (len(images_list), 3, 64, 64)   #TODO
    else:
        shape = (len(images_list), 1, 64, 64)
    test_images_tensor = torch.zeros(shape)
    for i in range(0, len(test_images_tensor)):
        test_images_tensor[i] = images_list[i]
    test_images_tensor = test_images_tensor.to(device)
    model.eval()

    with torch.no_grad():
        output, mu, logvar = model(test_images_tensor)

    original_loss = compute_original_loss(output, test_images_tensor, mu, logvar)

    test_images_to_display = test_images_tensor[0:5]
    output_to_display = output[0:5]

    return test_images_to_display, output_to_display, original_loss

################################## Decoding Latents ####################################

def decode_latent(latent, model):
    # model.eval()

    with torch.no_grad():
        image = model.decode(latent)
    
    return image


def go_though_decoder(latents, model):

    reconstructed_images = model.decode(latents)
    
    return reconstructed_images


def show_images(images_tensor, no_images_per_row):
    
    images_tensor_copy = images_tensor.clone().detach()
    no_images = len(images_tensor_copy)
    no_rows = no_images/no_images_per_row
    no_rows = int(no_rows)  

    for i in range(0 , no_images):
        plt.subplot(no_rows, no_images_per_row, i + 1)
        fixed_dimension_image = np.transpose(images_tensor_copy[i], (1, 2, 0))
        plt.imshow(fixed_dimension_image)
        plt.axis('off')
    
    plt.show()
    
############################ File System Manipulation ##########################

def create_folder(folder_to_create):

    # If this folder does not exist, then please create it
    if not (os.path.isdir(folder_to_create)):
        os.makedirs(folder_to_create)


########################### Dataframe manipulation #############################

def add_original_loss(r_df, original_loss):
    
    entry_to_add = [{    
            'subgroup' : 'Tested Original Loss:  ', 
            'interestingness' : original_loss.item(), 
            'target_share' : 'NA',
            'coverage' : 'NA', 
            'no_positive_inst_in_sg' : 'NA',
            'no_inst_in_sg' : 'NA',
                        }]

    entry_to_add_df = pd.DataFrame(entry_to_add)
    return r_df.append(entry_to_add_df)
