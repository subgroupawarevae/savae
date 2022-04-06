import torch
import pandas as pd
#from modules import beam_search_implementation_tensor as beam_search_T
from modules import beam_search_simple as beam_search_T
from modules import (
    utils,
    auxiliary as aux
)
from numpy import sqrt

# Batches of how many are made for the images given in the mini-batch. Can't fill up memory with too many images so we just
# process a few at a time

# The class we are tacking onto the end

def get_latents(model, images_tensor, image_ids, device, class_label):

    # Setup the tensor that will hold the class values
    no_images = images_tensor.size()[0]
    shape = (no_images, 1)
    classes_tensor = torch.zeros(shape)

    # Get the latents
    latents = aux.get_z_minibatch_wise(images_tensor, model)

    # Initialise counter and values that we will need to get the class values of each image
    image_counter = 0
    attr_map, id_attr_map = aux.get_attributes()
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
        for j in range(0, no_rows):
            latents_tensor[j,i] = get_binned_val(lb, ub, latents_tensor[j,i])


    return latents_tensor



def split_on_class_val(data, class_vals):

    # Squeeze so the logic vectors have only one dimension
    logic_vector_1 = torch.squeeze(class_vals == 1)
    logic_vector_neg_1 = torch.squeeze(class_vals == -1)
    
    return data[logic_vector_1, :], data[logic_vector_neg_1, :]


def point_biSerial_corr_loss(latents_tensor, latents_to_optimize):

    data = latents_tensor

    # Split off the last column
    class_vals = data[:, -1:]

    # Taking everything but the last column
    data = data[: , :-1]

    data_P, data_Q = split_on_class_val(data, class_vals)

    # Okay now we should compute the quantities that are not latent-specific
    n = len(data)
    np = len(data_P)
    nq = len(data_Q)

    # Now compute latent specific quantities, and store them in a tensor for each latent
    no_latents = len(latents_to_optimize)
    rpb_loss_for_each_latent = torch.zeros(no_latents)
    

    # Compute quantities for specific latents
    counter = 0
    for i in latents_to_optimize:
        s = data[:, i].std()
        Mp = data_P[:, i].mean()
        Mq = data_Q[:, i].mean()
        rpb = ((Mp-Mq)/s) * (sqrt((nq/n)*(np/n)))
        rpb_loss_for_each_latent[counter] = 1-torch.abs(rpb)
        counter += 1
       
    
    total_rpb_loss = rpb_loss_for_each_latent.sum()
    normalized_rpb_loss = total_rpb_loss/no_latents #Makes the rpb loss a value between 0 and 1

    return total_rpb_loss, normalized_rpb_loss
    

def generate_subgroups(model, images, image_ids, device, COVERAGE_COEFF, MIN_NUM_INDIVIDUALS_PER_SUBGROUP):

    latents_tensor = get_latents(model, images, image_ids, device)
    
    # Binning 
    latents_tensor_copy = latents_tensor.clone().detach()
    binned_latents_tensor = bin_dataset(latents_tensor_copy)

    results_as_df, sgd_loss, latents_to_optimize = beam_search_T.run(binned_latents_tensor, device, COVERAGE_COEFF, MIN_NUM_INDIVIDUALS_PER_SUBGROUP, NO_SAMPLES_CONSIDERED_IN_SGD_LOSS)
    
    print(latents_to_optimize)
    
    corr_loss, normalized_corr_loss = point_biSerial_corr_loss(latents_tensor, latents_to_optimize)

    return corr_loss, normalized_corr_loss, sgd_loss, results_as_df 




def add_class_vals(latents, image_ids, device, attr_csv_path, class_label):

    # Setup the tensor that will hold the class values
    no_images = latents.size()[0]
    shape = (no_images, 1)
    classes_tensor = torch.zeros(shape)
    
    # Initialise counter and values that we will need to get the class values of each image
    image_counter = 0
    attr_map, id_attr_map = aux.get_attributes(attr_csv_path)
    class_index = attr_map[class_label]

    # Get classes for each image
    for id in image_ids:
        class_val = id_attr_map[id][class_index]
        classes_tensor[image_counter] = class_val
        image_counter += 1

    classes_tensor = classes_tensor.to(device)
    #latents.to(device)
    # Concatenate latents and classes
    twod_matrix_of_reps = torch.cat([latents, classes_tensor], dim = 1)

    return twod_matrix_of_reps

    

def get_subgroups(latents_and_class_vals, device, COVERAGE_COEFF, MIN_NUM_INDIVIDUALS_PER_SUBGROUP, NO_SAMPLES_CONSIDERED_IN_SGD_LOSS, MAX_DEPTH, BEAM_WIDTH):

    latents_and_class_vals_copy = latents_and_class_vals.clone().detach()
    binned_latents_tensor = bin_dataset(latents_and_class_vals_copy)
    results_as_df, sgd_loss, latents_to_optimize = beam_search_T.run(binned_latents_tensor, device, COVERAGE_COEFF, MIN_NUM_INDIVIDUALS_PER_SUBGROUP, NO_SAMPLES_CONSIDERED_IN_SGD_LOSS, MAX_DEPTH, BEAM_WIDTH)

    return results_as_df, sgd_loss, latents_to_optimize




    




