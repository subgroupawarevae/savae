import torch
import torch.optim as optim
import multiprocessing
import time
from modules import autoencoder_models
from modules import utils
from modules import auxiliary as aux
from torchvision.utils import save_image
import os
from modules import sgd_manager as sgd_m
import argparse
import sys
import datetime
torch.manual_seed(0)

#iimport_command = "from configs import config_{} as tconst".format(args.config_id)
#eval(import_command)
#from modules import training_constants as tconst

def get_latent_space_losses(latents, image_ids, device):

    latents_and_class_vals = sgd_m.add_class_vals(latents, image_ids, device, tconst.ATTR_CSV_PATH, tconst.CLASS)
    r_df, sgd_loss, latents_to_optimize = sgd_m.get_subgroups(latents_and_class_vals, device, tconst.COVERAGE_COEFF, tconst.MIN_NUM_INDIVIDUALS_PER_SUBGROUP, tconst.NO_SAMPLES_CONSIDERED_IN_SGD_LOSS, tconst.MAX_DEPTH, tconst.BEAM_WIDTH)
    corr_loss, norm_corr_loss = sgd_m.point_biSerial_corr_loss(latents_and_class_vals, latents_to_optimize)
    print ("Latents to Optimize:")
    print(latents_to_optimize, corr_loss, norm_corr_loss)

    return r_df, corr_loss, norm_corr_loss, sgd_loss

def criterion(model, images_tensor, image_ids, device):

    # Put the images_tensor through the encoder and get the latents
    latents, mu, logvar = aux.go_through_encoder(images_tensor, model)
    
    # Generate the subgroups from the latents -> get the corr loss
    r_df, corr_loss, norm_corr_loss, sgd_loss = get_latent_space_losses(latents, image_ids, device)

    # Take the latents and decode them
    reconstructed_images = aux.go_though_decoder(latents, model)
    
    # Compute the original loss
    original_loss = model.only_vae_loss(reconstructed_images, images_tensor, mu, logvar)
   
    # Add the two losses together
    combined_loss = original_loss + tconst.CORR_LOSS_COEFF * corr_loss

    return combined_loss, original_loss, corr_loss, norm_corr_loss, sgd_loss, r_df



def criterion_vae(model, images_tensor, image_ids, device):

    # Put the images_tensor through the encoder and get the latents
    latents, mu, logvar = aux.go_through_encoder(images_tensor, model)

    # Take the latents and decode them
    reconstructed_images = aux.go_though_decoder(latents, model)

    # Compute the original loss
    original_loss = model.only_vae_loss(reconstructed_images, images_tensor, mu, logvar)

    return original_loss


def train_combined_loss_cntrl(model, device, images_tensor, image_ids, optimizer, mb_counter):
    
    model.train()
    images_tensor = images_tensor.to(device)
    optimizer.zero_grad()
    combined_loss, original_loss, norm_corr_loss, sgd_loss, r_df = criterion(model, images_tensor, image_ids, device)

    train_losses_for_epch = (combined_loss.item(), original_loss.item(), norm_corr_loss.item(), sgd_loss.item())

    print(
        '{} Train Iteration: {} [{}/{} ({:.0f}%)]\t Combined_Loss: {:.3E},  Original_loss: {:.3E},  Norm_corr_loss: {:.3E}  Sgd_loss: {:.3E}' .format(
            time.ctime(time.time()), mb_counter, mb_counter * len(images_tensor),
            tconst.NO_TRAINING_IMAGES, 100. * mb_counter * len(images_tensor) / tconst.NO_TRAINING_IMAGES, 
            combined_loss.item(), original_loss.item(), norm_corr_loss.item(), sgd_loss.item()
        ))

    print(r_df)

    return train_losses_for_epch, r_df

def train_combined_loss(model, device, images_tensor, image_ids, optimizer, epoch):
    
    model.train()
    images_tensor = images_tensor.to(device)
    optimizer.zero_grad()

    # Generate all the losses of this model and output them
    combined_loss, original_loss, corr_loss, norm_corr_loss, sgd_loss, r_df = criterion(model, images_tensor, image_ids, device)

    train_losses_for_epch = [combined_loss.item(), original_loss.item(), corr_loss.item(), norm_corr_loss.item(), sgd_loss.item()]

    print(
        '{} Train Epoch: {} [{}/{} ({:.0f}%)] Combined_Loss: {:.3E},  Original_loss: {:.3E},  Norm_corr_loss: {:.3E}  Sgd_loss: {:.3E}' .format(
            time.ctime(time.time()), epoch, 1 * len(images_tensor),
            len(images_tensor), 100. * 1 * len(images_tensor) / len(images_tensor), combined_loss.item(), 
            original_loss.item(), norm_corr_loss.item(), sgd_loss.item()
        ))

    print(r_df)

    # Optimize
    combined_loss.backward()
    optimizer.step()


    return train_losses_for_epch, r_df


def train_vae_loss(model, device, images_tensor, image_ids, optimizer, epoch):

    model.train()
    images_tensor = images_tensor.to(device)
    optimizer.zero_grad()

    # Generate all the losses of this model and output them
    original_loss = criterion_vae(model, images_tensor, image_ids, device)

    train_losses_for_epch = [original_loss.item()]

    print(
        '{} Train Epoch: {} [{}/{} ({:.0f}%)] Original_loss: {:.3E}' .format(
            time.ctime(time.time()), epoch, 1 * len(images_tensor),
            len(images_tensor), 100. * 1 * len(images_tensor) / len(images_tensor),
            original_loss.item()
        ))


    # Optimize
    original_loss.backward()
    optimizer.step()


    return train_losses_for_epch, None


def main():
    logger = utils.Logger(tconst.LOG_PATH)
    time_logger = utils.Logger(tconst.TIME_LOG_PATH)
    ####################
    ## Prep for training
    ####################

    #device = "cpu"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print ("available device")
    print (device)
    image_ids = ['{:06d}.jpg'.format(x) for x in range(1, tconst.NO_TRAINING_IMAGES + 1)]
    dataset_object = aux.ImageDiskLoader(image_ids, tconst.IMAGES_PATH)
    dataloader_object = torch.utils.data.DataLoader(dataset_object, batch_size=tconst.BATCH_SIZE)
    print (tconst.IMAGES_PATH)
    # Set up model and learning algorithm
    print('latent size:', tconst.LATENT_SIZE)
    model = autoencoder_models.DFCVAE(latent_size=tconst.LATENT_SIZE, channels_num=tconst.CHANNELS_NUM).to(device)

    print('learning rate: ', tconst.LEARNING_RATE)
    optimizer = optim.Adam(model.parameters(), lr=tconst.LEARNING_RATE)

    # Creating the folder to save all the additionally logged data
    aux.create_folder(tconst.ADDITIONAL_LOG_DATA_PATH)


    ############################
    ## Actually Begin Training
    ## Saving method - 57 implies this is how model was performing after optimization on epch 57
    ############################


    
    # Load last model
    epochs_prevous_models = model.load_last_model(tconst.MODEL_PATH) + 1

    if tconst.LOAD_PRE_TRAINED:
        epochs_pretrained = model.load_last_model(tconst.PRE_TRAINED_MODEL_PATH) + 1
    
    # Setup all the lists and counters
    #losses_prev_mb_lst = []


    for epoch in range(0, tconst.EPOCHS):
        # Now we are optimizing after every minibatch, not after every epoch, so the counter is fundamentally shifting
        # from going to recording epochs to minibatches
        minibatch_counter = 0
        
        for mb_info_object in dataloader_object:
            mb_data = mb_info_object[0]
            mb_ids = list(mb_info_object[1])
            print('From: ' + str(mb_ids[0]) + ' - ' + str(mb_ids[-1]))
            print (mb_data.shape) 
            if tconst.NO_MINIBATCHES_REFINEMENT and  minibatch_counter > tconst.NO_MINIBATCHES_REFINEMENT:
                print('##################################')
                print("Refinement Complete!")
                print('##################################')
                break

            print('##################################')
            print("Beginning training for minibatch: ", str(minibatch_counter))
            print('##################################')
            

            # Test the pre optimized model and then optimize
            test_images_tensor, reconstructions, original_loss = aux.test_reconstructions(model, device, tconst.IMAGES_PATH)
            print('Tested Original Loss: ', original_loss.item())
            save_image(
                        list(test_images_tensor) + list(reconstructions), 
                        os.path.join(tconst.ADDITIONAL_LOG_DATA_PATH, "{}_{}.png".format(epoch, minibatch_counter)), nrow = 5
                    )
            print ("Backpropagating...")
            if tconst.LOSS == "vae":
                train_losses_prev_mb, r_df = train_vae_loss(model, device, mb_data, mb_ids, optimizer, minibatch_counter)
            elif tconst.LOSS == "combined":
                train_losses_prev_mb, r_df = train_combined_loss(model, device, mb_data, mb_ids, optimizer, minibatch_counter)
            else:
                print ("invalid loss function. Exiting...")
                exit()
            print ("Done")
            # Save the log file, with all the relevant losses of the pre-optimized model
            losses_for_prev_mb = train_losses_prev_mb + [original_loss.item()]
            #losses_prev_mb_lst.append((minibatch_counter, losses_for_prev_mb))
            #utils.write_log(tconst.LOG_PATH, losses_prev_mb_lst)
            logger.log(str(losses_for_prev_mb))
            # The additionally logged data of the model pre-optimization are saved
            if r_df is not None:
                r_df = aux.add_original_loss(r_df, original_loss)
                r_df.to_csv(os.path.join(tconst.ADDITIONAL_LOG_DATA_PATH, "epoch_"+str(epoch)+'_mb_' +  str(minibatch_counter) + '.csv'))


            # Save models every 25 minibatches or on the last minibatch
            MODEL_FILE_PATH  = os.path.join(tconst.MODEL_PATH, "epoch_"+str(epoch)+'_mb_' +  str(minibatch_counter) + '.pt')
            if minibatch_counter % 1 == 0:
                model.save_model(MODEL_FILE_PATH, tconst.NUM_PREV_MODELS_TO_KEEP)
            elif minibatch_counter == 396:
                model.save_model(MODEL_FILE_PATH, tconst.NUM_PREV_MODELS_TO_KEEP)
            if  minibatch_counter % 1 == 0:
                time_logger.log("epoch: {}, mb: {}, time: {}".format(epoch, minibatch_counter, str(datetime.datetime.now())))
            print(' ')

            minibatch_counter += 1

 

def parse_args():
    parser = argparse.ArgumentParser(sys.argv[1:])
    parser.add_argument("--config_id", type=int, default=0)
    parser.add_argument("--loss", type=str, default="vae")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    import_command = "from configs import config_{} as tconst".format(args.config_id)
    exec (import_command)
    main()


