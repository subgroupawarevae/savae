#celeb refining
import os

RESULT_ROOT = "semantic_sgd_results" 
PROJECT_ROOT = "savae"
EXPERIMENT_NUMBER=1
DATASET= "celeba"

if DATASET=="celeba":
    IMAGES_PATH = "celeba/img_align_celeba/img_align_celeba/"
    ATTR_CSV_PATH = "celeba/list_attr_celeba.csv"
    CLASS= 'attractive'
    CHANNELS_NUM=3
    NO_TRAINING_IMAGES = 202599  #TODO  202599

# For general training
EPOCHS = 1  # 401 original
LATENT_SIZE = 100
MODEL = 'dfc_' + 'latents_'  + str(LATENT_SIZE)
LEARNING_RATE = 1e-3
PRINT_INTERVAL = 5
CORR_LOSS_COEFF = 10
NUM_PREV_MODELS_TO_KEEP = 15
LOSS= "combined"  # vae, combined, combined_fixed_latents
LOAD_PRE_TRAINED = True 

# Conventions with minibatch train
# -mb-<training method> - <corr_loss_coeff if applicable>- <mb size>- <how much of the original dataset> - <min individuals needed for subgroup> - <no of experiment>

# Conventions simple train
# -s-<training_method>-<corr_loss_coeff if applicable>-<dataset_size/no_training_images>-<min no individuals>-<experiment no>


# For the subgroup disovery loss computation
COVERAGE_COEFF = 0.17
MIN_NUM_INDIVIDUALS_PER_SUBGROUP = 50  #50
NORMALIZATION_CONST = 1 + 1*COVERAGE_COEFF
NO_SAMPLES_CONSIDERED_IN_SGD_LOSS = 1
# This is equivalent to 3x the number of latents
NO_LEN_1_SELECTORS = 3*LATENT_SIZE
MAX_DEPTH = 2
BEAM_WIDTH = 10


# Minibatch training constants
BATCH_SIZE = 512  #512
NO_MINIBATCHES_REFINEMENT = None  #TODO


# Paths
NAME = '_exp_{}_ds_{}_w_{}_bs_{}_su_{}'.format(EXPERIMENT_NUMBER, DATASET, CORR_LOSS_COEFF, BATCH_SIZE, MIN_NUM_INDIVIDUALS_PER_SUBGROUP )
LOG_PATH = os.path.join(RESULT_ROOT, 'logs', MODEL + NAME, 'log.pkl')
FULL_NAME = MODEL + NAME
#PRE_TRAINED_LOG_PATH = './logs/' + MODEL + '-pre-trained' + '/log.pkl'
MODEL_PATH = os.path.join(RESULT_ROOT, 'checkpoints/', FULL_NAME )
PRE_TRAINED_MODEL_PATH = os.path.join(PROJECT_ROOT, 'checkpoints', "{}_{}_preTrained".format(DATASET,MODEL ))
COMPARISON_PATH = os.path.join(RESULT_ROOT, 'comparisons_' + FULL_NAME + '.png')
PER_MINIBATCH_LOG_PATH = os.path.join(RESULT_ROOT, 'logs', FULL_NAME , 'per-minibatch-log.pkl')
ADDITIONAL_LOG_DATA_PATH = os.path.join(RESULT_ROOT, 'additional_log_info/', FULL_NAME)
TIME_LOG_PATH = os.path.join(RESULT_ROOT, 'timing_info', FULL_NAME, "info.txt")

# for generating latents
CSV_PATH = os.path.join(RESULT_ROOT, 'latent_reps', FULL_NAME + '.csv')
NO_IMAGES_COMPUTE_OVER = NO_TRAINING_IMAGES
INCREMENT = 5000 # original 5000

# mine subgroups
SUBGROUPS_PATH = os.path.join(RESULT_ROOT, 'subgroups_found', FULL_NAME + '.csv')

# traversal
TRAVERSAL_PATH =  os.path.join(RESULT_ROOT, 'traversal_images', FULL_NAME)
NO_TESTING_IMAGES = 1024

#sgd visualization
SGD_VIS_PATH = os.path.join(RESULT_ROOT, 'sgd_visualizations/',  FULL_NAME)

