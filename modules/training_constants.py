EXPERIMENT_NUMBER=1
DATASET= "celeba"

# For general training
EPOCHS = 401
LATENT_SIZE = 100
MODEL = 'dfc_' +  '_latents'  + str(LATENT_SIZE)
LEARNING_RATE = 1e-3
PRINT_INTERVAL = 5
CORR_LOSS_COEFF = 10
NUM_PREV_MODELS_TO_KEEP = 15
NO_TRAINING_IMAGES = 202599



# Conventions with minibatch train
# -mb-<training method> - <corr_loss_coeff if applicable>- <mb size>- <how much of the original dataset> - <min individuals needed for subgroup> - <no of experiment>

# Conventions simple train
# -s-<training_method>-<corr_loss_coeff if applicable>-<dataset_size/no_training_images>-<min no individuals>-<experiment no>


# For the subgroup disovery loss computation
COVERAGE_COEFF = 0.17
MIN_NUM_INDIVIDUALS_PER_SUBGROUP = 50
NORMALIZATION_CONST = 1 + 1*COVERAGE_COEFF
NO_SAMPLES_CONSIDERED_IN_SGD_LOSS = 1
# This is equivalent to 3x the number of latents
NO_LEN_1_SELECTORS = 3*LATENT_SIZE
MAX_DEPTH = 2
BEAM_WIDTH = 7


# Minibatch training constants
BATCH_SIZE = 512
NO_MINIBATCHES_REFINEMENT = 20


# Paths
NAME = 'exp_{}_ds_{}_w_{}-bs_{}_sup-{}'.format(CORR_LOSS_COEFF, DATASET, BATCH_SIZE, MIN_NUM_INDIVIDUALS_PER_SUBGROUP, EXPERIMENT_NUMBER )
LOG_PATH = './logs/' + MODEL + NAME + '/log.pkl'
#PRE_TRAINED_LOG_PATH = './logs/' + MODEL + '-pre-trained' + '/log.pkl'
MODEL_PATH = './checkpoints/' + MODEL + NAME + '/'
PRE_TRAINED_MODEL_PATH = './checkpoints/' + "{}_{}_preTrained".format(DATASET,MODEL )
COMPARISON_PATH = './comparisons/' + MODEL + NAME + '.png'
PER_MINIBATCH_LOG_PATH = './logs/' + MODEL + NAME + '/per-minibatch-log.pkl'
ADDITIONAL_LOG_DATA_PATH = './additional_log_info/' + MODEL + NAME + '/'

