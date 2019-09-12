def load_parameters():
    """
        Loads the defined parameters
    """
    # Input data params
    DATA_ROOT_PATH = '/SemSeg/Datasets_SemSeg/dataset_carotida/'  # Root path to the data
    DATASET_NAME = 'CAROTIDA_segmentation'  # Dataset name

    TYPE_OUT = '3DSemanticLabel'

    APPLY_MULTILABEL_CLASSIFICATION = False
    
    if TYPE_OUT == '3DSemanticLabel':
        MEAN_IMAGE = [113.7162] 	# Training mean image value for grayscale (0.2989 * R + 0.5870 * G + 0.1140 * B)
        MEAN_SUBSTRACTION = False 	# Substract the training mean
        IMG_SIZE = [470, 445, 1]
        IMG_CROP_SIZE = [224, 224, 1]
        RGB = False

        NUM_MODEL_POOLINGS = None

        # Image and features files (the chars {} will be replaced by each type of features)
        NUM_CLASSES = 6
        DISCARD_CLASSES = []
        WEIGHT_CLASSES = []

        IMG_FILES = {'train': ['data/train_gray_images.txt',
                               'data/train_labels.txt'],
                     'val': ['data/train_gray_images.txt',
                             'data/train_labels.txt'],
                     'test': ['data/test18_gray_images.txt',
                              'data/test18_labels.txt']
                     }
        CLASSES_PATH = 'data/classes.txt'

        # Evaluation
        METRICS = ['sem_seg_iou']   # Metric used for evaluating model after each epoch (AP = average precision)
                                    # (leave empty if only prediction is required)
        STOP_METRIC = 'mean IoU'    # Metric for the stop, possible values: 'semantic global accuracy' and 'mean IoU'

    # Prepare input mapping between dataset and model
    INPUTS_IDS_DATASET = ['images']  # Corresponding inputs of the dataset
    INPUTS_IDS_MODEL = ['input_1']  # Corresponding inputs of the built model ('input_1' for ResNet50)
    INPUTS_MAPPING = {'input_1': 0}

    # Prepare output mapping between dataset and model
    OUTPUTS_IDS_DATASET = ['out_3Dlabel']  # Corresponding outputs of the dataset
    OUTPUTS_IDS_MODEL = ['out_3Dlabel']  # Corresponding outputs of the built model
    OUTPUTS_MAPPING = {'out_3Dlabel': 0}

    if APPLY_MULTILABEL_CLASSIFICATION:
        OUTPUTS_IDS_DATASET.append('binary_multilabel')
        OUTPUTS_IDS_MODEL.append('binary_multilabel')
        OUTPUTS_MAPPING['binary_multilabel'] = 1

    # Evaluation params
    # METRICS = ['AP']                              # Metric used for evaluating model after each epoch (AP = average precision)
    #                                              # (leave empty if only prediction is required)
    EVAL_ON_SETS = ['val', 'test']  # Possible values: 'train', 'val' and 'test' (external evaluator)
    EVAL_ON_SETS_KERAS = ['val', 'test']  # Possible values: 'train', 'val' and 'test' (Keras' evaluator)
    START_EVAL_ON_EPOCH = 1  # First epoch where the model will be evaluated
    EVAL_EACH_EPOCHS = True  # Select whether evaluate between N epochs or N updates
    EVAL_EACH = 1  # Number of epochs between each evaluation
                   # (or iterations if EVAL_EACH_EPOCHS=False)

    # Input data parameters
    NORMALIZE = True                              # Normalize input data
    NORMALIZATION_TYPE = '(-1)-1'                 # Normalize input data to [0, 1]: '0-1', or to [-1,1]: '(-1)-1'

    DATA_AUGMENTATION = True  # Apply data augmentation on input data
                              #  (noise on features, random crop on images)
    SHUFFLE_TRAIN = True  # Apply shuffling on training data at the beginnin of each epoch
    IMAGE_CROPPING = True  # FULL: False. Switches on/off image cropping

    EVAL_ORIG_SIZE = False  # parameter for evaluating on original image size (useful in case of cropping)

    # Tiramisu parameters
    TIRAMISU_N_LAYERS_DOWN = [5, 5, 5, 5, 5]      # Tiramisu 56: [4, 4, 4, 4, 4], Tiramisu 67: [5, 5, 5, 5, 5], Tiramisu 103: [4, 5, 7, 10, 12]
    TIRAMISU_N_LAYERS_UP = [5, 5, 5, 5, 5]       # Tiramisu 56: [4, 4, 4, 4, 4], Tiramisu 67: [5, 5, 5, 5, 5], Tiramisu 103: [12, 10, 7, 5, 4]
    TIRAMISU_BOTTLENECK_LAYERS = 5                 # Tiramisu 56: 4, Tiramisu 67: 5, Tiramisu 103: 15

    TIRAMISU_INIT_FILTERS = 48
    TIRAMISU_GROWTH_RATE = 16       # Tiramisu 56: 12, Tiramisu 67-103: 16
    TIRAMISU_N_TRANSITION_BLOCKS = len(TIRAMISU_N_LAYERS_DOWN)

    # Add multiple att. mechanisms at different image scales and points of the model
    ATT_CONNECTIONS_SKIP = ['merge_7', 'merge_14', 'merge_21', 'merge_28', 'merge_35'] # Tiramisu67
    ATT_CONNECTIONS_DIMS = [[360, 480], [180, 240], [90, 120], [45, 60], [22, 30]] # Tiramisu67
    ATT_CONNECTIONS_BOTTLENECK = 'merge_41' # Tiramisu67
    USE_ATT = False  # Build/load a semantic segmentation model and add att. layers in the skip connections
    TRANSF_WEIGHTS_ATT = False  # Transfer weights from the old model to the new one when adding att. layers

    CLASSIFIER_ACTIVATION = 'softmax'  # Used for multilabel loss. e.g. 'softmax', 'sigmoid', etc.

    # Optimizer parameters (see model.compile() function)
    LOSS = 'categorical_crossentropy'  # 'binary_crossentropy', 'categorical_crossentropy'
    LR_DECAY = 1  # number of epochs before the next LR decay (set to None for disabling)
    LR_GAMMA = 0.995  # multiplier used for decreasing the LR

    OPTIMIZER = 'rmsprop'  # Optimizer
    LR = 0.001  # FULL: 0.0001(recommended values - Adam 0.001 - Adadelta 1.0
    EPSILON = 1e-6
    PRE_TRAINED_LR_MULTIPLIER = 0.01  # learning rate multiplier assigned to all previously trained layers
    PRE_TRAINED_LEARNABLE = True
    WEIGHT_DECAY = 1e-4  # L2 regularization
    CLIP_C = 2.0  # During training, clip gradients to this norm
    SAMPLE_WEIGHTS = False  # Select whether we use a weights matrix (mask) for the data outputs (for sequences only)

    # Training parameters
    MAX_EPOCH = 750  # Stop when computed this number of epochs
    BATCH_SIZE = 3 # FULL: 1.

    PARALLEL_LOADERS = 1  # Parallel data batch loaders
    EPOCHS_FOR_SAVE = 1  # Number of epochs between model saves
    WRITE_VALID_SAMPLES = False  # Write valid samples in file (i.e. predictions)

    # Early stop parameters
    EARLY_STOP = True       # Turns on/off the early stop protocol
    PATIENCE = 100          # FULL: 50 We'll stop if the val STOP_METRIC does not improve after this number of evaluations
    PATIENCE_SPLIT = 'val'  # Possible values: 'train' and 'val'
    RESET_BAD_COUNTER = True # Initializes the bad counter for early stopping to 0 (True), or not (False)

    # Model parameters
    # ===
    # Possible MODEL_TYPE values:
    #                          [ Available Models List ]
    #
    #                          VGG19_AttBased, VGG19_AttBased_Multilabel, VGG19_DenseNet
    #                          DebugModel
    #                          Tiramisu56, Tiramisu67
    #
    # ===
    MODEL_TYPE = 'Tiramisu'  # 'DebugModel', 'ClassicUpsampling', 'Tiramisu'

    # Layer dimensions

    # Regularizers / Normalizers
    WEIGHTS_INIT = 'he_uniform'  # weights initialization function, 'glorot_uniform', 'he_uniform' (default glorot_uniform)

    USE_DROPOUT = True  # Use dropout
    DROPOUT_P = 0.2  # Percentage of units to drop

    USE_NOISE = False  # Use gaussian noise during training
    NOISE_AMOUNT = 0.01  # Amount of noise

    USE_BATCH_NORMALIZATION = False  # If True it is recommended to deactivate Dropout
    USE_PRELU = False  # use PReLU activations
    USE_L2 = False  # L2 normalization on the features

    # Results plot and models storing parameters
    EXTRA_NAME = 'model'  # This will be appended to the end of the model name
    MODEL_NAME = DATASET_NAME + '_' + MODEL_TYPE + '_' + OPTIMIZER.lower()
    MODEL_NAME += '_' + EXTRA_NAME

    STORE_PATH = 'trained_models_carotida/' + MODEL_NAME + '/'  # Models and evaluation results will be stored here
    DATASET_STORE_PATH = 'datasets/'  # Dataset instance will be stored here

    SAMPLING_SAVE_MODE = '3DLabels'  # '3DLabels'
    VERBOSE = 1  # Verbosity level
    RELOAD = 0  # FULL: best epoch. If 0 start training from scratch, otherwise the model
                   # saved on epoch 'RELOAD' will be used
    REBUILD_DATASET = True  # Build again or use stored instance
    MODE = 'training'   # 'training' or 'predicting' (if 'predicting' then RELOAD must
                        # be greater than 0 and EVAL_ON_SETS will be used)

    # Extra parameters for special trainings
    TRAIN_ON_TRAINVAL = False  # train the model on both training and validation sets combined
    FORCE_RELOAD_VOCABULARY = False  # force building a new vocabulary from the training samples applicable if RELOAD > 1

    # ============================================
    parameters = locals().copy()
    return parameters
