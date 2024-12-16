class shd_arguments:
    DB = "SHD"
    SEED = 0

    AUGMENT_SHIFT = 40
    P_BLEND = 0.5

    BATCH_SIZE = 256
    NUM_INPUT = 700
    NUM_HIDDEN = 512
    NUM_OUTPUT = 20
    NUM_LAYER = 1
    READOUT = "li"
    FF_INIT = 150
    RECURRENT = True
    RECURRENT_INIT = 0

    DT = 1
    LEARN_FF = True
    LEARN_REC = True
    DELAYS_LR = 0.1

    NUM_EPOCHS = 500

    K_REG = 5e-11

class ssc_arguments:
    DB = "SSC"
    SEED = 0

    AUGMENT_SHIFT = 40

    BATCH_SIZE = 256
    NUM_INPUT = 700
    NUM_HIDDEN = 1024
    NUM_OUTPUT = 35
    NUM_LAYER = 2
    READOUT = "li"
    FF_INIT = 50
    RECURRENT = False
    RECURRENT_INIT = 0

    DT = 1
    LEARN_FF = True
    DELAYS_LR = 0.1

    NUM_EPOCHS = 500

    K_REG = [5e-12, 5e-12]

class yy_arguments:
    DB = "YY"
    SEED = 0

    NUM_INPUT = 5
    NUM_HIDDEN = 100
    NUM_LAYER = 1
    NUM_OUTPUT = 3
    BATCH_SIZE = 32
    NUM_EPOCHS = 300
    NUM_TRAIN = BATCH_SIZE * 10 * NUM_OUTPUT
    NUM_TEST = BATCH_SIZE  * 2 * NUM_OUTPUT
    EXAMPLE_TIME = 30.0
    DT = 0.01
    READOUT = "lif"
    FF_INIT = 0
    RECURRENT = False
    LEARN_FF = True
    LEARN_REC = False
    DELAYS_LR = 0.1

    
