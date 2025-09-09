class shd_rec_arguments:
    DB = "SHD"
    SEED = 0

    AUGMENT_SHIFT = 40
    P_BLEND = 0.5

    BATCH_SIZE = 256
    NUM_INPUT = 700
    NUM_HIDDEN = 512 #1024, 512, 256, 128, 64
    NUM_OUTPUT = 20
    NUM_LAYER = 1
    READOUT = "li"
    INPUT_HIDDEN_MEAN = 0.03
    INPUT_HIDDEN_SD = 0.01
    RECURRENT_MEAN = 0.0
    RECURRENT_SD = 0.02
    HIDDEN_OUT_MEAN = 0.0
    HIDDEN_OUT_SD = 0.03
    FF_INIT = 150
    RECURRENT = True
    RECURRENT_INIT = 0

    LR = 0.001 * 0.01
    DT = 1
    LEARN_FF = True
    LEARN_REC = True
    DELAYS_LR = 0.1

    NUM_EPOCHS = 500

    K_REG = 5e-11

class shd_ff_arguments:
    DB = "SHD"
    SEED = 0

    AUGMENT_SHIFT = 40
    P_BLEND = 0.5

    BATCH_SIZE = 256
    NUM_INPUT = 700
    NUM_HIDDEN = 512 #1024, 512, 256, 128, 64
    NUM_OUTPUT = 20
    NUM_LAYER = 1
    READOUT = "li"
    INPUT_HIDDEN_MEAN = 0.03
    INPUT_HIDDEN_SD = 0.01
    HIDDEN_HIDDEN_MEAN = 0.02
    HIDDEN_HIDDEN_SD = 0.03
    HIDDEN_OUT_MEAN = 0.0
    HIDDEN_OUT_SD = 0.03
    FF_INIT = 100
    RECURRENT = False
    RECURRENT_INIT = 0

    LR = 0.001 * 0.01
    DT = 1
    LEARN_FF = True
    LEARN_REC = True
    DELAYS_LR = 0.1

    NUM_EPOCHS = 500

    K_REG = 5e-11

class ssc_rec_arguments:
    DB = "SHD"
    SEED = 0

    AUGMENT_SHIFT = 40

    BATCH_SIZE = 256
    NUM_INPUT = 700
    NUM_HIDDEN = 512 #1024, 512, 256, 128, 64
    NUM_OUTPUT = 20
    NUM_LAYER = 1
    READOUT = "li"
    INPUT_HIDDEN_MEAN = 0.03
    INPUT_HIDDEN_SD = 0.01
    RECURRENT_MEAN = 0.0
    RECURRENT_SD = 0.02
    HIDDEN_OUT_MEAN = 0.0
    HIDDEN_OUT_SD = 0.03
    FF_INIT = 150
    RECURRENT = True
    RECURRENT_INIT = 0

    LR = 0.001 * 0.01
    DT = 1
    LEARN_FF = True
    LEARN_REC = True
    DELAYS_LR = 0.1

    NUM_EPOCHS = 500

    K_REG = 5e-11


class ssc_ff_arguments:
    DB = "SSC"
    SEED = 0

    AUGMENT_SHIFT = 40

    BATCH_SIZE = 256
    NUM_INPUT = 700
    NUM_HIDDEN = 1024 #1024, 512, 256, 128, 64
    NUM_OUTPUT = 35
    NUM_LAYER = 2
    READOUT = "li"
    INPUT_HIDDEN_MEAN = 0.03
    INPUT_HIDDEN_SD = 0.01
    HIDDEN_HIDDEN_MEAN = 0.02
    HIDDEN_HIDDEN_SD = 0.03
    HIDDEN_OUT_MEAN = 0.0
    HIDDEN_OUT_SD = 0.03
    FF_INIT = 50
    RECURRENT = False
    RECURRENT_INIT = 0

    LR = 0.001 * 0.01
    DT = 1
    LEARN_FF = True
    DELAYS_LR = 0.1

    NUM_EPOCHS = 500

    K_REG = [5e-12, 5e-12]

class braille_rec_arguments:
    DB = "BRAILLE"
    SEED = 0

    BATCH_SIZE = 256
    NUM_INPUT = 24
    NUM_HIDDEN = 1024 #1024, 512, 256, 128, 64
    NUM_OUTPUT = 28
    NUM_LAYER = 1
    READOUT = "li"
    INPUT_HIDDEN_MEAN = 0.03
    INPUT_HIDDEN_SD = 0.01
    RECURRENT_MEAN = 0.0
    RECURRENT_SD = 0.02
    HIDDEN_OUT_MEAN = 0.0
    HIDDEN_OUT_SD = 0.03
    FF_INIT = 0
    RECURRENT = True
    RECURRENT_INIT = 0

    LR = 0.0015 * 0.01
    DT = 1
    LEARN_FF = True
    LEARN_REC = True
    DELAYS_LR = 0.025

    NUM_EPOCHS = 500

    K_REG = 1e-10


class braille_ff_arguments:
    DB = "BRAILLE"
    SEED = 0


    BATCH_SIZE = 256
    NUM_INPUT = 24
    NUM_HIDDEN = 1024 #1024, 512, 256, 128, 64
    NUM_OUTPUT = 28
    NUM_LAYER = 2
    READOUT = "li"
    INPUT_HIDDEN_MEAN = 0.03
    INPUT_HIDDEN_SD = 0.01
    HIDDEN_HIDDEN_MEAN = 0.02
    HIDDEN_HIDDEN_SD = 0.03
    HIDDEN_OUT_MEAN = 0.0
    HIDDEN_OUT_SD = 0.03
    FF_INIT = 0
    RECURRENT = False
    RECURRENT_INIT = 0

    LR = 0.0015 * 0.01
    DT = 1
    LEARN_FF = True
    DELAYS_LR = 0.025

    NUM_EPOCHS = 500

    K_REG = [1e-10, 1e-10]

class yy_nodelay_arguments:
    DB = "YY"
    SEED = 0

    NUM_INPUT = 4
    NUM_HIDDEN = 30 # 25, 20, 15, 10, 5
    NUM_LAYER = 1
    NUM_OUTPUT = 3
    BATCH_SIZE = 150
    NUM_EPOCHS = 300
    NUM_TRAIN = 5000
    NUM_VALID = 1000
    NUM_TEST = 1000
    LATE = 20.0
    EARLY = 1.5
    LR = 0.005 # 0.01, 0.015, 0.01, 0.02 0.02
    DT = 0.01 # 0.005, 0.005 0.005, 0.005, 0.01
    DELAYS_LR = 0.0
    READOUT = "lif"
    INPUT_HIDDEN_MEAN = 1.9
    INPUT_HIDDEN_SD = 0.78
    HIDDEN_OUT_MEAN = 0.93
    HIDDEN_OUT_SD = 0.1
    FF_INIT = 0
    RECURRENT = False
    LEARN_FF = False
    LEARN_REC = False
    

class yy_delay_arguments:
    DB = "YY"
    SEED = 0

    NUM_INPUT = 4
    NUM_HIDDEN = 100
    NUM_LAYER = 1
    NUM_OUTPUT = 3
    BATCH_SIZE = 32
    NUM_EPOCHS = 300
    NUM_TRAIN = 5000
    NUM_VALID = 1000
    NUM_TEST = 1000
    LATE = 20.0
    EARLY = 1.5
    LR = 0.001
    DT = 0.01
    READOUT = "lif"
    INPUT_HIDDEN_MEAN = 1.9
    INPUT_HIDDEN_SD = 0.78
    HIDDEN_OUT_MEAN = 0.93
    HIDDEN_OUT_SD = 0.1
    FF_INIT = 0
    RECURRENT = False
    LEARN_FF = True
    LEARN_REC = False
    DELAYS_LR = 0.1
