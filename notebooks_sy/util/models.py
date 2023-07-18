from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, maximum, DepthwiseConv2D, AveragePooling2D, UpSampling2D
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, SpatialDropout2D
from tensorflow.keras.layers import Convolution2D, GlobalAveragePooling2D, GlobalAveragePooling1D, MaxPool2D, ZeroPadding2D
from tensorflow.keras.layers import add,concatenate
from tensorflow.keras.activations import relu, softmax, swish
import tensorflow
import numpy as np

def sigmoidal_decay(e, start=0, end=100, lr_start=1e-3, lr_end=1e-5):
    if e < start:
        return lr_start
    elif e > end:
        return lr_end

    middle = (start + end) / 2
    s = lambda x: 1 / (1 + np.exp(-x))

    return s(13 * (-e + middle) / np.abs(end - start)) * np.abs(lr_start - lr_end) + lr_end

def get_LCNN_o_1_dr_rr(mel_input_shape1,mel_input_shape2, mel_input_shape3, cqt_input_shape, stft_input_shape, use_mel = True, use_cqt = True, use_stft = True, use_rr = False, ord1 = True, dp = .5, fc = False, ext = False, ext2 = False):
        # Create a towy model.
    age = keras.Input(shape=(6,), name = 'age_cat')
    sex = keras.Input(shape=(2,), name = 'sex_cat')
    hw = keras.Input(shape=(2,), name = 'height_weight')
    preg = keras.Input(shape=(1,), name = 'is_preg')
    loc = keras.Input(shape=(5,), name = 'loc')
    mel1 = keras.Input(shape=(mel_input_shape1), name = 'mel1')
    mel2 = keras.Input(shape=(mel_input_shape2), name = 'mel2')
    mel3 = keras.Input(shape=(mel_input_shape3), name = 'mel3')
    cqt1 = keras.Input(shape=(cqt_input_shape), name = 'cqt')
    stft1 = keras.Input(shape=(stft_input_shape), name = 'stft')
    rr1 = keras.Input(shape=(1,), name = 'rr')
        
    ## age embeddig
    age1 = layers.Dense(2, activation = None)(age)

    ## sex embedding
    sex1 = layers.Dense(1, activation = None)(sex)

    ## hw embedding
    hw1 = layers.Dense(1, activation = None)(hw)

    ## loc embedding
    loc1 = layers.Dense(3, activation = None)(loc)

    ## mel embedding
    if use_mel :
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)

        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        mel1_1 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            mel1_1 = Dropout(dp)(mel1_1)

    if use_mel :
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel2)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel2)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)

        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        mel2_1 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            mel2_1 = Dropout(dp)(mel2_1)
            
            
    
    if use_mel :
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel3)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel3)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)

        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        mel3_1 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            mel3_1 = Dropout(dp)(mel3_1)
    
            
    
    if use_cqt :
        ## cqt embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])
        
        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        cqt2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            cqt2 = Dropout(dp)(cqt2)

    if use_stft :
        ## stft embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)
        
        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        
        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])
        
        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        stft2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            stft2 = Dropout(dp)(stft2)
    
    
    if use_mel and use_cqt and use_stft :
        concat2 = layers.Concatenate()([mel1_1,mel2_1, mel3_1, cqt2, stft2])
    if not use_mel and use_cqt and use_stft :
        concat2 = layers.Concatenate()([cqt2, stft2])
    if use_mel and not use_cqt and use_stft :
        concat2 = layers.Concatenate()([mel1_1,mel2_1, mel3_1, stft2])
    if use_mel and use_cqt and not use_stft :
        concat2 = layers.Concatenate()([mel1_1,mel2_1, mel3_1, cqt2])
    if not use_mel and not use_cqt and use_stft :  ## stft 만
        concat2 = stft2
    if use_mel and not use_cqt and not use_stft :  ### mel만
        concat2 = layers.Concatenate()([mel1_1,mel2_1, mel3_1])
    if not use_mel and use_cqt and not use_stft :  ### cqt만
        concat2 = cqt2

    if ext :
        concat1 = layers.Concatenate()([age1, sex1, hw1, loc1, preg, rr1])
        d1 = layers.Dense(3, activation = 'relu')(concat1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2, d1])

    if ext2 :
        d1 = layers.Dense(3, activation = 'relu')(rr1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2, d1])
        
    if fc :
        concat2 = layers.Dense(10, activation = "relu")(concat2)
        if dp :
            concat2 = Dropout(dp)(concat2)
        
    if ord1 :
        res1 = layers.Dense(2, activation = "softmax")(concat2)
    else :
        res1 = layers.Dense(3, activation = "softmax")(concat2)

        
    res2 = layers.Dense(2, activation = "softmax")(concat2)

    model = keras.Model(inputs = [age,sex,hw,preg,loc,mel1,mel2, mel3, cqt1,stft1, rr1] , outputs = res1 )
    
    model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy','AUC'])
    return(model)


def get_LCNN_2_dr_rr(mel_input_shape1,mel_input_shape2,mel_input_shape3, cqt_input_shape, stft_input_shape, use_mel = True, use_cqt = True, use_stft = True, use_rr = False, dp = False, fc = False, ext = False, ext2 = False):
        # Create a towy model.
    age = keras.Input(shape=(6,), name = 'age_cat')
    sex = keras.Input(shape=(2,), name = 'sex_cat')
    hw = keras.Input(shape=(2,), name = 'height_weight')
    preg = keras.Input(shape=(1,), name = 'is_preg')
    loc = keras.Input(shape=(5,), name = 'loc')
    mel1 = keras.Input(shape=(mel_input_shape1), name = 'mel1')
    mel2 = keras.Input(shape=(mel_input_shape2), name = 'mel2')
    mel3 = keras.Input(shape=(mel_input_shape3), name = 'mel3')
    cqt1 = keras.Input(shape=(cqt_input_shape), name = 'cqt')
    stft1 = keras.Input(shape=(stft_input_shape), name = 'stft')
    rr1 = keras.Input(shape=(1,), name = 'rr')
        
        
    ## age embeddig
    age1 = layers.Dense(2, activation = None)(age)

    ## sex embedding
    sex1 = layers.Dense(1, activation = None)(sex)

    ## hw embedding
    hw1 = layers.Dense(1, activation = None)(hw)

    ## loc embedding
    loc1 = layers.Dense(3, activation = None)(loc)

    
    ## mel embedding
    if use_mel :
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)

        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        mel1_1 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            mel1_1 = Dropout(dp)(mel1_1)

    if use_mel :
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel2)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel2)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)

        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        mel2_1 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            mel2_1 = Dropout(dp)(mel2_1)
            
            
    
    if use_mel :
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel3)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel3)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)

        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        mel3_1 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            mel3_1 = Dropout(dp)(mel3_1)
    
    if use_cqt :
        ## cqt embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)

        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        
        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        cqt2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            cqt2 = Dropout(dp)(cqt2)

    if use_stft :
        ## stft embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)
        
        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        stft2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            stft2 = Dropout(dp)(stft2)

    
    if use_mel and use_cqt and use_stft :
        concat2 = layers.Concatenate()([mel1_1,mel2_1, mel3_1, cqt2, stft2])
    if not use_mel and use_cqt and use_stft :
        concat2 = layers.Concatenate()([cqt2, stft2])
    if use_mel and not use_cqt and use_stft :
        concat2 = layers.Concatenate()([mel1_1,mel2_1, mel3_1, stft2])
    if use_mel and use_cqt and not use_stft :
        concat2 = layers.Concatenate()([mel1_1,mel2_1, mel3_1, cqt2])
    if not use_mel and not use_cqt and use_stft :  ## stft 만
        concat2 = stft2
    if use_mel and not use_cqt and not use_stft :  ### mel만
        concat2 = layers.Concatenate()([mel1_1,mel2_1, mel3_1])
    if not use_mel and use_cqt and not use_stft :  ### cqt만
        concat2 = cqt2
    if ext :
        concat1 = layers.Concatenate()([age1, sex1, hw1, loc1, preg, rr1])
        d1 = layers.Dense(3, activation = 'relu')(concat1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2, d1])

    if ext2 :
        d1 = layers.Dense(3, activation = 'relu')(rr1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2, d1])
        
    if fc :
        concat2 = layers.Dense(10, activation = 'relu')(concat2)
        if dp :
            concat2 = Dropout(dp)(concat2)

    res1 = layers.Dense(3, activation = "softmax")(concat2)
    res2 = layers.Dense(2, activation = "softmax")(concat2)

    model = keras.Model(inputs = [age,sex,hw,preg,loc,mel1,mel2, mel3,cqt1,stft1, rr1] , outputs = res2 )
    
    model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy','AUC'])
    return(model)






def get_LCNN_1_swish(mel_input_shape, cqt_input_shape, stft_input_shape, use_mel = True, use_cqt = True, use_stft = True, 
                       use_raw = False, use_rr = False, use_rr_seq = False, ord1 = True, dp = .5, fc = False, ext = False, ext2 = False):
        # Create a towy model.
    age = keras.Input(shape=(6,), name = 'age_cat')
    sex = keras.Input(shape=(2,), name = 'sex_cat')
    hw = keras.Input(shape=(2,), name = 'height_weight')
    preg = keras.Input(shape=(1,), name = 'is_preg')
    loc = keras.Input(shape=(5,), name = 'loc')
    mel1 = keras.Input(shape=(mel_input_shape), name = 'mel')
    cqt1 = keras.Input(shape=(cqt_input_shape), name = 'cqt')
    stft1 = keras.Input(shape=(stft_input_shape), name = 'stft')
    rr1 = keras.Input(shape=(1,), name = 'rr')
        
    ## age embeddig
    age1 = layers.Dense(2, activation = None)(age)

    ## sex embedding
    sex1 = layers.Dense(1, activation = None)(sex)

    ## hw embedding
    hw1 = layers.Dense(1, activation = None)(hw)

    ## loc embedding
    loc1 = layers.Dense(3, activation = None)(loc)

    
    
#         interval1_1 = tf.keras.layers.Conv1D(9, 3, padding="causal",activation=None,dilation_rate=1)(interval)
#     interval1_2 = tf.keras.layers.Conv1D(9, 3, padding="causal",activation=None,dilation_rate=1)(interval)
#     interval_mfm1 = tensorflow.keras.layers.maximum([interval1_1, interval1_2])
#     interval_maxpool_1 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(interval_mfm1)

#     interval2_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=2)(interval_maxpool_1)
#     interval2_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=2)(interval_maxpool_1)
#     interval_mfm2 = tensorflow.keras.layers.maximum([interval2_1, interval2_2])
#     interval_maxpool_2 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(interval_mfm2)
    
    
#     interval3_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=4)(interval_maxpool_2)
#     interval3_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=4)(interval_maxpool_2)
#     interval_mfm3 = tensorflow.keras.layers.maximum([interval3_1, interval3_2])
#     interval_maxpool_3 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(interval_mfm3)


    ## mel embedding
    if use_mel :
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation='swish')(mel1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation='swish')(mel1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)

        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        mel2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            mel2 = Dropout(dp)(mel2)

    if use_cqt :
        ## cqt embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])
        
        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        cqt2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            cqt2 = Dropout(dp)(cqt2)

    if use_stft :
        ## stft embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)
        
        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        
        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])
        
        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        stft2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            stft2 = Dropout(dp)(stft2)
    
    
    if use_mel and use_cqt and use_stft :
        concat2 = layers.Concatenate()([mel2, cqt2, stft2])
    if not use_mel and use_cqt and use_stft :
        concat2 = layers.Concatenate()([cqt2, stft2])
    if use_mel and not use_cqt and use_stft :
        concat2 = layers.Concatenate()([mel2, stft2])
    if use_mel and use_cqt and not use_stft :
        concat2 = layers.Concatenate()([mel2, cqt2])
    if not use_mel and not use_cqt and use_stft :  ## stft 만
        concat2 = stft2
    if use_mel and not use_cqt and not use_stft :  ### mel만
        concat2 = mel2
    if not use_mel and use_cqt and not use_stft :  ### cqt만
        concat2 = cqt2

    if ext :
        concat1 = layers.Concatenate()([age1, sex1, hw1, loc1, preg, rr1])
        d1 = layers.Dense(3, activation = 'relu')(concat1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2, d1])

    if ext2 :
        d1 = layers.Dense(3, activation = 'relu')(rr1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2, d1])
        
    if fc :
        concat2 = layers.Dense(10, activation = "relu")(concat2)
        if dp :
            concat2 = Dropout(dp)(concat2)
        
    if ord1 :
        res1 = layers.Dense(2, activation = "softmax")(concat2)
    else :
        res1 = layers.Dense(3, activation = "softmax")(concat2)

        
    res2 = layers.Dense(2, activation = "softmax")(concat2)

    model = keras.Model(inputs = [age,sex,hw,preg,loc,mel1,cqt1,stft1, rr1] , outputs = res1 )
    
    model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy','AUC'])
    return(model)



def get_LCNN_2_swish(mel_input_shape, cqt_input_shape, stft_input_shape, use_mel = True, use_cqt = True, use_stft = True, 
                     use_raw = False, use_rr = False, use_rr_seq = False, dp = False, fc = False, ext = False, ext2 = False):
        # Create a towy model.
    age = keras.Input(shape=(6,), name = 'age_cat')
    sex = keras.Input(shape=(2,), name = 'sex_cat')
    hw = keras.Input(shape=(2,), name = 'height_weight')
    preg = keras.Input(shape=(1,), name = 'is_preg')
    loc = keras.Input(shape=(5,), name = 'loc')
    mel1 = keras.Input(shape=(mel_input_shape), name = 'mel')
    cqt1 = keras.Input(shape=(cqt_input_shape), name = 'cqt')
    stft1 = keras.Input(shape=(stft_input_shape), name = 'stft')
    rr1 = keras.Input(shape=(1,), name = 'rr')
        
        
    ## age embeddig
    age1 = layers.Dense(2, activation = None)(age)

    ## sex embedding
    sex1 = layers.Dense(1, activation = None)(sex)

    ## hw embedding
    hw1 = layers.Dense(1, activation = None)(hw)

    ## loc embedding
    loc1 = layers.Dense(3, activation = None)(loc)

    if use_mel :
        ## mel embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation='swish')(mel1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation='swish')(mel1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)

        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)
        
        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        
        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        
        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        mel2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            mel2 = Dropout(dp)(mel2)

    if use_cqt :
        ## cqt embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)

        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        
        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        cqt2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            cqt2 = Dropout(dp)(cqt2)

    if use_stft :
        ## stft embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)
        
        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        stft2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            stft2 = Dropout(dp)(stft2)
    
#    concat1 = layers.Concatenate()([age1, sex1, hw1, loc1, preg])
#    d1 = layers.Dense(2, activation = 'relu')(concat1)
    
    if use_mel and use_cqt and use_stft :
        concat2 = layers.Concatenate()([mel2, cqt2, stft2])
    if not use_mel and use_cqt and use_stft :
        concat2 = layers.Concatenate()([cqt2, stft2])
    if use_mel and not use_cqt and use_stft :
        concat2 = layers.Concatenate()([mel2, stft2])
    if use_mel and use_cqt and not use_stft :
        concat2 = layers.Concatenate()([mel2, cqt2])
    if not use_mel and not use_cqt and use_stft :  ## stft 만
        concat2 = stft2
    if use_mel and not use_cqt and not use_stft :  ### mel만
        concat2 = mel2
    if not use_mel and use_cqt and not use_stft :  ### cqt만
        concat2 = cqt2

    if ext :
        concat1 = layers.Concatenate()([age1, sex1, hw1, loc1, preg, rr1])
        d1 = layers.Dense(3, activation = 'relu')(concat1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2, d1])

    if ext2 :
        d1 = layers.Dense(3, activation = 'relu')(rr1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2, d1])
        
    if fc :
        concat2 = layers.Dense(10, activation = 'relu')(concat2)
        if dp :
            concat2 = Dropout(dp)(concat2)

    res1 = layers.Dense(3, activation = "softmax")(concat2)
    res2 = layers.Dense(2, activation = "softmax")(concat2)

    model = keras.Model(inputs = [age,sex,hw,preg,loc,mel1,cqt1,stft1, rr1] , outputs = res2 )
    
    model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy','AUC'])
    return(model)


    
def get_LCNN1_6(mel_input_shape1,mel_input_shape2, mel_input_shape3, cqt_input_shape, 
stft_input_shape, use_mel = True, use_cqt = True, use_stft = True, use_rr = False, ord1 = True,
 dp = .5, fc = False, ext = False, ext2 = False,
  use_swish0=False, use_swish1=False, use_swish2=False, use_swish3=False, use_swish4=False, use_swish5=False,use_swish6=False,use_swish7=False):
        # Create a towy model.
    age = keras.Input(shape=(6,), name = 'age_cat')
    sex = keras.Input(shape=(2,), name = 'sex_cat')
    hw = keras.Input(shape=(2,), name = 'height_weight')
    preg = keras.Input(shape=(1,), name = 'is_preg')
    loc = keras.Input(shape=(5,), name = 'loc')
    mel1 = keras.Input(shape=(mel_input_shape1), name = 'mel1')
    mel2 = keras.Input(shape=(mel_input_shape2), name = 'mel2')
    mel3 = keras.Input(shape=(mel_input_shape3), name = 'mel3')
    cqt1 = keras.Input(shape=(cqt_input_shape), name = 'cqt')
    stft1 = keras.Input(shape=(stft_input_shape), name = 'stft')
    rr1 = keras.Input(shape=(1,), name = 'rr')
        
    ## age embeddig
    age1 = layers.Dense(2, activation = None)(age)

    ## sex embedding
    sex1 = layers.Dense(1, activation = None)(sex)

    ## hw embedding
    hw1 = layers.Dense(1, activation = None)(hw)

    ## loc embedding
    loc1 = layers.Dense(3, activation = None)(loc)


    ## mel embedding
    if use_mel :
        ## mel embedding
        if use_swish0:
            conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation='swish')(mel1)
            conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation='swish')(mel1)
            mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
            max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)

        else:
            conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
            conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
            mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
            max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)


        if use_swish1:
            conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(max3)
            conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(max3)
            mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
            batch6 = BatchNormalization(axis=3, scale=False)(mfm5)
        else:
            conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
            conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
            mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
            batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        if use_swish2:
            conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch6)
            conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch6)
            mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])

            max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
            batch10 = BatchNormalization(axis=3, scale=False)(max9)

            conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch10)
            conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch10)
            mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
            batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        
        else:
            conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
            conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
            mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])

            max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
            batch10 = BatchNormalization(axis=3, scale=False)(max9)

            conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
            conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
            mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
            batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        
        
        if use_swish3:
            conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch13)
            conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch13)
            mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
            
            max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        else:
            conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
            conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
            mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
            
            max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)


        if use_swish4:
            conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(max16)
            conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(max16)
            mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
            batch19 = BatchNormalization(axis=3, scale=False)(mfm18)
        else:
            conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
            conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
            mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
            batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        if use_swish5:
            conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch19)
            conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch19)
            mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
            batch22 = BatchNormalization(axis=3, scale=False)(mfm21)
            
        else:
            conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
            conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
            mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
            batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        if use_swish6:
            conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch22)
            conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch22)
            mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
            batch25 = BatchNormalization(axis=3, scale=False)(mfm24)
        else :
            conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
            conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
            mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
            batch25 = BatchNormalization(axis=3, scale=False)(mfm24) 

        if use_swish7:
            conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch25)
            conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch25)
            mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])
        else:
            conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
            conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
            mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        mel1_1 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            mel1_1 = Dropout(dp)(mel1_1)

    if use_mel :
        ## mel embedding
        if use_swish0:
            conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation='swish')(mel2)
            conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation='swish')(mel2)
            mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
            max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)

        else:
            conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel2)
            conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel2)
            mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
            max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)


        if use_swish1:
            conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(max3)
            conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(max3)
            mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
            batch6 = BatchNormalization(axis=3, scale=False)(mfm5)
        else:
            conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
            conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
            mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
            batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        if use_swish2:
            conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch6)
            conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch6)
            mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])

            max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
            batch10 = BatchNormalization(axis=3, scale=False)(max9)

            conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch10)
            conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch10)
            mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
            batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        
        else:
            conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
            conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
            mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])

            max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
            batch10 = BatchNormalization(axis=3, scale=False)(max9)

            conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
            conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
            mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
            batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        
        
        if use_swish3:
            conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch13)
            conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch13)
            mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
            
            max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        else:
            conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
            conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
            mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
            
            max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)


        if use_swish4:
            conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(max16)
            conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(max16)
            mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
            batch19 = BatchNormalization(axis=3, scale=False)(mfm18)
        else:
            conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
            conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
            mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
            batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        if use_swish5:
            conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch19)
            conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch19)
            mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
            batch22 = BatchNormalization(axis=3, scale=False)(mfm21)
            
        else:
            conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
            conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
            mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
            batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        if use_swish6:
            conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch22)
            conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch22)
            mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
            batch25 = BatchNormalization(axis=3, scale=False)(mfm24)
        else :
            conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
            conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
            mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
            batch25 = BatchNormalization(axis=3, scale=False)(mfm24) 

        if use_swish7:
            conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch25)
            conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch25)
            mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])
        else:
            conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
            conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
            mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        mel2_1 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            mel2_1 = Dropout(dp)(mel2_1)

    if use_mel :
        ## mel embedding
        if use_swish0:
            conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation='swish')(mel3)
            conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation='swish')(mel3)
            mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
            max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)

        else:
            conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel3)
            conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel3)
            mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
            max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)


        if use_swish1:
            conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(max3)
            conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(max3)
            mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
            batch6 = BatchNormalization(axis=3, scale=False)(mfm5)
        else:
            conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
            conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
            mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
            batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        if use_swish2:
            conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch6)
            conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch6)
            mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])

            max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
            batch10 = BatchNormalization(axis=3, scale=False)(max9)

            conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch10)
            conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch10)
            mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
            batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        
        else:
            conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
            conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
            mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])

            max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
            batch10 = BatchNormalization(axis=3, scale=False)(max9)

            conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
            conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
            mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
            batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        
        
        if use_swish3:
            conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch13)
            conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch13)
            mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
            
            max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        else:
            conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
            conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
            mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
            
            max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)


        if use_swish4:
            conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(max16)
            conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(max16)
            mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
            batch19 = BatchNormalization(axis=3, scale=False)(mfm18)
        else:
            conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
            conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
            mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
            batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        if use_swish5:
            conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch19)
            conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch19)
            mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
            batch22 = BatchNormalization(axis=3, scale=False)(mfm21)
            
        else:
            conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
            conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
            mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
            batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        if use_swish6:
            conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch22)
            conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch22)
            mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
            batch25 = BatchNormalization(axis=3, scale=False)(mfm24)
        else :
            conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
            conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
            mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
            batch25 = BatchNormalization(axis=3, scale=False)(mfm24) 

        if use_swish7:
            conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch25)
            conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch25)
            mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])
        else:
            conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
            conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
            mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        mel3_1 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            mel3_1 = Dropout(dp)(mel3_1)

    if use_cqt :
        ## cqt embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])
        
        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        cqt2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            cqt2 = Dropout(dp)(cqt2)

    if use_stft :
        ## stft embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)
        
        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        
        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
        
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])
        
        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        stft2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            stft2 = Dropout(dp)(stft2)
    
    
    if use_mel and use_cqt and use_stft :
        concat2 = layers.Concatenate()([mel1_1,mel2_1, mel3_1, cqt2, stft2])
    if not use_mel and use_cqt and use_stft :
        concat2 = layers.Concatenate()([cqt2, stft2])
    if use_mel and not use_cqt and use_stft :
        concat2 = layers.Concatenate()([mel1_1,mel2_1, mel3_1, stft2])
    if use_mel and use_cqt and not use_stft :
        concat2 = layers.Concatenate()([mel1_1,mel2_1, mel3_1, cqt2])
    if not use_mel and not use_cqt and use_stft :  ## stft 만
        concat2 = stft2
    if use_mel and not use_cqt and not use_stft :  ### mel만
        concat2 = layers.Concatenate()([mel1_1,mel2_1, mel3_1])
    if not use_mel and use_cqt and not use_stft :  ### cqt만
        concat2 = cqt2

    if ext :
        concat1 = layers.Concatenate()([age1, sex1, hw1, loc1, preg, rr1])
        d1 = layers.Dense(3, activation = 'relu')(concat1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2, d1])

    if ext2 :
        d1 = layers.Dense(3, activation = 'relu')(rr1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2, d1])
        
    if fc :
        concat2 = layers.Dense(10, activation = "relu")(concat2)
        if dp :
            concat2 = Dropout(dp)(concat2)
        
    if ord1 :
        res1 = layers.Dense(2, activation = "softmax")(concat2)
    else :
        res1 = layers.Dense(3, activation = "softmax")(concat2)

        
    res2 = layers.Dense(2, activation = "softmax")(concat2)

    model = keras.Model(inputs = [age,sex,hw,preg,loc,mel1,mel2, mel3,cqt1,stft1, rr1] , outputs = res1 )
    
    model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy','AUC'])
    return(model)


def get_LCNN2_6(mel_input_shape1,mel_input_shape2,mel_input_shape3, cqt_input_shape, stft_input_shape, 
use_mel = True, use_cqt = True, use_stft = True, use_rr = False, dp = False, fc = False, ext = False, ext2 = False,
use_swish0=False, use_swish1=False, use_swish2=False, use_swish3=False, use_swish4=False, use_swish5=False,use_swish6=False,use_swish7=False):

    # Create a towy model.
    age = keras.Input(shape=(6,), name = 'age_cat')
    sex = keras.Input(shape=(2,), name = 'sex_cat')
    hw = keras.Input(shape=(2,), name = 'height_weight')
    preg = keras.Input(shape=(1,), name = 'is_preg')
    loc = keras.Input(shape=(5,), name = 'loc')
    mel1 = keras.Input(shape=(mel_input_shape1), name = 'mel1')
    mel2 = keras.Input(shape=(mel_input_shape2), name = 'mel2')
    mel3 = keras.Input(shape=(mel_input_shape3), name = 'mel3')
    cqt1 = keras.Input(shape=(cqt_input_shape), name = 'cqt')
    stft1 = keras.Input(shape=(stft_input_shape), name = 'stft')
    rr1 = keras.Input(shape=(1,), name = 'rr')
        
        
    ## age embeddig
    age1 = layers.Dense(2, activation = None)(age)

    ## sex embedding
    sex1 = layers.Dense(1, activation = None)(sex)

    ## hw embedding
    hw1 = layers.Dense(1, activation = None)(hw)

    ## loc embedding
    loc1 = layers.Dense(3, activation = None)(loc)

    
    ## mel embedding
     ## mel embedding
    if use_mel :
        ## mel embedding
        if use_swish0:
            conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation='swish')(mel1)
            conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation='swish')(mel1)
            mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
            max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)

        else:
            conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
            conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
            mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
            max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)


        if use_swish1:
            conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(max3)
            conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(max3)
            mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
            batch6 = BatchNormalization(axis=3, scale=False)(mfm5)
        else:
            conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
            conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
            mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
            batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        if use_swish2:
            conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch6)
            conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch6)
            mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])

            max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
            batch10 = BatchNormalization(axis=3, scale=False)(max9)

            conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch10)
            conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch10)
            mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
            batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        
        else:
            conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
            conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
            mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])

            max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
            batch10 = BatchNormalization(axis=3, scale=False)(max9)

            conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
            conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
            mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
            batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        
        
        if use_swish3:
            conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch13)
            conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch13)
            mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
            
            max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        else:
            conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
            conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
            mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
            
            max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)


        if use_swish4:
            conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(max16)
            conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(max16)
            mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
            batch19 = BatchNormalization(axis=3, scale=False)(mfm18)
        else:
            conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
            conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
            mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
            batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        if use_swish5:
            conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch19)
            conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch19)
            mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
            batch22 = BatchNormalization(axis=3, scale=False)(mfm21)
            
        else:
            conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
            conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
            mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
            batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        if use_swish6:
            conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch22)
            conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch22)
            mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
            batch25 = BatchNormalization(axis=3, scale=False)(mfm24)
        else :
            conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
            conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
            mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
            batch25 = BatchNormalization(axis=3, scale=False)(mfm24) 

        if use_swish7:
            conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch25)
            conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch25)
            mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])
        else:
            conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
            conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
            mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        mel1_1 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            mel1_1 = Dropout(dp)(mel1_1)

    if use_mel :
        ## mel embedding
        if use_swish0:
            conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation='swish')(mel2)
            conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation='swish')(mel2)
            mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
            max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)

        else:
            conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel2)
            conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel2)
            mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
            max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)


        if use_swish1:
            conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(max3)
            conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(max3)
            mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
            batch6 = BatchNormalization(axis=3, scale=False)(mfm5)
        else:
            conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
            conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
            mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
            batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        if use_swish2:
            conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch6)
            conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch6)
            mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])

            max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
            batch10 = BatchNormalization(axis=3, scale=False)(max9)

            conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch10)
            conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch10)
            mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
            batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        
        else:
            conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
            conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
            mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])

            max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
            batch10 = BatchNormalization(axis=3, scale=False)(max9)

            conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
            conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
            mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
            batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        
        
        if use_swish3:
            conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch13)
            conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch13)
            mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
            
            max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        else:
            conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
            conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
            mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
            
            max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)


        if use_swish4:
            conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(max16)
            conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(max16)
            mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
            batch19 = BatchNormalization(axis=3, scale=False)(mfm18)
        else:
            conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
            conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
            mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
            batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        if use_swish5:
            conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch19)
            conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch19)
            mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
            batch22 = BatchNormalization(axis=3, scale=False)(mfm21)
            
        else:
            conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
            conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
            mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
            batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        if use_swish6:
            conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch22)
            conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch22)
            mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
            batch25 = BatchNormalization(axis=3, scale=False)(mfm24)
        else :
            conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
            conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
            mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
            batch25 = BatchNormalization(axis=3, scale=False)(mfm24) 

        if use_swish7:
            conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch25)
            conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch25)
            mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])
        else:
            conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
            conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
            mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        mel2_1 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            mel2_1 = Dropout(dp)(mel2_1)

    if use_mel :
        ## mel embedding
        if use_swish0:
            conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation='swish')(mel3)
            conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation='swish')(mel3)
            mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
            max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)

        else:
            conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel3)
            conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel3)
            mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
            max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)


        if use_swish1:
            conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(max3)
            conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(max3)
            mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
            batch6 = BatchNormalization(axis=3, scale=False)(mfm5)
        else:
            conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
            conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
            mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
            batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        if use_swish2:
            conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch6)
            conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch6)
            mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])

            max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
            batch10 = BatchNormalization(axis=3, scale=False)(max9)

            conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch10)
            conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch10)
            mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
            batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        
        else:
            conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
            conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
            mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])

            max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
            batch10 = BatchNormalization(axis=3, scale=False)(max9)

            conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
            conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
            mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
            batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        
        
        if use_swish3:
            conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch13)
            conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch13)
            mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
            
            max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        else:
            conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
            conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
            mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])
            
            max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)


        if use_swish4:
            conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(max16)
            conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(max16)
            mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
            batch19 = BatchNormalization(axis=3, scale=False)(mfm18)
        else:
            conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
            conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
            mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
            batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        if use_swish5:
            conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch19)
            conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation='swish')(batch19)
            mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
            batch22 = BatchNormalization(axis=3, scale=False)(mfm21)
            
        else:
            conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
            conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
            mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
            batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        if use_swish6:
            conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch22)
            conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch22)
            mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
            batch25 = BatchNormalization(axis=3, scale=False)(mfm24)
        else :
            conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
            conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
            mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
            batch25 = BatchNormalization(axis=3, scale=False)(mfm24) 

        if use_swish7:
            conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch25)
            conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation='swish')(batch25)
            mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])
        else:
            conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
            conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
            mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        mel3_1 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            mel3_1 = Dropout(dp)(mel3_1)
    
    if use_cqt :
        ## cqt embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)

        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])
        
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        
        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        cqt2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            cqt2 = Dropout(dp)(cqt2)

    if use_stft :
        ## stft embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        mfm2 = tensorflow.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tensorflow.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tensorflow.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tensorflow.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tensorflow.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tensorflow.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tensorflow.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tensorflow.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)
        
        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tensorflow.keras.layers.maximum([conv26_1, conv26_2])

        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        stft2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            stft2 = Dropout(dp)(stft2)

    
    if use_mel and use_cqt and use_stft :
        concat2 = layers.Concatenate()([mel1_1,mel2_1, mel3_1, cqt2, stft2])
    if not use_mel and use_cqt and use_stft :
        concat2 = layers.Concatenate()([cqt2, stft2])
    if use_mel and not use_cqt and use_stft :
        concat2 = layers.Concatenate()([mel1_1,mel2_1, mel3_1, stft2])
    if use_mel and use_cqt and not use_stft :
        concat2 = layers.Concatenate()([mel1_1,mel2_1, mel3_1, cqt2])
    if not use_mel and not use_cqt and use_stft :  ## stft 만
        concat2 = stft2
    if use_mel and not use_cqt and not use_stft :  ### mel만
        concat2 = layers.Concatenate()([mel1_1,mel2_1, mel3_1])
    if not use_mel and use_cqt and not use_stft :  ### cqt만
        concat2 = cqt2
    if ext :
        concat1 = layers.Concatenate()([age1, sex1, hw1, loc1, preg, rr1])
        d1 = layers.Dense(3, activation = 'relu')(concat1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2, d1])

    if ext2 :
        d1 = layers.Dense(3, activation = 'relu')(rr1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2, d1])
        
    if fc :
        concat2 = layers.Dense(10, activation = 'relu')(concat2)
        if dp :
            concat2 = Dropout(dp)(concat2)

    res1 = layers.Dense(3, activation = "softmax")(concat2)
    res2 = layers.Dense(2, activation = "softmax")(concat2)

    model = keras.Model(inputs = [age,sex,hw,preg,loc,mel1,mel2, mel3,cqt1,stft1, rr1] , outputs = res2 )
    
    model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy','AUC'])
    return(model)
