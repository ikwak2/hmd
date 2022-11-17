from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, maximum, DepthwiseConv2D, AveragePooling2D, UpSampling2D
from tensorflow.keras.layers import Input, Conv1D,Conv2D, MaxPooling2D, BatchNormalization, SpatialDropout2D
from tensorflow.keras.layers import Convolution2D, GlobalAveragePooling2D, GlobalAveragePooling1D, MaxPool2D, ZeroPadding2D,MaxPool1D
from tensorflow.keras.layers import add,concatenate
from tensorflow.keras.activations import relu, softmax, swish
import tensorflow as tf
import numpy as np
import tensorflow

def sigmoidal_decay(e, start=0, end=100, lr_start=1e-3, lr_end=1e-5):
    if e < start:
        return lr_start
    elif e > end:
        return lr_end

    middle = (start + end) / 2
    s = lambda x: 1 / (1 + np.exp(-x))

    return s(13 * (-e + middle) / np.abs(end - start)) * np.abs(lr_start - lr_end) + lr_end

def SubSpectralNorm(S, A = True, eps=1e-5):
    # S : number of sub-bands
    # A : 'Sub' Transform type if True, 
    #     'All' Transform type if False.          <- cannot be implemented yet.
    # Can be applied only for image inputs.
    
    
    def f(x):
        if S == 1:
            y = BatchNormalization()(x)
            
        else:
            F = x.shape[1]             # number of frequencies of the input.
            if F % S == 0:
                Q = F // S             # length of each sub-band.
                subs = []
                for i in range(S):     
                    subs.append(x[:, i*Q:(i+1)*Q, :,:])
                    
                for i in range(S):
                    subs[i] = BatchNormalization()(subs[i])
                    
            else:
                Q = F // S
                subs = []
                for i in range(S-1):
                    subs.append(x[:,i*Q:(i+1)*Q,:,:])
                    
                subs.append(x[:,(S-1)*Q:,:,:])      # the shape of x and y must be the same.
                
                for i in range(S):
                    subs[i] = BatchNormalization()(subs[i])
                    
            
            y = tf.concat(subs, axis=1)
            
        return y
    
    return f


def TF_ResMax_seq(out_channels = False, strides=(1,1), dilation_rate=(1,1),
                  freq_kernel_size=(3,1), temporal_kernel_size=(1,3), dropout_rate=0.3):
    
    def f(x):
        if out_channels:
            y = Conv2D(filters=out_channels, kernel_size=(1,1), strides=strides,
                       activation=None, kernel_initializer='he_uniform')(x)
            y = BatchNormalization()(y)
            y = relu(y)
            
            z = DepthwiseConv2D(kernel_size=freq_kernel_size, strides=(1,1), padding='same',
                                activation=None, kernel_initializer='he_uniform')(y)
            z = SubSpectralNorm(S=2)(z)
            
        else :
            z = DepthwiseConv2D(kernel_size=freq_kernel_size, strides=(1,1), padding='same',
                                activation=None, kernel_initializer='he_uniform')(x)
            z = SubSpectralNorm(S=2)(z)
            
        z = relu(z)
        z = DepthwiseConv2D(kernel_size=temporal_kernel_size, strides=(1,1), padding='same',
                                activation=None, kernel_initializer='he_uniform')(z)
        z = SubSpectralNorm(S=2)(z)
            #        y = relu(y)
        z = swish(z)
            
        if out_channels:
            z = Conv2D(filters=y.shape[3], kernel_size=(1,1), strides=(1,1),
                    activation='relu', kernel_initializer='he_uniform')(z)
        else :
            z = Conv2D(filters=x.shape[3], kernel_size=(1,1), strides=(1,1),
                       activation='relu', kernel_initializer='he_uniform')(z)
                
        z = SpatialDropout2D(dropout_rate)(z)
        ############################
            
        if out_channels:
            return add([y, z])
        else:
            return add([x, z])
            
    return f


    
def BC_ResMax(out_channels = False, strides=(1,1), dilation_rate=(1,1), 
              freq_kernel_size=(3,1), temporal_kernel_size=(1,3), dropout_rate=0.1):

    def f(x):
        if out_channels:
            y = Conv2D(filters=out_channels, kernel_size=(1,1), strides=strides, 
                       activation=None, kernel_initializer='he_uniform')(x)
            y = BatchNormalization()(y)
            y = relu(y)
            
            y = DepthwiseConv2D(kernel_size=freq_kernel_size, strides=(1,1), padding='same',
                                 activation=None, kernel_initializer='he_uniform')(y)
#            y2 = DepthwiseConv2D(kernel_size=freq_kernel_size, strides=(1,1), padding='same',
#                                 activation=None, kernel_initializer='he_uniform')(y)
#            y = tf.keras.layers.maximum([y1, y2])

        else :
            y = DepthwiseConv2D(kernel_size=freq_kernel_size, strides=(1,1), padding='same',
                                 activation=None, kernel_initializer='he_uniform')(x)
#            y2 = DepthwiseConv2D(kernel_size=freq_kernel_size, strides=(1,1), padding='same',
#                                 activation=None, kernel_initializer='he_uniform')(x)
#            y = tf.keras.layers.maximum([y1, y2])
            
        y = SubSpectralNorm(S=2)(y)                   # Should be changed to SSN
        y = relu(y)
        ############################
        
        z = AveragePooling2D(pool_size=(y.shape[1],1))(y)
        
        ########### f1 #############
        z = DepthwiseConv2D(kernel_size=temporal_kernel_size, strides=(1,1), dilation_rate=dilation_rate,
                            padding='same', activation=None, kernel_initializer='he_uniform')(z)
        z = BatchNormalization()(z)
        z = swish(z)

        z = Conv2D(filters=y.shape[3], kernel_size=(1,1), strides=(1,1),
                   activation='relu', kernel_initializer='he_uniform')(z)
        z = SpatialDropout2D(dropout_rate)(z)                  
        ############################
        
        
        ########### BC #############
        z = UpSampling2D(size=(y.shape[1],1), interpolation='nearest')(z)
        ############################
        
        if out_channels:
            return add([y, z])
        else: 
            return add([x, y, z])
        
    return f

def ResMax(n_output, k, l, upscale=False):
    def f(x):
        conv1_1 = Conv2D(filters = n_output, kernel_size =k, strides=(1, 1), padding='same', activation=None)(x)
        conv1_2 = Conv2D(filters = n_output, kernel_size =k, strides=(1, 1), padding='same', activation=None)(x)
        h = maximum([conv1_1, conv1_2])
        if l :
            conv1_1 = Conv2D(filters = n_output, kernel_size =k, strides=(1, 1), padding='same', activation=None)(h)
            conv1_2 = Conv2D(filters = n_output, kernel_size =k, strides=(1, 1), padding='same', activation=None)(h)
            h = maximum([conv1_1, conv1_2])
        if upscale:
            f = Conv2D(kernel_size=1, filters=n_output, strides=1, padding='same')(x)
        else:
            f = x
        return add([f, h])
    return f




def get_ResMax_1(mel_input_shape, cqt_input_shape, stft_input_shape, use_mel = True, use_cqt = True, use_stft = True):
        # Create a towy model.
    age = keras.Input(shape=(6,), name = 'age_cat')
    sex = keras.Input(shape=(2,), name = 'sex_cat')
    hw = keras.Input(shape=(2,), name = 'height_weight')
    preg = keras.Input(shape=(1,), name = 'is_preg')
    loc = keras.Input(shape=(5,), name = 'loc')
    mel1 = keras.Input(shape=(mel_input_shape), name = 'mel')
    cqt1 = keras.Input(shape=(cqt_input_shape), name = 'cqt')
    stft1 = keras.Input(shape=(stft_input_shape), name = 'stft')
        
        
    ## age embeddig
    age1 = layers.Dense(2, activation = None)(age)

    ## sex embedding
    sex1 = layers.Dense(1, activation = None)(sex)

    ## hw embedding
    hw1 = layers.Dense(1, activation = None)(hw)

    ## loc embedding
    loc1 = layers.Dense(3, activation = None)(loc)

    ## mel embedding
    x = ResMax(16,3,1, upscale = True)(mel1)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = ResMax(16,5,1, upscale = False)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = ResMax(24,3,1, upscale = True)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
#    x = Dropout(dropout_rate)(x)
    x = ResMax(32,3,0, upscale = True)(x)
    x = BatchNormalization()(x)
    x = ResMax(32,3,0, upscale = False)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
#    x = Dropout(dropout_rate)(x)
    x = ResMax(48,3,0, upscale = True)(x)
    x = BatchNormalization()(x)
    x = ResMax(48,1,0, upscale = False)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
#    x = Dropout(dropout_rate)(x)
    x = ResMax(64,1,0, upscale = True)(x)
    x = BatchNormalization()(x)
#    x = Dropout(dropout_rate)(x)
    x = ResMax(64,1,0, upscale = False)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    mel2 = GlobalAveragePooling2D()(x)
    
    ## cqt embedding
    x = ResMax(16,3,1, upscale = True)(cqt1)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = ResMax(16,5,1, upscale = False)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = ResMax(24,3,1, upscale = True)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
#    x = Dropout(dropout_rate)(x)
    x = ResMax(32,3,0, upscale = True)(x)
    x = BatchNormalization()(x)
    x = ResMax(32,3,0, upscale = False)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
#    x = Dropout(dropout_rate)(x)
    x = ResMax(48,3,0, upscale = True)(x)
    x = BatchNormalization()(x)
    x = ResMax(48,1,0, upscale = False)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
#    x = Dropout(dropout_rate)(x)
    x = ResMax(64,1,0, upscale = True)(x)
    x = BatchNormalization()(x)
#    x = Dropout(dropout_rate)(x)
    x = ResMax(64,1,0, upscale = False)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    cqt2 = GlobalAveragePooling2D()(x)

    ## stft embedding
    x = ResMax(16,3,1, upscale = True)(stft1)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = ResMax(16,5,1, upscale = False)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = ResMax(24,3,1, upscale = True)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
#    x = Dropout(dropout_rate)(x)
    x = ResMax(32,3,0, upscale = True)(x)
    x = BatchNormalization()(x)
    x = ResMax(32,3,0, upscale = False)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
#    x = Dropout(dropout_rate)(x)
    x = ResMax(48,3,0, upscale = True)(x)
    x = BatchNormalization()(x)
    x = ResMax(48,1,0, upscale = False)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
#    x = Dropout(dropout_rate)(x)
    x = ResMax(64,1,0, upscale = True)(x)
    x = BatchNormalization()(x)
#    x = Dropout(dropout_rate)(x)
    x = ResMax(64,1,0, upscale = False)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    stft2 = GlobalAveragePooling2D()(x)
    
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
#    concat2 = layers.Dense(10, activation = 'relu')(concat2)
    res1 = layers.Dense(3, activation = "softmax")(concat2)
    res2 = layers.Dense(2, activation = "softmax")(concat2)

    model = keras.Model(inputs = [age,sex,hw,preg,loc,mel1,cqt1,stft1] , outputs = res1 )
    
    model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy','AUC'])
    return(model)

def get_ResMax_o_1(mel_input_shape, cqt_input_shape, stft_input_shape, use_mel = True, use_cqt = True, use_stft = True, ord1 = True):
        # Create a towy model.
    age = keras.Input(shape=(6,), name = 'age_cat')
    sex = keras.Input(shape=(2,), name = 'sex_cat')
    hw = keras.Input(shape=(2,), name = 'height_weight')
    preg = keras.Input(shape=(1,), name = 'is_preg')
    loc = keras.Input(shape=(5,), name = 'loc')
    mel1 = keras.Input(shape=(mel_input_shape), name = 'mel')
    cqt1 = keras.Input(shape=(cqt_input_shape), name = 'cqt')
    stft1 = keras.Input(shape=(stft_input_shape), name = 'stft')
        
        
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
        x = ResMax(16,3,1, upscale = True)(mel1)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ResMax(16,5,1, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ResMax(24,3,1, upscale = True)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(32,3,0, upscale = True)(x)
        x = BatchNormalization()(x)
        x = ResMax(32,3,0, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(48,3,0, upscale = True)(x)
        x = BatchNormalization()(x)
        x = ResMax(48,1,0, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(64,1,0, upscale = True)(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(64,1,0, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        mel2 = GlobalAveragePooling2D()(x)
        mel2 = Dropout(.5)(mel2)

    if use_cqt :
        ## cqt embedding
        x = ResMax(16,3,1, upscale = True)(cqt1)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ResMax(16,5,1, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ResMax(24,3,1, upscale = True)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(32,3,0, upscale = True)(x)
        x = BatchNormalization()(x)
        x = ResMax(32,3,0, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(48,3,0, upscale = True)(x)
        x = BatchNormalization()(x)
        x = ResMax(48,1,0, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(64,1,0, upscale = True)(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(64,1,0, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        cqt2 = GlobalAveragePooling2D()(x)
        cqt2 = Dropout(.5)(cqt2)

    if use_stft :
        ## stft embedding
        x = ResMax(16,3,1, upscale = True)(stft1)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ResMax(16,5,1, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ResMax(24,3,1, upscale = True)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(32,3,0, upscale = True)(x)
        x = BatchNormalization()(x)
        x = ResMax(32,3,0, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(48,3,0, upscale = True)(x)
        x = BatchNormalization()(x)
        x = ResMax(48,1,0, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(64,1,0, upscale = True)(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(64,1,0, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        stft2 = GlobalAveragePooling2D()(x)
        stft2 = Dropout(.5)(stft2)
    
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

    concat2 = Dense(20, activation = "relu")(concat2)
    concat2 = Dropout(.4)(concat2)

    if ord1 :
        res1 = layers.Dense(2, activation = "softmax")(concat2)
    else :
        res1 = layers.Dense(3, activation = "softmax")(concat2)
        
    res2 = layers.Dense(2, activation = "softmax")(concat2)

    model = keras.Model(inputs = [age,sex,hw,preg,loc,mel1,cqt1,stft1] , outputs = res1 )
    
    model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy','AUC'])
    return(model)

def get_ResMax_2(mel_input_shape, cqt_input_shape, stft_input_shape, use_mel = True, use_cqt = True, use_stft = True):
        # Create a towy model.
    age = keras.Input(shape=(6,), name = 'age_cat')
    sex = keras.Input(shape=(2,), name = 'sex_cat')
    hw = keras.Input(shape=(2,), name = 'height_weight')
    preg = keras.Input(shape=(1,), name = 'is_preg')
    loc = keras.Input(shape=(5,), name = 'loc')
    mel1 = keras.Input(shape=(mel_input_shape), name = 'mel')
    cqt1 = keras.Input(shape=(cqt_input_shape), name = 'cqt')
    stft1 = keras.Input(shape=(stft_input_shape), name = 'stft')
        
        
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
        x = ResMax(16,3,1, upscale = True)(mel1)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ResMax(16,5,1, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ResMax(24,3,1, upscale = True)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(32,3,0, upscale = True)(x)
        x = BatchNormalization()(x)
        x = ResMax(32,3,0, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(48,3,0, upscale = True)(x)
        x = BatchNormalization()(x)
        x = ResMax(48,1,0, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(64,1,0, upscale = True)(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(64,1,0, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        mel2 = GlobalAveragePooling2D()(x)
        mel2 = Dropout(.5)(mel2)

    if use_cqt :
        ## cqt embedding
        x = ResMax(16,3,1, upscale = True)(cqt1)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ResMax(16,5,1, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ResMax(24,3,1, upscale = True)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(32,3,0, upscale = True)(x)
        x = BatchNormalization()(x)
        x = ResMax(32,3,0, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(48,3,0, upscale = True)(x)
        x = BatchNormalization()(x)
        x = ResMax(48,1,0, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(64,1,0, upscale = True)(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(64,1,0, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        cqt2 = GlobalAveragePooling2D()(x)
        cqt2 = Dropout(.5)(cqt2)

    if use_stft :
        ## stft embedding
        x = ResMax(16,3,1, upscale = True)(stft1)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ResMax(16,5,1, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ResMax(24,3,1, upscale = True)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(32,3,0, upscale = True)(x)
        x = BatchNormalization()(x)
        x = ResMax(32,3,0, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(48,3,0, upscale = True)(x)
        x = BatchNormalization()(x)
        x = ResMax(48,1,0, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(64,1,0, upscale = True)(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(64,1,0, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        stft2 = GlobalAveragePooling2D()(x)
        stft2 = Dropout(.5)(stft2)
    
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

    concat2 = Dense(20, activation = "relu")(concat2)
    concat2 = Dropout(.4)(concat2)

    res1 = layers.Dense(3, activation = "softmax")(concat2)
    res2 = layers.Dense(2, activation = "softmax")(concat2)

    model = keras.Model(inputs = [age,sex,hw,preg,loc,mel1,cqt1,stft1] , outputs = res2 )
    
    model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy','AUC'])
    return(model)


def get_ResMax_o_1_dr(mel_input_shape, cqt_input_shape, stft_input_shape, use_mel = True, use_cqt = True, use_stft = True, ord1 = True, dp = False, fc = False, ext = False):
        # Create a towy model.
    age = keras.Input(shape=(6,), name = 'age_cat')
    sex = keras.Input(shape=(2,), name = 'sex_cat')
    hw = keras.Input(shape=(2,), name = 'height_weight')
    preg = keras.Input(shape=(1,), name = 'is_preg')
    loc = keras.Input(shape=(5,), name = 'loc')
    mel1 = keras.Input(shape=(mel_input_shape), name = 'mel')
    cqt1 = keras.Input(shape=(cqt_input_shape), name = 'cqt')
    stft1 = keras.Input(shape=(stft_input_shape), name = 'stft')
        
        
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
        x = ResMax(16,3,1, upscale = True)(mel1)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ResMax(16,3,1, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ResMax(24,3,1, upscale = True)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(32,3,0, upscale = True)(x)
        x = BatchNormalization()(x)
        x = ResMax(32,3,0, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(48,3,0, upscale = True)(x)
        x = BatchNormalization()(x)
        x = ResMax(48,1,0, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(64,3,0, upscale = True)(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(64,1,0, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        mel2 = GlobalAveragePooling2D()(x)
        if dp :
            mel2 = Dropout(dp)(mel2)

    if use_cqt :
        ## cqt embedding
        x = ResMax(16,3,1, upscale = True)(cqt1)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ResMax(16,3,1, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ResMax(24,3,1, upscale = True)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(32,3,0, upscale = True)(x)
        x = BatchNormalization()(x)
        x = ResMax(32,3,0, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(48,3,0, upscale = True)(x)
        x = BatchNormalization()(x)
        x = ResMax(48,1,0, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(64,3,0, upscale = True)(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(64,1,0, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        cqt2 = layers.GlobalAveragePooling2D()(x)
        if dp :
            cqt2 = Dropout(dp)(cqt2)

    if use_stft :
        ## stft embedding
        x = ResMax(16,3,1, upscale = True)(stft1)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ResMax(16,3,1, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ResMax(24,3,1, upscale = True)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(32,3,0, upscale = True)(x)
        x = BatchNormalization()(x)
        x = ResMax(32,3,0, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(48,3,0, upscale = True)(x)
        x = BatchNormalization()(x)
        x = ResMax(48,1,0, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(64,3,0, upscale = True)(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(64,1,0, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        stft2 = layers.GlobalAveragePooling2D()(x)
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
        concat1 = layers.Concatenate()([age1, sex1, hw1, loc1, preg])
        d1 = layers.Dense(2, activation = 'relu')(concat1)
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

    model = keras.Model(inputs = [age,sex,hw,preg,loc,mel1,cqt1,stft1] , outputs = res1 )
    
    model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy','AUC'])
    return(model)


def get_ResMax_2_dr(mel_input_shape, cqt_input_shape, stft_input_shape, use_mel = True, use_cqt = True, use_stft = True, dp = False, fc = False, ext = False):
        # Create a towy model.
    age = keras.Input(shape=(6,), name = 'age_cat')
    sex = keras.Input(shape=(2,), name = 'sex_cat')
    hw = keras.Input(shape=(2,), name = 'height_weight')
    preg = keras.Input(shape=(1,), name = 'is_preg')
    loc = keras.Input(shape=(5,), name = 'loc')
    mel1 = keras.Input(shape=(mel_input_shape), name = 'mel')
    cqt1 = keras.Input(shape=(cqt_input_shape), name = 'cqt')
    stft1 = keras.Input(shape=(stft_input_shape), name = 'stft')
        
        
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
        x = ResMax(16,3,1, upscale = True)(mel1)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ResMax(16,3,1, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ResMax(24,3,1, upscale = True)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(32,3,0, upscale = True)(x)
        x = BatchNormalization()(x)
        x = ResMax(32,3,0, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(48,3,0, upscale = True)(x)
        x = BatchNormalization()(x)
        x = ResMax(48,1,0, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(64,3,0, upscale = True)(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(64,1,0, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        mel2 = GlobalAveragePooling2D()(x)
        if dp :
            mel2 = Dropout(dp)(mel2)

    if use_cqt :
        ## cqt embedding
        x = ResMax(16,3,1, upscale = True)(cqt1)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ResMax(16,3,1, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ResMax(24,3,1, upscale = True)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(32,3,0, upscale = True)(x)
        x = BatchNormalization()(x)
        x = ResMax(32,3,0, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(48,3,0, upscale = True)(x)
        x = BatchNormalization()(x)
        x = ResMax(48,1,0, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(64,3,0, upscale = True)(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(64,1,0, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        cqt2 = GlobalAveragePooling2D()(x)
        if dp :
            cqt2 = Dropout(dp)(cqt2)

    if use_stft :
        ## stft embedding
        x = ResMax(16,3,1, upscale = True)(stft1)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ResMax(16,3,1, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ResMax(24,3,1, upscale = True)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(32,3,0, upscale = True)(x)
        x = BatchNormalization()(x)
        x = ResMax(32,3,0, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(48,3,0, upscale = True)(x)
        x = BatchNormalization()(x)
        x = ResMax(48,1,0, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(64,3,0, upscale = True)(x)
        x = BatchNormalization()(x)
        #    x = Dropout(dropout_rate)(x)
        x = ResMax(64,1,0, upscale = False)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        stft2 = GlobalAveragePooling2D()(x)
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
        concat1 = layers.Concatenate()([age1, sex1, hw1, loc1, preg])
        d1 = layers.Dense(2, activation = 'relu')(concat1)
        concat2 = layers.Concatenate()([concat2, d1])
        
    if fc :
        concat2 = layers.Dense(10, activation = 'relu')(concat2)
        if dp :
            concat2 = Dropout(dp)(concat2)

    res1 = layers.Dense(3, activation = "softmax")(concat2)
    res2 = layers.Dense(2, activation = "softmax")(concat2)

    model = keras.Model(inputs = [age,sex,hw,preg,loc,mel1,cqt1,stft1] , outputs = res2 )
    
    model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy','AUC'])
    return(model)





def LCNN_seq_mean_model_1(mel_input_shape, cqt_input_shape, stft_input_shape, interval_input_shape,use_mel = True, use_cqt = False, use_stft = False, ord1 = True, dp = .5, fc = False, ext = False, ext2 = False,use_interval_seq=False):
        # Create a towy model.
    age = keras.Input(shape=(6,), name = 'age_cat')
    sex = keras.Input(shape=(2,), name = 'sex_cat')
    hw = keras.Input(shape=(2,), name = 'height_weight')
    preg = keras.Input(shape=(1,), name = 'is_preg')
    loc = keras.Input(shape=(5,), name = 'loc')
    mel1 = keras.Input(shape=(mel_input_shape), name = 'mel')
    cqt1 = keras.Input(shape=(cqt_input_shape), name = 'cqt')
    stft1 = keras.Input(shape=(stft_input_shape), name = 'stft')
    interval = keras.Input(shape=interval_input_shape, name = 'interval')
    interval_mean=keras.Input(shape=(1,), name = 'interval_mean')
        
    ## age embeddig
    age1 = layers.Dense(2, activation = None)(age)

    ## sex embedding
    sex1 = layers.Dense(1, activation = None)(sex)

    ## hw embedding
    hw1 = layers.Dense(1, activation = None)(hw)

    ## loc embedding
    loc1 = layers.Dense(3, activation = None)(loc)
    
    interval_mean1 = layers.Dense(1, activation = None)(interval_mean)
    
    # USE _INTERVAL_MEAN
        
    conv1_1 = Conv1D(filters = 8, kernel_size =5, strides=1, padding='same', activation=None,dilation_rate=1)(interval)
    conv1_2 = Conv1D(filters = 8, kernel_size =5, strides=1, padding='same', activation=None,dilation_rate=1)(interval)
    mfm2 = tf.keras.layers.maximum([conv1_1, conv1_2])
    max3 = MaxPool1D(pool_size=2, strides=1, padding='same')(mfm2)

    conv4_1 = Conv1D(filters = 8, kernel_size =1, strides=1, padding='same', activation=None,dilation_rate=2)(max3)
    conv4_2 = Conv1D(filters = 8, kernel_size =1, strides=1, padding='same', activation=None,dilation_rate=2)(max3)
    mfm5 = tf.keras.layers.maximum([conv4_1, conv4_2])
    batch6 = BatchNormalization(axis=2, scale=False)(mfm5)

    conv7_1 = Conv1D(filters = 16, kernel_size =3, strides=1, padding='same', activation=None,dilation_rate=4)(batch6)
    conv7_2 = Conv1D(filters = 16, kernel_size =3, strides=1, padding='same', activation=None,dilation_rate=4)(batch6)
    mfm8 = tf.keras.layers.maximum([conv7_1, conv7_2])

    max9 = MaxPool1D(pool_size=2, strides=1, padding='same')(mfm8)
    batch10 = BatchNormalization(axis=2, scale=False)(max9)

    conv11_1 = Conv1D(filters = 16, kernel_size =1, strides=1, padding='same', activation=None,dilation_rate=8)(batch10)
    conv11_2 = Conv1D(filters = 16, kernel_size =1, strides=1, padding='same', activation=None,dilation_rate=8)(batch10)
    mfm12 = tf.keras.layers.maximum([conv11_1, conv11_2])
    batch13 = BatchNormalization(axis=2, scale=False)(mfm12)

    conv14_1 = Conv1D(filters = 32, kernel_size =3, strides=1, padding='same', activation=None,dilation_rate=16)(batch13)
    conv14_2 = Conv1D(filters = 32, kernel_size =3, strides=1, padding='same', activation=None,dilation_rate=16)(batch13)
    mfm15 = tf.keras.layers.maximum([conv14_1, conv14_2])
    max16 = MaxPool1D(pool_size=2, strides=1, padding='same')(mfm15)

    conv17_1 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None,dilation_rate=32)(max16)
    conv17_2 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None,dilation_rate=32)(max16)
    mfm18 = tf.keras.layers.maximum([conv17_1, conv17_2])
    batch19 = BatchNormalization(axis=2, scale=False)(mfm18)

    conv20_1 = Conv1D(filters = 64, kernel_size =3, strides=1, padding='same', activation=None,dilation_rate=64)(batch19)
    conv20_2 = Conv1D(filters = 64, kernel_size =3, strides=1, padding='same', activation=None,dilation_rate=64)(batch19)
    mfm21 = tf.keras.layers.maximum([conv20_1, conv20_2])
    batch22 = BatchNormalization(axis=2, scale=False)(mfm21)

    conv23_1 = Conv1D(filters = 64, kernel_size =1, strides=1, padding='same', activation=None,dilation_rate=128)(batch22)
    conv23_2 = Conv1D(filters = 64, kernel_size =1, strides=1, padding='same', activation=None,dilation_rate=128)(batch22)
    mfm24 = tf.keras.layers.maximum([conv23_1, conv23_2])
    batch25 = BatchNormalization(axis=2, scale=False)(mfm24)

    conv26_1 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None,dilation_rate=256)(batch25)
    conv26_2 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None,dilation_rate=256)(batch25)
    mfm27 = tf.keras.layers.maximum([conv26_1, conv26_2])

    interval1 = layers.GlobalAveragePooling1D()(mfm27)
    interval1 = Dropout(dp)(interval1)
    

    ## mel embedding
    if use_mel :
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        mfm2 = tf.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)

        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tf.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tf.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tf.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tf.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tf.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tf.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tf.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tf.keras.layers.maximum([conv26_1, conv26_2])

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
        
        # max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        cqt2 = layers.GlobalAveragePooling2D()(mfm27)
        if dp :
            cqt2 = Dropout(dp)(cqt2)

    if use_stft :
        ## stft embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        mfm2 = tf.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tf.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)
        
        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tf.keras.layers.maximum([conv7_1, conv7_2])
        
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        
        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tf.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tf.keras.layers.maximum([conv14_1, conv14_2])
        
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tf.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tf.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tf.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tf.keras.layers.maximum([conv26_1, conv26_2])
        
        # max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        stft2 = layers.GlobalAveragePooling2D()(mfm27)
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
        concat1 = layers.Concatenate()([age1, sex1, hw1, loc1, preg, interval1,interval_mean1])
        d1 = layers.Dense(3, activation = 'relu')(concat1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2,d1])

    if ext2 :
        concat1 = layers.Concatenate()([interval_mean1,interval1])
        d1 = layers.Dense(3, activation = 'relu')(concat1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2, d1])


        
    # if fc :
    #     concat2 = layers.Dense(10, activation = "relu")(concat2)
    #     if dp :
    #         concat2 = Dropout(dp)(concat2)
        
    if ord1 :
        res1 = layers.Dense(2, activation = "softmax")(concat2)
    else :
        res1 = layers.Dense(3, activation = "softmax")(concat2)


    model = keras.Model(inputs = [age,sex,hw,preg,loc,mel1,cqt1,stft1, interval_mean,interval] , outputs = res1 )
    
    model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy','AUC'])
    return(model)

def LCNN_seq_mean_model_2(mel_input_shape, cqt_input_shape, stft_input_shape,interval_input_shape, use_mel = True, use_cqt = False, use_stft = False, dp = False, fc = False, ext = False, ext2 = False,use_interval_seq=False):
        # Create a towy model.
    age = keras.Input(shape=(6,), name = 'age_cat')
    sex = keras.Input(shape=(2,), name = 'sex_cat')
    hw = keras.Input(shape=(2,), name = 'height_weight')
    preg = keras.Input(shape=(1,), name = 'is_preg')
    loc = keras.Input(shape=(5,), name = 'loc')
    mel1 = keras.Input(shape=(mel_input_shape), name = 'mel')
    cqt1 = keras.Input(shape=(cqt_input_shape), name = 'cqt')
    stft1 = keras.Input(shape=(stft_input_shape), name = 'stft')
    interval = keras.Input(shape=interval_input_shape, name = 'interval')
    interval_mean=keras.Input(shape=(1,), name = 'interval_mean')
        
        
    ## age embeddig
    age1 = layers.Dense(2, activation = None)(age)

    ## sex embedding
    sex1 = layers.Dense(1, activation = None)(sex)

    ## hw embedding
    hw1 = layers.Dense(1, activation = None)(hw)

    ## loc embedding
    loc1 = layers.Dense(3, activation = None)(loc)
    
    interval_mean1 = layers.Dense(1, activation = None)(interval_mean)
    
        
    conv1_1 = Conv1D(filters = 8, kernel_size =5, strides=1, padding='same', activation=None,dilation_rate=1)(interval)
    conv1_2 = Conv1D(filters = 8, kernel_size =5, strides=1, padding='same', activation=None,dilation_rate=1)(interval)
    mfm2 = tf.keras.layers.maximum([conv1_1, conv1_2])
    max3 = MaxPool1D(pool_size=2, strides=1, padding='same')(mfm2)

    conv4_1 = Conv1D(filters = 8, kernel_size =1, strides=1, padding='same', activation=None,dilation_rate=2)(max3)
    conv4_2 = Conv1D(filters = 8, kernel_size =1, strides=1, padding='same', activation=None,dilation_rate=2)(max3)
    mfm5 = tf.keras.layers.maximum([conv4_1, conv4_2])
    batch6 = BatchNormalization(axis=2, scale=False)(mfm5)

    conv7_1 = Conv1D(filters = 16, kernel_size =3, strides=1, padding='same', activation=None,dilation_rate=4)(batch6)
    conv7_2 = Conv1D(filters = 16, kernel_size =3, strides=1, padding='same', activation=None,dilation_rate=4)(batch6)
    mfm8 = tf.keras.layers.maximum([conv7_1, conv7_2])

    max9 = MaxPool1D(pool_size=2, strides=1, padding='same')(mfm8)
    batch10 = BatchNormalization(axis=2, scale=False)(max9)

    conv11_1 = Conv1D(filters = 16, kernel_size =1, strides=1, padding='same', activation=None,dilation_rate=8)(batch10)
    conv11_2 = Conv1D(filters = 16, kernel_size =1, strides=1, padding='same', activation=None,dilation_rate=8)(batch10)
    mfm12 = tf.keras.layers.maximum([conv11_1, conv11_2])
    batch13 = BatchNormalization(axis=2, scale=False)(mfm12)

    conv14_1 = Conv1D(filters = 32, kernel_size =3, strides=1, padding='same', activation=None,dilation_rate=16)(batch13)
    conv14_2 = Conv1D(filters = 32, kernel_size =3, strides=1, padding='same', activation=None,dilation_rate=16)(batch13)
    mfm15 = tf.keras.layers.maximum([conv14_1, conv14_2])
    max16 = MaxPool1D(pool_size=2, strides=1, padding='same')(mfm15)

    conv17_1 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None,dilation_rate=32)(max16)
    conv17_2 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None,dilation_rate=32)(max16)
    mfm18 = tf.keras.layers.maximum([conv17_1, conv17_2])
    batch19 = BatchNormalization(axis=2, scale=False)(mfm18)

    conv20_1 = Conv1D(filters = 64, kernel_size =3, strides=1, padding='same', activation=None,dilation_rate=64)(batch19)
    conv20_2 = Conv1D(filters = 64, kernel_size =3, strides=1, padding='same', activation=None,dilation_rate=64)(batch19)
    mfm21 = tf.keras.layers.maximum([conv20_1, conv20_2])
    batch22 = BatchNormalization(axis=2, scale=False)(mfm21)

    conv23_1 = Conv1D(filters = 64, kernel_size =1, strides=1, padding='same', activation=None,dilation_rate=128)(batch22)
    conv23_2 = Conv1D(filters = 64, kernel_size =1, strides=1, padding='same', activation=None,dilation_rate=128)(batch22)
    mfm24 = tf.keras.layers.maximum([conv23_1, conv23_2])
    batch25 = BatchNormalization(axis=2, scale=False)(mfm24)

    conv26_1 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None,dilation_rate=256)(batch25)
    conv26_2 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None,dilation_rate=256)(batch25)
    mfm27 = tf.keras.layers.maximum([conv26_1, conv26_2])

    interval1 = layers.GlobalAveragePooling1D()(mfm27)
    interval1 = Dropout(dp)(interval1)
    
    if use_mel :
        ## mel embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        mfm2 = tf.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)

        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tf.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)
        
        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tf.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        
        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tf.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        
        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tf.keras.layers.maximum([conv14_1, conv14_2])
        
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tf.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tf.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tf.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tf.keras.layers.maximum([conv26_1, conv26_2])

        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        mel2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            mel2 = Dropout(dp)(mel2)

    if use_cqt :
        ## cqt embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        mfm2 = tf.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)

        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tf.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tf.keras.layers.maximum([conv7_1, conv7_2])
        
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        
        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tf.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tf.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tf.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tf.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tf.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tf.keras.layers.maximum([conv26_1, conv26_2])

        # max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        cqt2 = layers.GlobalAveragePooling2D()(mfm27)
        if dp :
            cqt2 = Dropout(dp)(cqt2)

    if use_stft :
        ## stft embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        mfm2 = tf.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tf.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tf.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tf.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tf.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tf.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tf.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tf.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)
        
        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tf.keras.layers.maximum([conv26_1, conv26_2])

        # max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        stft2 = layers.GlobalAveragePooling2D()(mfm27)
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
        concat1 = layers.Concatenate()([age1, sex1, hw1, loc1, preg, interval1,interval_mean1])
        d1 = layers.Dense(3, activation = 'relu')(concat1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2, d1])

    if ext2 :
        concat1 = layers.Concatenate()([interval1,interval_mean1])
        d1 = layers.Dense(3, activation = 'relu')(concat1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2, d1,interval1])

    # if fc :
    #     concat2 = layers.Dense(10, activation = 'relu')(concat2)
    #     if dp :
    #         concat2 = Dropout(dp)(concat2)

    res2 = layers.Dense(2, activation = "softmax")(concat2)

    model = keras.Model(inputs = [age,sex,hw,preg,loc,mel1,cqt1,stft1, interval_mean,interval] , outputs = res2 )
    
    model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy','AUC'])
    return(model)



def LCNN_mean_model_1(mel_input_shape, cqt_input_shape, stft_input_shape, interval_input_shape,use_mel = True, use_cqt = False, use_stft = False, ord1 = True, dp = .5, fc = False, ext = False, ext2 = False,use_interval_seq=False):
        # Create a towy model.
    age = keras.Input(shape=(6,), name = 'age_cat')
    sex = keras.Input(shape=(2,), name = 'sex_cat')
    hw = keras.Input(shape=(2,), name = 'height_weight')
    preg = keras.Input(shape=(1,), name = 'is_preg')
    loc = keras.Input(shape=(5,), name = 'loc')
    mel1 = keras.Input(shape=(mel_input_shape), name = 'mel')
    cqt1 = keras.Input(shape=(cqt_input_shape), name = 'cqt')
    stft1 = keras.Input(shape=(stft_input_shape), name = 'stft')
    interval = keras.Input(shape=interval_input_shape, name = 'interval')
    interval_mean=keras.Input(shape=(1,), name = 'interval_mean')
        
    ## age embeddig
    age1 = layers.Dense(2, activation = None)(age)

    ## sex embedding
    sex1 = layers.Dense(1, activation = None)(sex)

    ## hw embedding
    hw1 = layers.Dense(1, activation = None)(hw)

    ## loc embedding
    loc1 = layers.Dense(3, activation = None)(loc)
    
    
    interval_mean1 = layers.Dense(1, activation = None)(interval_mean)
    
    # USE _INTERVAL
        
    # conv1_1 = Conv1D(filters = 8, kernel_size =5, strides=1, padding='same', activation=None,dilation_rate=1)(interval)
    # conv1_2 = Conv1D(filters = 8, kernel_size =5, strides=1, padding='same', activation=None,dilation_rate=1)(interval)
    # mfm2 = tf.keras.layers.maximum([conv1_1, conv1_2])
    # max3 = MaxPool1D(pool_size=2, strides=1, padding='same')(mfm2)

    # conv4_1 = Conv1D(filters = 8, kernel_size =1, strides=1, padding='same', activation=None,dilation_rate=2)(max3)
    # conv4_2 = Conv1D(filters = 8, kernel_size =1, strides=1, padding='same', activation=None,dilation_rate=2)(max3)
    # mfm5 = tf.keras.layers.maximum([conv4_1, conv4_2])
    # batch6 = BatchNormalization(axis=2, scale=False)(mfm5)

    # conv7_1 = Conv1D(filters = 16, kernel_size =3, strides=1, padding='same', activation=None,dilation_rate=4)(batch6)
    # conv7_2 = Conv1D(filters = 16, kernel_size =3, strides=1, padding='same', activation=None,dilation_rate=4)(batch6)
    # mfm8 = tf.keras.layers.maximum([conv7_1, conv7_2])

    # max9 = MaxPool1D(pool_size=2, strides=1, padding='same')(mfm8)
    # batch10 = BatchNormalization(axis=2, scale=False)(max9)

    # conv11_1 = Conv1D(filters = 16, kernel_size =1, strides=1, padding='same', activation=None,dilation_rate=8)(batch10)
    # conv11_2 = Conv1D(filters = 16, kernel_size =1, strides=1, padding='same', activation=None,dilation_rate=8)(batch10)
    # mfm12 = tf.keras.layers.maximum([conv11_1, conv11_2])
    # batch13 = BatchNormalization(axis=2, scale=False)(mfm12)

    # conv14_1 = Conv1D(filters = 32, kernel_size =3, strides=1, padding='same', activation=None,dilation_rate=16)(batch13)
    # conv14_2 = Conv1D(filters = 32, kernel_size =3, strides=1, padding='same', activation=None,dilation_rate=16)(batch13)
    # mfm15 = tf.keras.layers.maximum([conv14_1, conv14_2])
    # max16 = MaxPool1D(pool_size=2, strides=1, padding='same')(mfm15)

    # conv17_1 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None,dilation_rate=32)(max16)
    # conv17_2 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None,dilation_rate=32)(max16)
    # mfm18 = tf.keras.layers.maximum([conv17_1, conv17_2])
    # batch19 = BatchNormalization(axis=2, scale=False)(mfm18)

    # conv20_1 = Conv1D(filters = 64, kernel_size =3, strides=1, padding='same', activation=None,dilation_rate=64)(batch19)
    # conv20_2 = Conv1D(filters = 64, kernel_size =3, strides=1, padding='same', activation=None,dilation_rate=64)(batch19)
    # mfm21 = tf.keras.layers.maximum([conv20_1, conv20_2])
    # batch22 = BatchNormalization(axis=2, scale=False)(mfm21)

    # conv23_1 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None,dilation_rate=128)(batch22)
    # conv23_2 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None,dilation_rate=128)(batch22)
    # mfm24 = tf.keras.layers.maximum([conv23_1, conv23_2])
    # batch25 = BatchNormalization(axis=2, scale=False)(mfm24)

    # conv26_1 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None,dilation_rate=256)(batch25)
    # conv26_2 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None,dilation_rate=256)(batch25)
    # mfm27 = tf.keras.layers.maximum([conv26_1, conv26_2])

    # interval1 = layers.GlobalAveragePooling1D()(mfm27)
    # interval1 = Dropout(dp)(interval1)
    

    ## mel embedding
    if use_mel :
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        mfm2 = tf.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)

        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tf.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tf.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tf.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tf.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tf.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tf.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tf.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tf.keras.layers.maximum([conv26_1, conv26_2])

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
        
        # max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        cqt2 = layers.GlobalAveragePooling2D()(mfm27)
        if dp :
            cqt2 = Dropout(dp)(cqt2)

    if use_stft :
        ## stft embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        mfm2 = tf.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tf.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)
        
        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tf.keras.layers.maximum([conv7_1, conv7_2])
        
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        
        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tf.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tf.keras.layers.maximum([conv14_1, conv14_2])
        
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tf.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tf.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tf.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tf.keras.layers.maximum([conv26_1, conv26_2])
        
        # max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        stft2 = layers.GlobalAveragePooling2D()(mfm27)
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
        concat1 = layers.Concatenate()([age1, sex1, hw1, loc1, preg, interval_mean1])
        d1 = layers.Dense(3, activation = 'relu')(concat1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2,d1])

    if ext2 :
        d1 = layers.Dense(3, activation = 'relu')(interval_mean1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2, d1])


        
    # if fc :
    #     concat2 = layers.Dense(10, activation = "relu")(concat2)
    #     if dp :
    #         concat2 = Dropout(dp)(concat2)
        
    if ord1 :
        res1 = layers.Dense(2, activation = "softmax")(concat2)
    else :
        res1 = layers.Dense(3, activation = "softmax")(concat2)


    model = keras.Model(inputs = [age,sex,hw,preg,loc,mel1,cqt1,stft1, interval_mean,interval] , outputs = res1 )
    
    model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy','AUC'])
    return(model)

def LCNN_mean_model_2(mel_input_shape, cqt_input_shape, stft_input_shape,interval_input_shape, use_mel = True, use_cqt = False, use_stft = False, dp = False, fc = False, ext = False, ext2 = False,use_interval_seq=False):
        # Create a towy model.
    age = keras.Input(shape=(6,), name = 'age_cat')
    sex = keras.Input(shape=(2,), name = 'sex_cat')
    hw = keras.Input(shape=(2,), name = 'height_weight')
    preg = keras.Input(shape=(1,), name = 'is_preg')
    loc = keras.Input(shape=(5,), name = 'loc')
    mel1 = keras.Input(shape=(mel_input_shape), name = 'mel')
    cqt1 = keras.Input(shape=(cqt_input_shape), name = 'cqt')
    stft1 = keras.Input(shape=(stft_input_shape), name = 'stft')
    interval = keras.Input(shape=interval_input_shape, name = 'interval')
    interval_mean=keras.Input(shape=(1,), name = 'interval_mean')
        
        
    ## age embeddig
    age1 = layers.Dense(2, activation = None)(age)

    ## sex embedding
    sex1 = layers.Dense(1, activation = None)(sex)

    ## hw embedding
    hw1 = layers.Dense(1, activation = None)(hw)

    ## loc embedding
    loc1 = layers.Dense(3, activation = None)(loc)
    
    interval_mean1 = layers.Dense(1, activation = None)(interval_mean)
    
        
    # conv1_1 = Conv1D(filters = 8, kernel_size =5, strides=1, padding='same', activation=None,dilation_rate=1)(interval)
    # conv1_2 = Conv1D(filters = 8, kernel_size =5, strides=1, padding='same', activation=None,dilation_rate=1)(interval)
    # mfm2 = tf.keras.layers.maximum([conv1_1, conv1_2])
    # max3 = MaxPool1D(pool_size=2, strides=1, padding='same')(mfm2)

    # conv4_1 = Conv1D(filters = 8, kernel_size =1, strides=1, padding='same', activation=None,dilation_rate=2)(max3)
    # conv4_2 = Conv1D(filters = 8, kernel_size =1, strides=1, padding='same', activation=None,dilation_rate=2)(max3)
    # mfm5 = tf.keras.layers.maximum([conv4_1, conv4_2])
    # batch6 = BatchNormalization(axis=2, scale=False)(mfm5)

    # conv7_1 = Conv1D(filters = 16, kernel_size =3, strides=1, padding='same', activation=None,dilation_rate=4)(batch6)
    # conv7_2 = Conv1D(filters = 16, kernel_size =3, strides=1, padding='same', activation=None,dilation_rate=4)(batch6)
    # mfm8 = tf.keras.layers.maximum([conv7_1, conv7_2])

    # max9 = MaxPool1D(pool_size=2, strides=1, padding='same')(mfm8)
    # batch10 = BatchNormalization(axis=2, scale=False)(max9)

    # conv11_1 = Conv1D(filters = 16, kernel_size =1, strides=1, padding='same', activation=None,dilation_rate=8)(batch10)
    # conv11_2 = Conv1D(filters = 16, kernel_size =1, strides=1, padding='same', activation=None,dilation_rate=8)(batch10)
    # mfm12 = tf.keras.layers.maximum([conv11_1, conv11_2])
    # batch13 = BatchNormalization(axis=2, scale=False)(mfm12)

    # conv14_1 = Conv1D(filters = 32, kernel_size =3, strides=1, padding='same', activation=None,dilation_rate=16)(batch13)
    # conv14_2 = Conv1D(filters = 32, kernel_size =3, strides=1, padding='same', activation=None,dilation_rate=16)(batch13)
    # mfm15 = tf.keras.layers.maximum([conv14_1, conv14_2])
    # max16 = MaxPool1D(pool_size=2, strides=1, padding='same')(mfm15)

    # conv17_1 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None,dilation_rate=32)(max16)
    # conv17_2 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None,dilation_rate=32)(max16)
    # mfm18 = tf.keras.layers.maximum([conv17_1, conv17_2])
    # batch19 = BatchNormalization(axis=2, scale=False)(mfm18)

    # conv20_1 = Conv1D(filters = 64, kernel_size =3, strides=1, padding='same', activation=None,dilation_rate=64)(batch19)
    # conv20_2 = Conv1D(filters = 64, kernel_size =3, strides=1, padding='same', activation=None,dilation_rate=64)(batch19)
    # mfm21 = tf.keras.layers.maximum([conv20_1, conv20_2])
    # batch22 = BatchNormalization(axis=2, scale=False)(mfm21)

    # conv23_1 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None,dilation_rate=128)(batch22)
    # conv23_2 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None,dilation_rate=128)(batch22)
    # mfm24 = tf.keras.layers.maximum([conv23_1, conv23_2])
    # batch25 = BatchNormalization(axis=2, scale=False)(mfm24)

    # conv26_1 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None,dilation_rate=256)(batch25)
    # conv26_2 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None,dilation_rate=256)(batch25)
    # mfm27 = tf.keras.layers.maximum([conv26_1, conv26_2])

    # interval1 = layers.GlobalAveragePooling1D()(mfm27)
    # interval1 = Dropout(dp)(interval1)
    
    if use_mel :
        ## mel embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        mfm2 = tf.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)

        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tf.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)
        
        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tf.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        
        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tf.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)
        
        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tf.keras.layers.maximum([conv14_1, conv14_2])
        
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tf.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tf.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tf.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tf.keras.layers.maximum([conv26_1, conv26_2])

        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        mel2 = layers.GlobalAveragePooling2D()(max28)
        if dp :
            mel2 = Dropout(dp)(mel2)

    if use_cqt :
        ## cqt embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        mfm2 = tf.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)

        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tf.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tf.keras.layers.maximum([conv7_1, conv7_2])
        
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        
        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tf.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tf.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tf.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tf.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tf.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tf.keras.layers.maximum([conv26_1, conv26_2])

        # max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        cqt2 = layers.GlobalAveragePooling2D()(mfm27)
        if dp :
            cqt2 = Dropout(dp)(cqt2)

    if use_stft :
        ## stft embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        mfm2 = tf.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tf.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tf.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tf.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tf.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tf.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tf.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tf.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)
        
        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tf.keras.layers.maximum([conv26_1, conv26_2])

        # max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        stft2 = layers.GlobalAveragePooling2D()(mfm27)
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
        concat1 = layers.Concatenate()([age1, sex1, hw1, loc1, preg, interval_mean1])
        d1 = layers.Dense(3, activation = 'relu')(concat1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2, d1])

    if ext2 :
        
        d1 = layers.Dense(3, activation = 'relu')(interval_mean1)
        d1 = layers.Dense(2, activation = 'relu')(d1)
        concat2 = layers.Concatenate()([concat2, d1])

    # if fc :
    #     concat2 = layers.Dense(10, activation = 'relu')(concat2)
    #     if dp :
    #         concat2 = Dropout(dp)(concat2)

    res2 = layers.Dense(2, activation = "softmax")(concat2)

    model = keras.Model(inputs = [age,sex,hw,preg,loc,mel1,cqt1,stft1, interval_mean,interval] , outputs = res2 )
    
    model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy','AUC'])
    return(model)

def LCNN_mean_jk_model_1(mel_input_shape, cqt_input_shape, stft_input_shape, interval_input_shape,use_mel = True, use_cqt = False, use_stft = False, ord1 = True, dp = .5, fc = False, ext = False, ext2 = False,use_interval_seq=False):
    
   
    
        # Create a towy model.
    age = keras.Input(shape=(6,), name = 'age_cat')
    sex = keras.Input(shape=(2,), name = 'sex_cat')
    hw = keras.Input(shape=(2,), name = 'height_weight')
    preg = keras.Input(shape=(1,), name = 'is_preg')
    loc = keras.Input(shape=(5,), name = 'loc')
    interval = keras.Input(shape=interval_input_shape, name = 'interval')
    interval_mean=keras.Input(shape=(1,), name = 'interval_mean')
    mel1 = keras.Input(shape=mel_input_shape, name = 'mel')
    cqt1 = keras.Input(shape=cqt_input_shape, name = 'cqt')
    stft1 = keras.Input(shape=stft_input_shape, name = 'stft')

    ## age embeddig
    age1 = layers.Dense(2, activation = None)(age)

    ## sex embedding
    sex1 = layers.Dense(1, activation = None)(sex)

    ## hw embedding
    hw1 = layers.Dense(1, activation = None)(hw)

    ## loc embedding
    loc1 = layers.Dense(3, activation = None)(loc)
    
    interval_mean1 = layers.Dense(1, activation = None)(interval_mean)
    
    ## interval embedding
    


    # interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=1)(interval)
    # conv1_1 = Conv1D(filters = 32, kernel_size =5, strides=1, padding='same', activation=None)(interval1)
    # conv1_2 = Conv1D(filters = 32, kernel_size =5, strides=1, padding='same', activation=None)(interval1)
    # mfm2 = tf.keras.layers.maximum([conv1_1, conv1_2])
    # max3 = MaxPool1D(pool_size=2, strides=1, padding='same')(mfm2)

    # interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=2)(max3)
    # conv4_1 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    # conv4_2 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    # mfm5 = tf.keras.layers.maximum([conv4_1, conv4_2])
    # batch6 = BatchNormalization(axis=2, scale=False)(mfm5)

    # interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=4)(batch6)
    # conv7_1 = Conv1D(filters = 48, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    # conv7_2 = Conv1D(filters = 48, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    # mfm8 = tf.keras.layers.maximum([conv7_1, conv7_2])

    # max9 = MaxPool1D(pool_size=2, strides=1, padding='same')(mfm8)
    # batch10 = BatchNormalization(axis=2, scale=False)(max9)

    # interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=8)(batch10)
    # conv11_1 = Conv1D(filters = 48, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    # conv11_2 = Conv1D(filters = 48, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    # mfm12 = tf.keras.layers.maximum([conv11_1, conv11_2])
    # batch13 = BatchNormalization(axis=2, scale=False)(mfm12)

    # interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=16)(batch13)
    # conv14_1 = Conv1D(filters = 64, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    # conv14_2 = Conv1D(filters = 64, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    # mfm15 = tf.keras.layers.maximum([conv14_1, conv14_2])
    # max16 = MaxPool1D(pool_size=2, strides=1, padding='same')(mfm15)

    # interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=32)(max16)
    # conv17_1 = Conv1D(filters = 64, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    # conv17_2 = Conv1D(filters = 64, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    # mfm18 = tf.keras.layers.maximum([conv17_1, conv17_2])
    # batch19 = BatchNormalization(axis=2, scale=False)(mfm18)

    # interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=64)(batch19)
    # conv20_1 = Conv1D(filters = 32, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    # conv20_2 = Conv1D(filters = 32, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    # mfm21 = tf.keras.layers.maximum([conv20_1, conv20_2])
    # batch22 = BatchNormalization(axis=2, scale=False)(mfm21)

    # interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=128)(batch22)
    # conv23_1 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    # conv23_2 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    # mfm24 = tf.keras.layers.maximum([conv23_1, conv23_2])
    # batch25 = BatchNormalization(axis=2, scale=False)(mfm24)

    # interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=256)(batch25)
    # conv26_1 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    # conv26_2 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    # mfm27 = tf.keras.layers.maximum([conv26_1, conv26_2])


    # interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=1)(mfm27)
    # conv1_1 = Conv1D(filters = 32, kernel_size =5, strides=1, padding='same', activation=None)(interval1)
    # conv1_2 = Conv1D(filters = 32, kernel_size =5, strides=1, padding='same', activation=None)(interval1)
    # mfm2 = tf.keras.layers.maximum([conv1_1, conv1_2])
    # max3 = MaxPool1D(pool_size=2, strides=1, padding='same')(mfm2)

    # interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=2)(max3)
    # conv4_1 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    # conv4_2 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    # mfm5 = tf.keras.layers.maximum([conv4_1, conv4_2])
    # batch6 = BatchNormalization(axis=2, scale=False)(mfm5)

    # interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=4)(batch6)
    # conv7_1 = Conv1D(filters = 48, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    # conv7_2 = Conv1D(filters = 48, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    # mfm8 = tf.keras.layers.maximum([conv7_1, conv7_2])
    # max9 = MaxPool1D(pool_size=2, strides=1, padding='same')(mfm8)
    # batch10 = BatchNormalization(axis=2, scale=False)(max9)


    # interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=8)(batch10)
    # conv11_1 = Conv1D(filters = 48, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    # conv11_2 = Conv1D(filters = 48, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    # mfm12 = tf.keras.layers.maximum([conv11_1, conv11_2])
    # batch13 = BatchNormalization(axis=2, scale=False)(mfm12)

    # interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=8)(batch13)
    # conv14_1 = Conv1D(filters = 64, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    # conv14_2 = Conv1D(filters = 64, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    # mfm15 = tf.keras.layers.maximum([conv14_1, conv14_2])

    # max16 = MaxPool1D(pool_size=2, strides=1, padding='same')(mfm15)

    # interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=16)(max16)
    # conv17_1 = Conv1D(filters = 64, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    # conv17_2 = Conv1D(filters = 64, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    # mfm18 = tf.keras.layers.maximum([conv17_1, conv17_2])
    # batch19 = BatchNormalization(axis=2, scale=False)(mfm18)


    # interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=32)(batch19)
    # conv20_1 = Conv1D(filters = 32, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    # conv20_2 = Conv1D(filters = 32, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    # mfm21 = tf.keras.layers.maximum([conv20_1, conv20_2])
    # batch22 = BatchNormalization(axis=2, scale=False)(mfm21)


    # interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=64)(batch22)
    # conv23_1 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    # conv23_2 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    # mfm24 = tf.keras.layers.maximum([conv23_1, conv23_2])
    # batch25 = BatchNormalization(axis=2, scale=False)(mfm24)

    # interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=128)(batch25)
    # conv26_1 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    # conv26_2 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    # mfm27 = tf.keras.layers.maximum([conv26_1, conv26_2])

    # interval1 = layers.GlobalAveragePooling1D()(mfm27)
    # interval1 = Dropout(dp)(interval1)
    
    if use_mel :
        
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        mfm2 = tf.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)

        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tf.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tf.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tf.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tf.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tf.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tf.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tf.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tf.keras.layers.maximum([conv26_1, conv26_2])

      

        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mfm27)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mfm27)
        mfm2 = tf.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)

        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tf.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tf.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tf.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tf.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tf.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tf.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tf.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tf.keras.layers.maximum([conv26_1, conv26_2])

        mel2 = layers.GlobalAveragePooling2D()(mfm27)
        mel2 = Dropout(dp)(mel2)

    
    
    if use_cqt:
        ## cqt embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        mfm2 = tf.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tf.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tf.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tf.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tf.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tf.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tf.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tf.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tf.keras.layers.maximum([conv26_1, conv26_2])
        
        # max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        cqt2 = layers.GlobalAveragePooling2D()(mfm27)
        cqt2 = Dropout(dp)(cqt2)

    if use_stft :
        ## stft embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        mfm2 = tf.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tf.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)
        
        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tf.keras.layers.maximum([conv7_1, conv7_2])
        
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        
        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tf.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tf.keras.layers.maximum([conv14_1, conv14_2])
        
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tf.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tf.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tf.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tf.keras.layers.maximum([conv26_1, conv26_2])
        
        # max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        stft2 = layers.GlobalAveragePooling2D()(mfm27)
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
        
        concat1 = layers.Concatenate()([age1, sex1, hw1, loc1, preg,interval_mean1])
        d1 = layers.Dense(10, activation='relu')(concat1)
        concat2 = layers.Concatenate()([concat2, d1])
    
    if ext2 :
        
        concat2 = layers.Concatenate()([concat2, interval_mean1])
    
    
        
    if fc :
        concat2 = layers.Dense(10, activation = "relu")(concat2)
        concat2 = Dropout(dp)(concat2)
        
    if ord1 :
        res1 = layers.Dense(2, activation = "softmax")(concat2)
        
        

        
    else :
        res1 = layers.Dense(3, activation = "softmax")(concat2)

        
        
#     res2 = layers.Dense(2, activation = "softmax")(concat2)

    model = keras.Model(inputs = [age,sex,hw,preg,loc,mel1,cqt1,stft1, interval_mean,interval] , outputs = res1 )
    
    model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy','AUC'])
    return(model)



####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################

def LCNN_mean_jk_model_2(mel_input_shape, cqt_input_shape, stft_input_shape, interval_input_shape,use_mel = True, use_cqt = False, use_stft = False,dp = .5, fc = False, ext = False, ext2 = False,use_interval_seq=False):
    
   
    
        # Create a towy model.
    
    
    age = keras.Input(shape=(6,), name = 'age_cat')
    sex = keras.Input(shape=(2,), name = 'sex_cat')
    hw = keras.Input(shape=(2,), name = 'height_weight')
    preg = keras.Input(shape=(1,), name = 'is_preg')
    loc = keras.Input(shape=(5,), name = 'loc')
    interval = keras.Input(shape=interval_input_shape, name = 'interval')
    mel1 = keras.Input(shape=mel_input_shape, name = 'mel')
    cqt1 = keras.Input(shape=cqt_input_shape, name = 'cqt')
    stft1 = keras.Input(shape=stft_input_shape, name = 'stft')
    interval_mean=keras.Input(shape=(1,), name = 'interval_mean')
        
    ## age embeddig
    age1 = layers.Dense(2, activation = None)(age)

    ## sex embedding
    sex1 = layers.Dense(1, activation = None)(sex)

    ## hw embedding
    hw1 = layers.Dense(1, activation = None)(hw)

    ## loc embedding
    loc1 = layers.Dense(3, activation = None)(loc)
    
    interval_mean1 = layers.Dense(1, activation = None)(interval_mean)
   
    
    ## interval embedding
    
    
    # interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=1)(interval)
    # conv1_1 = Conv1D(filters = 32, kernel_size =5, strides=1, padding='same', activation=None)(interval1)
    # conv1_2 = Conv1D(filters = 32, kernel_size =5, strides=1, padding='same', activation=None)(interval1)
    # mfm2 = tf.keras.layers.maximum([conv1_1, conv1_2])
    # max3 = MaxPool1D(pool_size=2, strides=1, padding='same')(mfm2)

    # interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=2)(max3)
    # conv4_1 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    # conv4_2 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    # mfm5 = tf.keras.layers.maximum([conv4_1, conv4_2])
    # batch6 = BatchNormalization(axis=2, scale=False)(mfm5)

    # interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=4)(batch6)
    # conv7_1 = Conv1D(filters = 48, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    # conv7_2 = Conv1D(filters = 48, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    # mfm8 = tf.keras.layers.maximum([conv7_1, conv7_2])

    # max9 = MaxPool1D(pool_size=2, strides=1, padding='same')(mfm8)
    # batch10 = BatchNormalization(axis=2, scale=False)(max9)

    # interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=8)(batch10)
    # conv11_1 = Conv1D(filters = 48, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    # conv11_2 = Conv1D(filters = 48, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    # mfm12 = tf.keras.layers.maximum([conv11_1, conv11_2])
    # batch13 = BatchNormalization(axis=2, scale=False)(mfm12)

    # interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=16)(batch13)
    # conv14_1 = Conv1D(filters = 64, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    # conv14_2 = Conv1D(filters = 64, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    # mfm15 = tf.keras.layers.maximum([conv14_1, conv14_2])
    # max16 = MaxPool1D(pool_size=2, strides=1, padding='same')(mfm15)

    # interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=32)(max16)
    # conv17_1 = Conv1D(filters = 64, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    # conv17_2 = Conv1D(filters = 64, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    # mfm18 = tf.keras.layers.maximum([conv17_1, conv17_2])
    # batch19 = BatchNormalization(axis=2, scale=False)(mfm18)

    # interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=64)(batch19)
    # conv20_1 = Conv1D(filters = 32, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    # conv20_2 = Conv1D(filters = 32, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    # mfm21 = tf.keras.layers.maximum([conv20_1, conv20_2])
    # batch22 = BatchNormalization(axis=2, scale=False)(mfm21)

    # interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=128)(batch22)
    # conv23_1 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    # conv23_2 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    # mfm24 = tf.keras.layers.maximum([conv23_1, conv23_2])
    # batch25 = BatchNormalization(axis=2, scale=False)(mfm24)

    # interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=256)(batch25)
    # conv26_1 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    # conv26_2 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    # mfm27 = tf.keras.layers.maximum([conv26_1, conv26_2])


    # interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=1)(mfm277)
    # conv1_1 = Conv1D(filters = 32, kernel_size =5, strides=1, padding='same', activation=None)(interval1)
    # conv1_2 = Conv1D(filters = 32, kernel_size =5, strides=1, padding='same', activation=None)(interval1)
    # mfm2 = tf.keras.layers.maximum([conv1_1, conv1_2])
    # max3 = MaxPool1D(pool_size=2, strides=1, padding='same')(mfm2)

    # interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=2)(max3)
    # conv4_1 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    # conv4_2 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    # mfm5 = tf.keras.layers.maximum([conv4_1, conv4_2])
    # batch6 = BatchNormalization(axis=2, scale=False)(mfm5)

    # interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=4)(batch6)
    # conv7_1 = Conv1D(filters = 48, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    # conv7_2 = Conv1D(filters = 48, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    # mfm8 = tf.keras.layers.maximum([conv7_1, conv7_2])
    # max9 = MaxPool1D(pool_size=2, strides=1, padding='same')(mfm8)
    # batch10 = BatchNormalization(axis=2, scale=False)(max9)


    # interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=8)(batch10)
    # conv11_1 = Conv1D(filters = 48, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    # conv11_2 = Conv1D(filters = 48, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    # mfm12 = tf.keras.layers.maximum([conv11_1, conv11_2])
    # batch13 = BatchNormalization(axis=2, scale=False)(mfm12)

    # interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=8)(batch13)
    # conv14_1 = Conv1D(filters = 64, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    # conv14_2 = Conv1D(filters = 64, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    # mfm15 = tf.keras.layers.maximum([conv14_1, conv14_2])

    # max16 = MaxPool1D(pool_size=2, strides=1, padding='same')(mfm15)

    # interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=16)(max16)
    # conv17_1 = Conv1D(filters = 64, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    # conv17_2 = Conv1D(filters = 64, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    # mfm18 = tf.keras.layers.maximum([conv17_1, conv17_2])
    # batch19 = BatchNormalization(axis=2, scale=False)(mfm18)


    # interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=32)(batch19)
    # conv20_1 = Conv1D(filters = 32, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    # conv20_2 = Conv1D(filters = 32, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    # mfm21 = tf.keras.layers.maximum([conv20_1, conv20_2])
    # batch22 = BatchNormalization(axis=2, scale=False)(mfm21)


    # interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=64)(batch22)
    # conv23_1 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    # conv23_2 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    # mfm24 = tf.keras.layers.maximum([conv23_1, conv23_2])
    # batch25 = BatchNormalization(axis=2, scale=False)(mfm24)

    # interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=128)(batch25)
    # conv26_1 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    # conv26_2 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    # mfm27 = tf.keras.layers.maximum([conv26_1, conv26_2])
    
    
    # interval1 = layers.GlobalAveragePooling1D()(mfm27)
    # interval1 = Dropout(dp)(interval1)

    
    
    
   
    ## mel embedding
    if use_mel :
        
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        mfm2 = tf.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)

        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tf.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tf.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tf.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tf.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tf.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tf.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tf.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tf.keras.layers.maximum([conv26_1, conv26_2])
        
        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        mel2 = layers.GlobalAveragePooling2D()(max28)
        mel2 = Dropout(dp)(mel2)

    if use_cqt:
        ## cqt embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        mfm2 = tf.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tf.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tf.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tf.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tf.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tf.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tf.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tf.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tf.keras.layers.maximum([conv26_1, conv26_2])
        
        # max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        cqt2 = layers.GlobalAveragePooling2D()(mfm27)
        cqt2 = Dropout(dp)(cqt2)

    if use_stft :
        ## stft embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        mfm2 = tf.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tf.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)
        
        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tf.keras.layers.maximum([conv7_1, conv7_2])
        
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        
        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tf.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tf.keras.layers.maximum([conv14_1, conv14_2])
        
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tf.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tf.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tf.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tf.keras.layers.maximum([conv26_1, conv26_2])
        
        # max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        stft2 = layers.GlobalAveragePooling2D()(mfm27)
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
        
        concat1 = layers.Concatenate()([age1, sex1, hw1, loc1, preg,interval_mean1])
        d1 = layers.Dense(10, activation='relu')(concat1)
        concat2 = layers.Concatenate()([concat2, d1])
    
    if ext2 :
        
        concat2 = layers.Concatenate()([concat2, interval_mean1])
        
        
    if fc :
        concat2 = layers.Dense(10, activation = "relu")(concat2)
        concat2 = Dropout(dp)(concat2)
        
    # if ord1 :
    #     res1 = layers.Dense(2, activation = "softmax")(concat2)
        
        

        
    # else :
    #     res1 = layers.Dense(3, activation = "softmax")(concat2)

        
        
        
    res2 = layers.Dense(2, activation = "softmax")(concat2)

    model = keras.Model(inputs = [age,sex,hw,preg,loc,mel1,stft1,cqt1,interval_mean,interval] , outputs = res2 )
    
    model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy','AUC'])
    return(model)


##############################################################################################################
###############################################################################################################



def LCNN_seq_mean_jk_model_1(mel_input_shape, cqt_input_shape, stft_input_shape, interval_input_shape,use_mel = True, use_cqt = False, use_stft = False, ord1 = True, dp = .5, fc = False, ext = False, ext2 = False,use_interval_seq=False):
    
   
    
        # Create a towy model.
    age = keras.Input(shape=(6,), name = 'age_cat')
    sex = keras.Input(shape=(2,), name = 'sex_cat')
    hw = keras.Input(shape=(2,), name = 'height_weight')
    preg = keras.Input(shape=(1,), name = 'is_preg')
    loc = keras.Input(shape=(5,), name = 'loc')
    interval = keras.Input(shape=interval_input_shape, name = 'interval')
    interval_mean=keras.Input(shape=(1,), name = 'interval_mean')
    mel1 = keras.Input(shape=mel_input_shape, name = 'mel')
    cqt1 = keras.Input(shape=cqt_input_shape, name = 'cqt')
    stft1 = keras.Input(shape=stft_input_shape, name = 'stft')

    ## age embeddig
    age1 = layers.Dense(2, activation = None)(age)

    ## sex embedding
    sex1 = layers.Dense(1, activation = None)(sex)

    ## hw embedding
    hw1 = layers.Dense(1, activation = None)(hw)

    ## loc embedding
    loc1 = layers.Dense(3, activation = None)(loc)
    
    interval_mean1 = layers.Dense(1, activation=None)(interval_mean)
    
    ## interval embedding
    


    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=1)(interval)
    conv1_1 = Conv1D(filters = 32, kernel_size =5, strides=1, padding='same', activation=None)(interval1)
    conv1_2 = Conv1D(filters = 32, kernel_size =5, strides=1, padding='same', activation=None)(interval1)
    mfm2 = tf.keras.layers.maximum([conv1_1, conv1_2])
    max3 = MaxPool1D(pool_size=2, strides=1, padding='same')(mfm2)

    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=2)(max3)
    conv4_1 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    conv4_2 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    mfm5 = tf.keras.layers.maximum([conv4_1, conv4_2])
    batch6 = BatchNormalization(axis=2, scale=False)(mfm5)

    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=4)(batch6)
    conv7_1 = Conv1D(filters = 48, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    conv7_2 = Conv1D(filters = 48, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    mfm8 = tf.keras.layers.maximum([conv7_1, conv7_2])

    max9 = MaxPool1D(pool_size=2, strides=1, padding='same')(mfm8)
    batch10 = BatchNormalization(axis=2, scale=False)(max9)

    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=8)(batch10)
    conv11_1 = Conv1D(filters = 48, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    conv11_2 = Conv1D(filters = 48, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    mfm12 = tf.keras.layers.maximum([conv11_1, conv11_2])
    batch13 = BatchNormalization(axis=2, scale=False)(mfm12)

    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=16)(batch13)
    conv14_1 = Conv1D(filters = 64, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    conv14_2 = Conv1D(filters = 64, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    mfm15 = tf.keras.layers.maximum([conv14_1, conv14_2])
    max16 = MaxPool1D(pool_size=2, strides=1, padding='same')(mfm15)

    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=32)(max16)
    conv17_1 = Conv1D(filters = 64, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    conv17_2 = Conv1D(filters = 64, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    mfm18 = tf.keras.layers.maximum([conv17_1, conv17_2])
    batch19 = BatchNormalization(axis=2, scale=False)(mfm18)

    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=64)(batch19)
    conv20_1 = Conv1D(filters = 32, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    conv20_2 = Conv1D(filters = 32, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    mfm21 = tf.keras.layers.maximum([conv20_1, conv20_2])
    batch22 = BatchNormalization(axis=2, scale=False)(mfm21)

    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=128)(batch22)
    conv23_1 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    conv23_2 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    mfm24 = tf.keras.layers.maximum([conv23_1, conv23_2])
    batch25 = BatchNormalization(axis=2, scale=False)(mfm24)

    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=256)(batch25)
    conv26_1 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    conv26_2 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    mfm27 = tf.keras.layers.maximum([conv26_1, conv26_2])


    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=1)(mfm27)
    conv1_1 = Conv1D(filters = 32, kernel_size =5, strides=1, padding='same', activation=None)(interval1)
    conv1_2 = Conv1D(filters = 32, kernel_size =5, strides=1, padding='same', activation=None)(interval1)
    mfm2 = tf.keras.layers.maximum([conv1_1, conv1_2])
    max3 = MaxPool1D(pool_size=2, strides=1, padding='same')(mfm2)

    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=2)(max3)
    conv4_1 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    conv4_2 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    mfm5 = tf.keras.layers.maximum([conv4_1, conv4_2])
    batch6 = BatchNormalization(axis=2, scale=False)(mfm5)

    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=4)(batch6)
    conv7_1 = Conv1D(filters = 48, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    conv7_2 = Conv1D(filters = 48, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    mfm8 = tf.keras.layers.maximum([conv7_1, conv7_2])
    max9 = MaxPool1D(pool_size=2, strides=1, padding='same')(mfm8)
    batch10 = BatchNormalization(axis=2, scale=False)(max9)


    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=8)(batch10)
    conv11_1 = Conv1D(filters = 48, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    conv11_2 = Conv1D(filters = 48, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    mfm12 = tf.keras.layers.maximum([conv11_1, conv11_2])
    batch13 = BatchNormalization(axis=2, scale=False)(mfm12)

    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=8)(batch13)
    conv14_1 = Conv1D(filters = 64, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    conv14_2 = Conv1D(filters = 64, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    mfm15 = tf.keras.layers.maximum([conv14_1, conv14_2])

    max16 = MaxPool1D(pool_size=2, strides=1, padding='same')(mfm15)

    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=16)(max16)
    conv17_1 = Conv1D(filters = 64, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    conv17_2 = Conv1D(filters = 64, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    mfm18 = tf.keras.layers.maximum([conv17_1, conv17_2])
    batch19 = BatchNormalization(axis=2, scale=False)(mfm18)


    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=32)(batch19)
    conv20_1 = Conv1D(filters = 32, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    conv20_2 = Conv1D(filters = 32, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    mfm21 = tf.keras.layers.maximum([conv20_1, conv20_2])
    batch22 = BatchNormalization(axis=2, scale=False)(mfm21)


    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=64)(batch22)
    conv23_1 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    conv23_2 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    mfm24 = tf.keras.layers.maximum([conv23_1, conv23_2])
    batch25 = BatchNormalization(axis=2, scale=False)(mfm24)

    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=128)(batch25)
    conv26_1 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    conv26_2 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    mfm27 = tf.keras.layers.maximum([conv26_1, conv26_2])

    interval1 = layers.GlobalAveragePooling1D()(mfm27)
    interval1 = Dropout(dp)(interval1)
    
    if use_mel :
        
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        mfm2 = tf.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)

        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tf.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tf.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tf.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tf.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tf.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tf.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tf.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tf.keras.layers.maximum([conv26_1, conv26_2])
        
        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        mel2 = layers.GlobalAveragePooling2D()(max28)
        mel2 = Dropout(dp)(mel2)
      
    if use_cqt:
        ## cqt embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        mfm2 = tf.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tf.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tf.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tf.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tf.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tf.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tf.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tf.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tf.keras.layers.maximum([conv26_1, conv26_2])
        
        # max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        cqt2 = layers.GlobalAveragePooling2D()(mfm27)
        cqt2 = Dropout(dp)(cqt2)

    if use_stft :
        ## stft embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        mfm2 = tf.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tf.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)
        
        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tf.keras.layers.maximum([conv7_1, conv7_2])
        
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        
        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tf.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tf.keras.layers.maximum([conv14_1, conv14_2])
        
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tf.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tf.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tf.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tf.keras.layers.maximum([conv26_1, conv26_2])
        
        # max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        stft2 = layers.GlobalAveragePooling2D()(mfm27)
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
        
        concat1 = layers.Concatenate()([age1, sex1, hw1, loc1, preg,interval1,interval_mean1])
        d1 = layers.Dense(10, activation='relu')(concat1)
        concat2 = layers.Concatenate()([concat2, d1])
    
    if ext2 :
        
        concat2 = layers.Concatenate()([concat2, interval1,interval_mean1])
    
    
        
    if fc :
        concat2 = layers.Dense(10, activation = "relu")(concat2)
        concat2 = Dropout(dp)(concat2)
        
        
    if ord1 :
        res1 = layers.Dense(2, activation = "softmax")(concat2)
        
        

        
    else :
        res1 = layers.Dense(3, activation = "softmax")(concat2)

        
        
        
#     res2 = layers.Dense(2, activation = "softmax")(concat2)

    model = keras.Model(inputs = [age,sex,hw,preg,loc,mel1,cqt1,stft1, interval_mean,interval] , outputs = res1 )
    
    model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy','AUC'])
    return(model)



####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################

def LCNN_seq_mean_jk_model_2(mel_input_shape, cqt_input_shape, stft_input_shape, interval_input_shape,use_mel = True, use_cqt = False, use_stft = False,dp = .5, fc = False, ext = False, ext2 = False,use_interval_seq=False):
    
   
    
        # Create a towy model.
    
    
    age = keras.Input(shape=(6,), name = 'age_cat')
    sex = keras.Input(shape=(2,), name = 'sex_cat')
    hw = keras.Input(shape=(2,), name = 'height_weight')
    preg = keras.Input(shape=(1,), name = 'is_preg')
    loc = keras.Input(shape=(5,), name = 'loc')
    interval = keras.Input(shape=interval_input_shape, name = 'interval')
#     wav2 = keras.Input(shape=wav2_input_shape, name = 'wav2')
    mel1 = keras.Input(shape=mel_input_shape, name = 'mel')
    cqt1 = keras.Input(shape=cqt_input_shape, name = 'cqt')
    stft1 = keras.Input(shape=stft_input_shape, name = 'stft')
    interval_mean=keras.Input(shape=(1,), name = 'interval_mean')
        
    ## age embeddig
    age1 = layers.Dense(2, activation = None)(age)

    ## sex embedding
    sex1 = layers.Dense(1, activation = None)(sex)

    ## hw embedding
    hw1 = layers.Dense(1, activation = None)(hw)

    ## loc embedding
    loc1 = layers.Dense(3, activation = None)(loc)
    
    interval_mean1 = layers.Dense(1, activation=None)(interval_mean)
    
    
#     mel_mobile_1 = Conv2D(3,(3,3),padding='same')(mel_mo1)
#     mel_mobile_1 = MobileNet2(mel_mobile_1)
    
    
    
    ## interval embedding
    
    
    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=1)(interval)
    conv1_1 = Conv1D(filters = 32, kernel_size =5, strides=1, padding='same', activation=None)(interval1)
    conv1_2 = Conv1D(filters = 32, kernel_size =5, strides=1, padding='same', activation=None)(interval1)
    mfm2 = tf.keras.layers.maximum([conv1_1, conv1_2])
    max3 = MaxPool1D(pool_size=2, strides=1, padding='same')(mfm2)

    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=2)(max3)
    conv4_1 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    conv4_2 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    mfm5 = tf.keras.layers.maximum([conv4_1, conv4_2])
    batch6 = BatchNormalization(axis=2, scale=False)(mfm5)

    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=4)(batch6)
    conv7_1 = Conv1D(filters = 48, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    conv7_2 = Conv1D(filters = 48, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    mfm8 = tf.keras.layers.maximum([conv7_1, conv7_2])

    max9 = MaxPool1D(pool_size=2, strides=1, padding='same')(mfm8)
    batch10 = BatchNormalization(axis=2, scale=False)(max9)

    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=8)(batch10)
    conv11_1 = Conv1D(filters = 48, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    conv11_2 = Conv1D(filters = 48, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    mfm12 = tf.keras.layers.maximum([conv11_1, conv11_2])
    batch13 = BatchNormalization(axis=2, scale=False)(mfm12)

    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=16)(batch13)
    conv14_1 = Conv1D(filters = 64, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    conv14_2 = Conv1D(filters = 64, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    mfm15 = tf.keras.layers.maximum([conv14_1, conv14_2])
    max16 = MaxPool1D(pool_size=2, strides=1, padding='same')(mfm15)

    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=32)(max16)
    conv17_1 = Conv1D(filters = 64, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    conv17_2 = Conv1D(filters = 64, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    mfm18 = tf.keras.layers.maximum([conv17_1, conv17_2])
    batch19 = BatchNormalization(axis=2, scale=False)(mfm18)

    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=64)(batch19)
    conv20_1 = Conv1D(filters = 32, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    conv20_2 = Conv1D(filters = 32, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    mfm21 = tf.keras.layers.maximum([conv20_1, conv20_2])
    batch22 = BatchNormalization(axis=2, scale=False)(mfm21)

    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=128)(batch22)
    conv23_1 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    conv23_2 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    mfm24 = tf.keras.layers.maximum([conv23_1, conv23_2])
    batch25 = BatchNormalization(axis=2, scale=False)(mfm24)

    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=256)(batch25)
    conv26_1 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    conv26_2 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    mfm27 = tf.keras.layers.maximum([conv26_1, conv26_2])


    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=1)(mfm27)
    conv1_1 = Conv1D(filters = 32, kernel_size =5, strides=1, padding='same', activation=None)(interval1)
    conv1_2 = Conv1D(filters = 32, kernel_size =5, strides=1, padding='same', activation=None)(interval1)
    mfm2 = tf.keras.layers.maximum([conv1_1, conv1_2])
    max3 = MaxPool1D(pool_size=2, strides=1, padding='same')(mfm2)

    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=2)(max3)
    conv4_1 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    conv4_2 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    mfm5 = tf.keras.layers.maximum([conv4_1, conv4_2])
    batch6 = BatchNormalization(axis=2, scale=False)(mfm5)

    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=4)(batch6)
    conv7_1 = Conv1D(filters = 48, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    conv7_2 = Conv1D(filters = 48, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    mfm8 = tf.keras.layers.maximum([conv7_1, conv7_2])
    max9 = MaxPool1D(pool_size=2, strides=1, padding='same')(mfm8)
    batch10 = BatchNormalization(axis=2, scale=False)(max9)


    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=8)(batch10)
    conv11_1 = Conv1D(filters = 48, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    conv11_2 = Conv1D(filters = 48, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    mfm12 = tf.keras.layers.maximum([conv11_1, conv11_2])
    batch13 = BatchNormalization(axis=2, scale=False)(mfm12)

    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=8)(batch13)
    conv14_1 = Conv1D(filters = 64, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    conv14_2 = Conv1D(filters = 64, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    mfm15 = tf.keras.layers.maximum([conv14_1, conv14_2])

    max16 = MaxPool1D(pool_size=2, strides=1, padding='same')(mfm15)

    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=16)(max16)
    conv17_1 = Conv1D(filters = 64, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    conv17_2 = Conv1D(filters = 64, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    mfm18 = tf.keras.layers.maximum([conv17_1, conv17_2])
    batch19 = BatchNormalization(axis=2, scale=False)(mfm18)


    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=32)(batch19)
    conv20_1 = Conv1D(filters = 32, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    conv20_2 = Conv1D(filters = 32, kernel_size =3, strides=1, padding='same', activation=None)(interval1)
    mfm21 = tf.keras.layers.maximum([conv20_1, conv20_2])
    batch22 = BatchNormalization(axis=2, scale=False)(mfm21)


    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=64)(batch22)
    conv23_1 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    conv23_2 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    mfm24 = tf.keras.layers.maximum([conv23_1, conv23_2])
    batch25 = BatchNormalization(axis=2, scale=False)(mfm24)

    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=128)(batch25)
    conv26_1 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    conv26_2 = Conv1D(filters = 32, kernel_size =1, strides=1, padding='same', activation=None)(interval1)
    mfm27 = tf.keras.layers.maximum([conv26_1, conv26_2])
    
    
    interval1 = layers.GlobalAveragePooling1D()(mfm27)
    interval1 = Dropout(dp)(interval1)

    
    
    
   
    ## mel embedding
    if use_mel :
        
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(mel1)
        mfm2 = tf.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)

        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tf.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tf.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tf.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tf.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tf.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tf.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tf.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tf.keras.layers.maximum([conv26_1, conv26_2])
        
        max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        mel2 = layers.GlobalAveragePooling2D()(max28)
        mel2 = Dropout(dp)(mel2)

    if use_cqt:
        ## cqt embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(cqt1)
        mfm2 = tf.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tf.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)

        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tf.keras.layers.maximum([conv7_1, conv7_2])

        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)

        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tf.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tf.keras.layers.maximum([conv14_1, conv14_2])

        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tf.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tf.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tf.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tf.keras.layers.maximum([conv26_1, conv26_2])
        
        # max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        cqt2 = layers.GlobalAveragePooling2D()(mfm27)
        cqt2 = Dropout(dp)(cqt2)

    if use_stft :
        ## stft embedding
        conv1_1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        conv1_2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(stft1)
        mfm2 = tf.keras.layers.maximum([conv1_1, conv1_2])
        max3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm2)
        
        conv4_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        conv4_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max3)
        mfm5 = tf.keras.layers.maximum([conv4_1, conv4_2])
        batch6 = BatchNormalization(axis=3, scale=False)(mfm5)
        
        conv7_1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        conv7_2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch6)
        mfm8 = tf.keras.layers.maximum([conv7_1, conv7_2])
        
        max9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm8)
        batch10 = BatchNormalization(axis=3, scale=False)(max9)
        
        conv11_1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        conv11_2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch10)
        mfm12 = tf.keras.layers.maximum([conv11_1, conv11_2])
        batch13 = BatchNormalization(axis=3, scale=False)(mfm12)

        conv14_1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        conv14_2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch13)
        mfm15 = tf.keras.layers.maximum([conv14_1, conv14_2])
        
        max16 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(mfm15)

        conv17_1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        conv17_2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(max16)
        mfm18 = tf.keras.layers.maximum([conv17_1, conv17_2])
        batch19 = BatchNormalization(axis=3, scale=False)(mfm18)

        conv20_1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        conv20_2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(batch19)
        mfm21 = tf.keras.layers.maximum([conv20_1, conv20_2])
        batch22 = BatchNormalization(axis=3, scale=False)(mfm21)

        conv23_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        conv23_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch22)
        mfm24 = tf.keras.layers.maximum([conv23_1, conv23_2])
        batch25 = BatchNormalization(axis=3, scale=False)(mfm24)

        conv26_1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        conv26_2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(batch25)
        mfm27 = tf.keras.layers.maximum([conv26_1, conv26_2])
        
        # max28 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(mfm27)
        stft2 = layers.GlobalAveragePooling2D()(mfm27)
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
        
        concat1 = layers.Concatenate()([age1, sex1, hw1, loc1, preg,interval1,interval_mean1])
        d1 = layers.Dense(10, activation='relu')(concat1)
        concat2 = layers.Concatenate()([concat2, d1])
    
    if ext2 :
        
        concat2 = layers.Concatenate()([concat2, interval1,interval_mean1])
        
        
    if fc :
        concat2 = layers.Dense(10, activation = "relu")(concat2)
        concat2 = Dropout(dp)(concat2)
        
    # if ord1 :
    #     res1 = layers.Dense(2, activation = "softmax")(concat2)
        
        

        
    # else :
    #     res1 = layers.Dense(3, activation = "softmax")(concat2)

        
        
        
    res2 = layers.Dense(2, activation = "softmax")(concat2)

    model = keras.Model(inputs = [age,sex,hw,preg,loc,mel1,cqt1,stft1, interval_mean,interval] , outputs = res2 )
    
    model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy','AUC'])
    return(model)






