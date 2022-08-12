from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, maximum, DepthwiseConv2D, AveragePooling2D, UpSampling2D
from tensorflow.keras.layers import Input, Conv1D,Conv2D, MaxPooling2D,MaxPooling1D, BatchNormalization, SpatialDropout2D
from tensorflow.keras.layers import Convolution2D, GlobalAveragePooling2D, MaxPool2D, ZeroPadding2D,MaxPool1D,GlobalAveragePooling1D
from tensorflow.keras.layers import add,concatenate
from tensorflow.keras.activations import relu, softmax, swish
import tensorflow
import tensorflow as tf
import tensorflow as tf
import tensorflow_hub as hub





def get_LCNN_o_4_dr(mel_input_shape, cqt_input_shape, stft_input_shape,interval_input_shape,wav2_input_shape,
                    use_mel = True, use_cqt = False, use_stft = False, ord1 = True, dp = .5, fc = False, ext = False,use_wav2=False):
    
    
    MobileNet2 = tf.keras.applications.MobileNetV2(
    input_shape=(128,128,3),
    alpha=1.0,
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    pooling='avg',
    classes=3,
    classifier_activation="softmax")
    
        # Create a towy model.
    age = keras.Input(shape=(6,), name = 'age_cat')
    sex = keras.Input(shape=(2,), name = 'sex_cat')
    hw = keras.Input(shape=(2,), name = 'height_weight')
    preg = keras.Input(shape=(1,), name = 'is_preg')
    loc = keras.Input(shape=(5,), name = 'loc')
    interval = keras.Input(shape=interval_input_shape, name = 'interval')
    wav2 = keras.Input(shape=wav2_input_shape, name = 'wav2')
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
    
    ## interval embedding
    
#     interval1 = tf.keras.layers.LSTM(20,return_sequences=True,use_bias=True, 
#                                 dropout=0.1,recurrent_dropout=0.05)(interval)
#     interval1 = tf.keras.layers.LSTM(15,return_sequences=False,use_bias=True, dropout=0.3,
#                                 recurrent_dropout=0.05)(interval1)
#     interval1 = layers.Dense(15, activation='relu')(interval1)
#     interval1 = tf.keras.layers.Conv1D(20, 3, activation='relu')(interval)
#     interval1 = tf.keras.layers.Conv1D(10, 3, activation='relu')(interval1)
#     interval1 = tf.keras.layers.GlobalAveragePooling1D()(interval1)

    
#     interval1 = tf.keras.layers.Conv1D(9, 3, activation='relu')(interval)
#     interval1 = tf.keras.layers.Conv1D(9, 3, activation='relu')(interval1)
#     interval1 = tf.keras.layers.Conv1D(7, 3, activation='relu')(interval1)
#     interval1 = tf.keras.layers.Conv1D(7, 3, activation='relu')(interval1)
#     interval1 = tf.keras.layers.GlobalAveragePooling1D()(interval1)
#     interval1 = layers.Dense(3, activation='relu')(interval1)
    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=1)(interval)
    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=2)(interval1)
    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=4)(interval1)
    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=8)(interval1)
    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=16)(interval1)
    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=32)(interval1)
    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=64)(interval1)
    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=128)(interval1)
    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=256)(interval1)
    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=512)(interval1)
    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=1024)(interval1)

#     interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=1)(interval1)
#     interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=2)(interval1)
#     interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=4)(interval1)
#     interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=8)(interval1)
#     interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=16)(interval1)
#     interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=32)(interval1)
#     interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=64)(interval1)
#     interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=128)(interval1)
#     interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=256)(interval1)
#     interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=512)(interval1)
#     interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=1024)(interval1)    

    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=1)(interval1)
    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=2)(interval1)
    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=4)(interval1)
    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=8)(interval1)
    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=16)(interval1)
    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=32)(interval1)
    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=64)(interval1)
    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=128)(interval1)
    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=256)(interval1)
    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=512)(interval1)
    interval1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=1024)(interval1)
    
    interval1 = tf.keras.layers.GlobalMaxPooling1D()(interval1)
    
    ## wav2 embedding
    
    
    #1))))
    
#     wav2_1 = tf.keras.layers.Conv1D(9, 3, activation='relu')(wav2)
#     wav2_1 = tf.keras.layers.Conv1D(9, 3, activation='relu')(wav2_1)
#     wav2_1 = tf.keras.layers.Conv1D(7, 3, activation='relu')(wav2_1)
#     wav2_1 = tf.keras.layers.Conv1D(7, 3, activation='relu')(wav2_1)
#     wav2_1 = tf.keras.layers.GlobalAveragePooling1D()(wav2_1)
#     wav2_1 = layers.Dense(3, activation='relu')(wav2_1)

    #2))))
    if use_wav2:
        
        wav2_1 = Conv2D(3,(3,3),padding='same')(wav2)
        wav2_2 = MobileNet2(wav2_1)

    
    
#     if use_raw:
#         raw1_reshape = tf.keras.layers.Reshape((600,-1))(raw1)
#         raw1_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=1)(raw1_reshape)
#         raw1_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=2)(raw1_1)
#         raw1_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=4)(raw1_1)
#         raw1_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=8)(raw1_1)
#         raw1_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=16)(raw1_1)
#         raw1_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=32)(raw1_1)
#         raw1_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=64)(raw1_1)
#         raw1_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=128)(raw1_1)
#         raw1_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=256)(raw1_1)
#         raw1_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=512)(raw1_1)
#         raw1_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=1024)(raw1_1)

#         raw1_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=1)(raw1_1)
#         raw1_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=2)(raw1_1)
#         raw1_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=4)(raw1_1)
#         raw1_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=8)(raw1_1)
#         raw1_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=16)(raw1_1)
#         raw1_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=32)(raw1_1)
#         raw1_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=64)(raw1_1)
#         raw1_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=128)(raw1_1)
#         raw1_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=256)(raw1_1)
#         raw1_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=512)(raw1_1)
#         raw1_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=1024)(raw1_1)

#         raw1_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=1)(raw1_1)
#         raw1_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=2)(raw1_1)
#         raw1_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=4)(raw1_1)
#         raw1_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=8)(raw1_1)
#         raw1_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=16)(raw1_1)
#         raw1_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=32)(raw1_1)
#         raw1_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=64)(raw1_1)
#         raw1_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=128)(raw1_1)
#         raw1_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=256)(raw1_1)
#         raw1_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=512)(raw1_1)
#         raw1_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=1024)(raw1_1)
        
#         raw1_1 = tf.keras.layers.GlobalMaxPooling1D()(raw1_1)
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
        mel2 = layers.GlobalAveragePooling2D()(max28)
        mel2 = Dropout(dp)(mel2)

    if use_cqt:
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
        
        concat1 = layers.Concatenate()([age1, sex1, hw1, loc1, interval1,preg])
        d1 = layers.Dense(10, activation='relu')(concat1)
        concat2 = layers.Concatenate()([concat2, d1,wav2_2])
        
    if fc :
        concat2 = layers.Dense(10, activation = "relu")(concat2)
        concat2 = Dropout(dp)(concat2)
        
    if ord1 :
        res1 = layers.Dense(2, activation = "softmax")(concat2)
        
        

        
    else :
        res1 = layers.Dense(3, activation = "softmax")(concat2)

        
        
        
#     res2 = layers.Dense(2, activation = "softmax")(concat2)

    model = keras.Model(inputs = [age,sex,hw,preg,loc,interval,wav2,mel1,stft1,cqt1] , outputs = res1 )
    
    model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy','AUC'])
    return(model)



def get_LCNN_o_4_dr_1(mel_input_shape, cqt_input_shape, stft_input_shape,interval_input_shape,wav2_input_shape,raw1_input_shape,
                      use_mel = True, use_cqt = False, use_stft = False, use_raw=False, 
                      use_wav2=False, ord1 = True, dp = .5, fc = False, ext = False):
    
    # Create a towy model.
    age = keras.Input(shape=(6,), name = 'age_cat')
    sex = keras.Input(shape=(2,), name = 'sex_cat')
    hw = keras.Input(shape=(2,), name = 'height_weight')
    preg = keras.Input(shape=(1,), name = 'is_preg')
    loc = keras.Input(shape=(5,), name = 'loc')
    interval = keras.Input(shape=interval_input_shape, name = 'interval')
    raw1 = keras.Input(shape=raw1_input_shape, name = 'raw1')
    wav2 = keras.Input(shape=wav2_input_shape, name = 'wav2')
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
    
    ## interval embedding
    

    interval1_1 = tf.keras.layers.Conv1D(9, 3, padding="causal",activation='relu',dilation_rate=1)(interval)
    interval1_2 = tf.keras.layers.Conv1D(9, 3, padding="causal",activation='relu',dilation_rate=1)(interval)
    interval_mfm1 = tensorflow.keras.layers.maximum([interval1_1, interval1_2])
    interval_maxpool_1 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(interval_mfm1)

        
    interval2_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=2)(interval_maxpool_1)
    interval2_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=2)(interval_maxpool_1)
    interval_mfm2 = tensorflow.keras.layers.maximum([interval2_1, interval2_2])
    interval_batch_1 = BatchNormalization(axis=2, scale=False)(interval_mfm2)
    
    
    interval3_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=4)(interval_batch_1)
    interval3_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=4)(interval_batch_1)
    interval_mfm3 = tensorflow.keras.layers.maximum([interval3_1, interval3_2])
    interval_maxpool_3 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(interval_mfm3)
    
    interval4_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=8)(interval_maxpool_3)
    interval4_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=8)(interval_maxpool_3)
    interval_mfm4 = tensorflow.keras.layers.maximum([interval4_1, interval4_2])
    interval_batch_2 = BatchNormalization(axis=2, scale=False)(interval_mfm4)
    
    interval5_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=16)(interval_batch_2)
    interval5_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=16)(interval_batch_2)
    interval_mfm5 = tensorflow.keras.layers.maximum([interval5_1, interval5_1])
    interval_maxpool_5 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(interval_mfm5)
    
    interval6_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=32)(interval_maxpool_5)
    interval6_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=32)(interval_maxpool_5)
    interval_mfm6 = tensorflow.keras.layers.maximum([interval6_1, interval6_2])
    interval_batch_3 = BatchNormalization(axis=2, scale=False)(interval_mfm6)
    
    interval7_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=64)(interval_batch_3)
    interval7_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=64)(interval_batch_3)
    interval_mfm7 = tensorflow.keras.layers.maximum([interval7_1, interval7_2])
    interval_maxpool_7 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(interval_mfm7)
    
    interval8_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=128)(interval_maxpool_7)
    interval8_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=128)(interval_maxpool_7)
    interval_mfm8 = tensorflow.keras.layers.maximum([interval8_1, interval8_2])
    interval_batch_4 = BatchNormalization(axis=2, scale=False)(interval_mfm8)
    
    interval9_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=256)(interval_batch_4)
    interval9_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=256)(interval_batch_4)
    interval_mfm9 = tensorflow.keras.layers.maximum([interval9_1, interval9_2])
    interval_maxpool_9 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(interval_mfm9)
    
    interval10_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=512)(interval_maxpool_9)
    interval10_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=512)(interval_maxpool_9)
    interval_mfm10 = tensorflow.keras.layers.maximum([interval10_1, interval10_2])
    interval_maxpool_10 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(interval_mfm10)
    
    
#     interval11_1 = tf.keras.layers.Conv1D(9, 3, padding="causal",activation=None,dilation_rate=1)(interval_maxpool_10)
#     interval11_2 = tf.keras.layers.Conv1D(9, 3, padding="causal",activation=None,dilation_rate=1)(interval_maxpool_10)
#     interval_mfm11 = tensorflow.keras.layers.maximum([interval11_1, interval11_2])
#     interval_maxpool_11 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(interval_mfm11)

    
    
#     interval12_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=2)(interval_maxpool_11)
#     interval12_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=2)(interval_maxpool_11)
#     interval_mfm12 = tensorflow.keras.layers.maximum([interval12_1, interval12_2])
#     interval_maxpool_12 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(interval_mfm12)
    
    
#     interval13_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=4)(interval_maxpool_12)
#     interval13_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=4)(interval_maxpool_12)
#     interval_mfm13 = tensorflow.keras.layers.maximum([interval13_1, interval13_2])
#     interval_maxpool_13 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(interval_mfm13)
    
#     interval14_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=8)(interval_maxpool_13)
#     interval14_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=8)(interval_maxpool_13)
#     interval_mfm14 = tensorflow.keras.layers.maximum([interval14_1, interval14_2])
#     interval_maxpool_14 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(interval_mfm14)
    
#     interval15_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=16)(interval_maxpool_14)
#     interval15_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=16)(interval_maxpool_14)
#     interval_mfm15 = tensorflow.keras.layers.maximum([interval15_1, interval15_2])
#     interval_maxpool_15 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(interval_mfm15)
    
#     interval16_1 = tf.keras.layers.Conv1D(64, 2, padding="causal",activation=None,dilation_rate=32)(interval_maxpool_15)
#     interval16_2 = tf.keras.layers.Conv1D(64, 2, padding="causal",activation=None,dilation_rate=32)(interval_maxpool_15)
#     interval_mfm16 = tensorflow.keras.layers.maximum([interval16_1, interval16_2])
#     interval_maxpool_16 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(interval_mfm16)
    
#     interval17_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=64)(interval_maxpool_16)
#     interval17_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=64)(interval_maxpool_16)
#     interval_mfm17 = tensorflow.keras.layers.maximum([interval17_1, interval17_2])
#     interval_maxpool_17 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(interval_mfm17)
    
#     interval18_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=128)(interval_maxpool_17)
#     interval18_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=128)(interval_maxpool_17)
#     interval_mfm18 = tensorflow.keras.layers.maximum([interval18_1, interval18_2])
#     interval_maxpool_18 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(interval_mfm18)
    
#     interval19_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=256)(interval_maxpool_18)
#     interval19_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=256)(interval_maxpool_18)
#     interval_mfm19 = tensorflow.keras.layers.maximum([interval19_1, interval19_2])
#     interval_maxpool_19 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(interval_mfm19)
    
#     interval20_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=512)(interval_maxpool_19)
#     interval20_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation=None,dilation_rate=512)(interval_maxpool_19)
#     interval_mfm20 = tensorflow.keras.layers.maximum([interval20_1, interval20_2])
#     interval_maxpool_20 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(interval_mfm20)    
    
    interval1 = layers.GlobalAveragePooling1D()(interval_maxpool_10)
    interval1 = Dropout(dp)(interval1)
   

    if use_raw:
        
        raw1_reshape = tf.keras.layers.Reshape((300,-1))(raw1)
        raw1_1_1 = tf.keras.layers.Conv1D(9, 3, padding="causal",activation='relu',dilation_rate=1)(raw1_reshape)
        raw1_1_2 = tf.keras.layers.Conv1D(9, 3, padding="causal",activation='relu',dilation_rate=1)(raw1_reshape)
        raw1_mfm1 = tensorflow.keras.layers.maximum([raw1_1_1, raw1_1_2])
        raw1_maxpool_1 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(raw1_mfm1)



        raw1_2_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=2)(raw1_maxpool_1)
        raw1_2_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=2)(raw1_maxpool_1)
        raw1_mfm2 = tensorflow.keras.layers.maximum([raw1_2_1, raw1_2_2])
        raw1_batch_1 = BatchNormalization(axis=2, scale=False)(raw1_mfm2)


        raw1_3_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=4)(raw1_batch_1)
        raw1_3_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=4)(raw1_batch_1)
        raw1_mfm3 = tensorflow.keras.layers.maximum([raw1_3_1, raw1_3_2])
        raw1_maxpool_3 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(raw1_mfm3)

        raw1_4_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=8)(raw1_maxpool_3)
        raw1_4_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=8)(raw1_maxpool_3)
        raw1_mfm4 = tensorflow.keras.layers.maximum([raw1_4_1, raw1_4_2])
        raw1_batch_2 = BatchNormalization(axis=2, scale=False)(raw1_mfm4)

        raw1_5_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=16)(raw1_batch_2)
        raw1_5_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=16)(raw1_batch_2)
        raw1_mfm5 = tensorflow.keras.layers.maximum([raw1_5_1, raw1_5_2])
        raw1_maxpool_5 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(raw1_mfm5)

        raw1_6_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=32)(raw1_maxpool_5)
        raw1_6_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=32)(raw1_maxpool_5)
        raw1_mfm6 = tensorflow.keras.layers.maximum([raw1_6_1, raw1_6_2])
        raw1_batch_3 = BatchNormalization(axis=2, scale=False)(raw1_mfm6)

        raw1_7_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=64)(raw1_batch_3)
        raw1_7_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=64)(raw1_batch_3)
        raw1_mfm7 = tensorflow.keras.layers.maximum([raw1_7_1, raw1_7_2])
        raw1_maxpool_7 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(raw1_mfm7)

        raw1_8_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=128)(raw1_maxpool_7)
        raw1_8_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=128)(raw1_maxpool_7)
        raw1_mfm8 = tensorflow.keras.layers.maximum([raw1_8_1, raw1_8_2])
        raw1_batch_4 = BatchNormalization(axis=2, scale=False)(raw1_mfm8)

        raw1_9_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=256)(raw1_batch_4)
        raw1_9_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=256)(raw1_batch_4)
        raw1_mfm9 = tensorflow.keras.layers.maximum([raw1_9_1, raw1_9_2])
        raw1_maxpool_9 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(raw1_mfm9)

        raw1_10_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=512)(raw1_maxpool_9)
        raw1_10_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=512)(raw1_maxpool_9)
        raw1_mfm10 = tensorflow.keras.layers.maximum([raw1_10_1, raw1_10_2])
        raw1_batch_5 = BatchNormalization(axis=2, scale=False)(raw1_mfm10)




        raw1_11_1 = tf.keras.layers.Conv1D(9, 3, padding="causal",activation='relu',dilation_rate=1)(raw1_batch_5)
        raw1_11_2 = tf.keras.layers.Conv1D(9, 3, padding="causal",activation='relu',dilation_rate=1)(raw1_batch_5)
        raw1_mfm11 = tensorflow.keras.layers.maximum([raw1_11_1, raw1_11_2])
        raw1_maxpool_11 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(raw1_mfm11)



        raw1_12_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=2)(raw1_maxpool_11)
        raw1_12_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=2)(raw1_maxpool_11)
        raw1_mfm12 = tensorflow.keras.layers.maximum([raw1_12_1, raw1_12_2])
        raw1_batch_6 = BatchNormalization(axis=2, scale=False)(raw1_mfm12)


        raw1_13_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=4)(raw1_batch_6)
        raw1_13_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=4)(raw1_batch_6)
        raw1_mfm13 = tensorflow.keras.layers.maximum([raw1_13_1, raw1_13_2])
        raw1_maxpool_13 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(raw1_mfm13)

        raw1_14_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=8)(raw1_maxpool_13)
        raw1_14_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=8)(raw1_maxpool_13)
        raw1_mfm14 = tensorflow.keras.layers.maximum([raw1_14_1, raw1_14_2])
        raw1_batch_7 = BatchNormalization(axis=2, scale=False)(raw1_mfm14)

        raw1_15_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=16)(raw1_batch_7)
        raw1_15_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=16)(raw1_batch_7)
        raw1_mfm15 = tensorflow.keras.layers.maximum([raw1_15_1, raw1_15_2])
        raw1_maxpool_15 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(raw1_mfm15)

        raw1_16_1 = tf.keras.layers.Conv1D(64, 2, padding="causal",activation='relu',dilation_rate=32)(raw1_maxpool_15)
        raw1_16_2 = tf.keras.layers.Conv1D(64, 2, padding="causal",activation='relu',dilation_rate=32)(raw1_maxpool_15)
        raw1_mfm16 = tensorflow.keras.layers.maximum([raw1_16_1, raw1_16_2])
        raw1_batch_8 = BatchNormalization(axis=2, scale=False)(raw1_mfm16)

        raw1_17_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=64)(raw1_batch_8)
        raw1_17_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=64)(raw1_batch_8)
        raw1_mfm17 = tensorflow.keras.layers.maximum([raw1_17_1, raw1_17_2])
        raw1_maxpool_17 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(raw1_mfm17)

        raw1_18_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=128)(raw1_maxpool_17)
        raw1_18_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=128)(raw1_maxpool_17)
        raw1_mfm18 = tensorflow.keras.layers.maximum([raw1_18_1, raw1_18_2])
        raw1_batch_8 = BatchNormalization(axis=2, scale=False)(raw1_mfm18)

        raw1_19_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=256)(raw1_batch_8)
        raw1_19_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=256)(raw1_batch_8)
        raw1_mfm19 = tensorflow.keras.layers.maximum([raw1_19_1, raw1_19_2])
        raw1_maxpool_19 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(raw1_mfm19)

        raw1_20_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=512)(raw1_maxpool_19)
        raw1_20_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=512)(raw1_maxpool_19)
        raw1_mfm20 = tensorflow.keras.layers.maximum([raw1_20_1, raw1_20_2])
        raw1_maxpool_20 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(raw1_mfm20)    

        raw1_1 = layers.GlobalAveragePooling1D()(raw1_maxpool_20)
        raw1_1 = Dropout(dp)(raw1_1)
    #4))))))))00
    
    
    if use_wav2:
        
        
 
        wav2_1_1 = tf.keras.layers.Conv1D(9, 3, padding="causal",activation='relu',dilation_rate=1)(wav2)
        wav2_1_2 = tf.keras.layers.Conv1D(9, 3, padding="causal",activation='relu',dilation_rate=1)(wav2)
        wav2_mfm1 = tensorflow.keras.layers.maximum([wav2_1_1, wav2_1_2])
        wav2_maxpool_1 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(wav2_mfm1)



        wav2_2_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=2)(wav2_maxpool_1)
        wav2_2_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=2)(wav2_maxpool_1)
        wav2_mfm2 = tensorflow.keras.layers.maximum([wav2_2_1, wav2_2_2])
        wav2_batch_1 = BatchNormalization(axis=2, scale=False)(wav2_mfm2)


        wav2_3_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=4)(wav2_batch_1)
        wav2_3_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=4)(wav2_batch_1)
        wav2_mfm3 = tensorflow.keras.layers.maximum([wav2_3_1, wav2_3_2])
        wav2_maxpool_3 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(wav2_mfm3)

        wav2_4_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=8)(wav2_maxpool_3)
        wav2_4_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=8)(wav2_maxpool_3)
        wav2_mfm4 = tensorflow.keras.layers.maximum([wav2_4_1, wav2_4_2])
        wav2_batch_2 = BatchNormalization(axis=2, scale=False)(wav2_mfm4)

        wav2_5_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=16)(wav2_batch_2)
        wav2_5_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=16)(wav2_batch_2)
        wav2_mfm5 = tensorflow.keras.layers.maximum([wav2_5_1, wav2_5_2])
        wav2_maxpool_5 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(wav2_mfm5)

        wav2_6_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=32)(wav2_maxpool_5)
        wav2_6_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=32)(wav2_maxpool_5)
        wav2_mfm6 = tensorflow.keras.layers.maximum([wav2_6_1, wav2_6_2])
        wav2_batch_3 = BatchNormalization(axis=2, scale=False)(wav2_mfm6)

        wav2_7_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=64)(wav2_batch_3)
        wav2_7_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=64)(wav2_batch_3)
        wav2_mfm7 = tensorflow.keras.layers.maximum([wav2_7_1, wav2_7_2])
        wav2_maxpool_7 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(wav2_mfm7)

        wav2_8_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=128)(wav2_maxpool_7)
        wav2_8_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=128)(wav2_maxpool_7)
        wav2_mfm8 = tensorflow.keras.layers.maximum([wav2_8_1, wav2_8_2])
        wav2_batch_4 = BatchNormalization(axis=2, scale=False)(wav2_mfm8)

        wav2_9_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=256)(wav2_batch_4)
        wav2_9_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=256)(wav2_batch_4)
        wav2_mfm9 = tensorflow.keras.layers.maximum([wav2_9_1, wav2_9_2])
        wav2_maxpool_9 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(wav2_mfm9)

        wav2_10_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=512)(wav2_maxpool_9)
        wav2_10_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=512)(wav2_maxpool_9)
        wav2_mfm10 = tensorflow.keras.layers.maximum([wav2_10_1, wav2_10_2])
        wav2_batch_5 = BatchNormalization(axis=2, scale=False)(wav2_mfm10)

        
        
        wav2_11_1 = tf.keras.layers.Conv1D(9, 3, padding="causal",activation='relu',dilation_rate=1)(wav2_batch_5)
        wav2_11_2 = tf.keras.layers.Conv1D(9, 3, padding="causal",activation='relu',dilation_rate=1)(wav2_batch_5)
        wav2_mfm11 = tensorflow.keras.layers.maximum([wav2_11_1, wav2_11_2])
        wav2_maxpool_11 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(wav2_mfm11)



        wav2_12_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=2)(wav2_maxpool_11)
        wav2_12_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=2)(wav2_maxpool_11)
        wav2_mfm12 = tensorflow.keras.layers.maximum([wav2_12_1, wav2_12_2])
        wav2_batch_6 = BatchNormalization(axis=2, scale=False)(wav2_mfm12)


        wav2_13_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=4)(wav2_batch_6)
        wav2_13_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=4)(wav2_batch_6)
        wav2_mfm13 = tensorflow.keras.layers.maximum([wav2_13_1, wav2_13_2])
        wav2_maxpool_13 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(wav2_mfm13)

        wav2_14_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=8)(wav2_maxpool_13)
        wav2_14_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=8)(wav2_maxpool_13)
        wav2_mfm14 = tensorflow.keras.layers.maximum([wav2_14_1, wav2_14_2])
        wav2_batch_7 = BatchNormalization(axis=2, scale=False)(wav2_mfm14)

        wav2_15_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=16)(wav2_batch_7)
        wav2_15_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=16)(wav2_batch_7)
        wav2_mfm15 = tensorflow.keras.layers.maximum([wav2_15_1, wav2_15_2])
        wav2_maxpool_15 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(wav2_mfm15)

        wav2_16_1 = tf.keras.layers.Conv1D(64, 2, padding="causal",activation='relu',dilation_rate=32)(wav2_maxpool_15)
        wav2_16_2 = tf.keras.layers.Conv1D(64, 2, padding="causal",activation='relu',dilation_rate=32)(wav2_maxpool_15)
        wav2_mfm16 = tensorflow.keras.layers.maximum([wav2_16_1, wav2_16_2])
        wav2_batch_8 = BatchNormalization(axis=2, scale=False)(wav2_mfm16)

        wav2_17_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=64)(wav2_batch_8)
        wav2_17_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=64)(wav2_batch_8)
        wav2_mfm17 = tensorflow.keras.layers.maximum([wav2_17_1, wav2_17_2])
        wav2_maxpool_17 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(wav2_mfm17)

        wav2_18_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=128)(wav2_maxpool_17)
        wav2_18_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=128)(wav2_maxpool_17)
        wav2_mfm18 = tensorflow.keras.layers.maximum([wav2_18_1, wav2_18_2])
        wav2_batch_9 = BatchNormalization(axis=2, scale=False)(wav2_mfm18)

        wav2_19_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=256)(wav2_batch_9)
        wav2_19_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=256)(wav2_batch_9)
        wav2_mfm19 = tensorflow.keras.layers.maximum([wav2_19_1, wav2_19_2])
        wav2_maxpool_19 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(wav2_mfm19)

        wav2_20_1 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=512)(wav2_maxpool_19)
        wav2_20_2 = tf.keras.layers.Conv1D(9, 2, padding="causal",activation='relu',dilation_rate=512)(wav2_maxpool_19)
        wav2_mfm20 = tensorflow.keras.layers.maximum([wav2_20_1, wav2_20_2])
        wav2_maxpool_20 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')(wav2_mfm20)  
        
                        
        wav2_1 = layers.GlobalAveragePooling1D()(wav2_maxpool_20)
        wav2_1 = Dropout(dp)(wav2_1)
        
        
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
        mel2 = layers.GlobalAveragePooling2D()(max28)
        mel2 = Dropout(dp)(mel2)

    if use_cqt:
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
        
        concat1 = layers.Concatenate()([age1, sex1, hw1, loc1,preg,interval1])
        d1 = layers.Dense(10, activation='relu')(concat1)
        concat2 = layers.Concatenate()([concat2, d1,wav2_1,raw1_1])
        
    if fc :
        concat2 = layers.Dense(10, activation = "relu")(concat2)
        concat2 = Dropout(dp)(concat2)
        
    if ord1 :
        res1 = layers.Dense(2, activation = "softmax")(concat2)
        

    else :
        res1 = layers.Dense(3, activation = "softmax")(concat2)

        
        
        
#     res2 = layers.Dense(2, activation = "softmax")(concat2)

    model = keras.Model(inputs = [age,sex,hw,preg,loc,raw1,interval,wav2,mel1,stft1,cqt1] , outputs = res1 )
    
    model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy','AUC'])
    return(model)



