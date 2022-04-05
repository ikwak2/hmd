from tensorflow import keras
from tensorflow.keras import layers

def get_toy(mel_input_shape):
        # Create a towy model.
    age = keras.Input(shape=(6,), name = 'age_cat')
    sex = keras.Input(shape=(2,), name = 'sex_cat')
    hw = keras.Input(shape=(2,), name = 'height_weight')
    preg = keras.Input(shape=(1,), name = 'is_preg')
    loc = keras.Input(shape=(5,), name = 'loc')
    mel1 = keras.Input(shape=(mel_input_shape), name = 'mel')
        
    ## age embeddig
    age1 = layers.Dense(2, activation = 'relu')(age)

    ## sex embedding
    sex1 = layers.Dense(1, activation = 'relu')(sex)

    ## hw embedding
    hw1 = layers.Dense(1, activation = 'relu')(hw)

    ## loc embedding
    loc1 = layers.Dense(3, activation = 'relu')(loc)

    ## mel embedding
    mel2 = layers.Conv2D(16, (3,3), activation = 'relu')(mel1)
    mel2 = layers.MaxPooling2D()(mel2)
    mel2 = layers.Conv2D(32, (5,5), activation = 'relu')(mel2)
    mel2 = layers.MaxPooling2D()(mel2)
    mel2 = layers.Conv2D(32, (3,3), activation = 'relu')(mel2)
    mel2 = layers.MaxPooling2D()(mel2)
    mel2 = layers.Conv2D(64, (3,3), activation = 'relu')(mel2)
    mel2 = layers.MaxPooling2D()(mel2)
    mel2 = layers.GlobalAveragePooling2D()(mel2)

    concat1 = layers.Concatenate()([age1, sex1, hw1, loc1, mel2, preg])
    concat1 = layers.Dense(10, activation = 'relu')(concat1)
    concat1 = layers.Dense(3, activation = "softmax")(concat1)
    model = keras.Model(inputs = [age,sex,hw,preg,loc,mel1] , outputs = concat1 )
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return(model)