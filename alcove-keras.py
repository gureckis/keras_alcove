import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow_addons as tfa
import seaborn as sns
import pandas as pd



stims = np.array([[0,0,0],
         [0,0,1],
         [0,1,0],
         [0,1,1],
         [1,0,0],
         [1,0,1],
         [1,1,0],
         [1,1,1]], dtype=np.float32)

label_codes = {'A': [-1.0, 1.0], #A
               'B': [1.0, -1.0]} #B

def convert_labels(x):
    return [label_codes[i] for i in x]

labels_abs = [
    ['A','A','A','A','B','B','B','B'], # type 1
    ['A','A','B','B','B','B','A','A'], # type 2
    ['B','B','B','A','A','B','A','A'], # type 3
    ['B','B','B','A','B','A','A','A'], # type 4
    ['B','B','B','A','A','A','A','B'], # type 5
    ['B','A','A','B','A','B','B','A']  # type 6
]

labels = np.array(list(map(convert_labels, labels_abs)), dtype=np.float32)


class ALCOVE_RBF(keras.layers.Layer):

    def __init__(self, exemplars, c=6.5, **kwargs):
        super().__init__(**kwargs)
        self.exemplars = tf.dtypes.cast(exemplars, tf.float32)
        self.ne = exemplars.shape[0]
        self.ndims = exemplars.shape[1]
        self.c = c

    def get_attention(self):
        return self.attn.numpy()
    
    def build(self, input_shape):
        self.ndims = input_shape[-1]
        self.attn = tf.Variable(np.ones((self.ndims ,))/float(self.ndims), 
                                dtype=tf.float32, 
                                trainable=True,
                                constraint=tf.keras.constraints.NonNeg())

    def call(self, inputs):
        h_acts = []
        def activation(input_pat):
            return tf.math.exp(-self.c*tf.reduce_sum(self.attn*tf.math.abs(self.exemplars - input_pat),1))
        h_acts=tf.map_fn(lambda x: activation(x), inputs)
        output = tf.stack(h_acts)
        return output


problem = 1
model = keras.Sequential(name="ALCOVE")
model.add(keras.Input(shape=(3,)))  
model.add(ALCOVE_RBF(stims, name="exemplar_layer"))
model.add(layers.Dense(2, name="output_layer"))

# this is a custom call back whcih prints the value of the attention weights
print_attn = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda batch, logs: print(model.layers[0].get_attention()))


optimizers = [
    tf.keras.optimizers.RMSprop(learning_rate=0.0033),  # different learning rate for first later
    tf.keras.optimizers.RMSprop(learning_rate=0.03) # than for the rest
]
optimizers_and_layers = [(optimizers[0], model.layers[0]), (optimizers[1], model.layers[1])]
optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)

# class categorization_accuracy(keras.metrics.Metric):
#     def __init__(self, name="categorization_accuracy", **kwargs):

# @pam this custom metric isn't working
def categorization_accuracy(y_true, y_pred):
    print("calling the metric")
    probs = tf.keras.activations.softmax(y_pred)
    print(y_pred)
    print(y_true)
    print('-----')
    return y_pred[1]

# change the loss!!
model.compile(optimizer=optimizer, loss=keras.losses.SquaredHinge(), metrics=[categorization_accuracy])
history = model.fit(stims, labels[problem], epochs=50, batch_size=8, verbose=True, callbacks=[print_attn])
run_df=pd.DataFrame(history.history)
run_df['error']=1.0-run_df['categorization_accuracy']
run_df['block']=run_df.index
run_df['problem']=problem+1

keras.utils.plot_model(model, 'model.png', show_shapes=True)