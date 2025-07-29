import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def my_freeze_graph(output_node_names, destination, name="frozen_model.pb"):
    """
    Freeze the current graph alongside its weights (converted to constants) into a protobuf file.
    :param output_node_names: The name of the output node names we are interested in
    :param destination: Destination folder or remote service (eg. gs://)
    :param name: Filename of the saved graph
    :return:
    """
    #tf.keras.backend.set_learning_phase(0)  # set inference phase

    sess = tf.keras.backend.get_session()
    input_graph_def = sess.graph.as_graph_def()     # get graph def proto from keras session's graph

    with sess.as_default():
        # Convert variables into constants so they will be stored into the graph def
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names=output_node_names)

        tf.train.write_graph(graph_or_graph_def=output_graph_def, logdir=destination, name=name, as_text=False)

    tf.keras.backend.clear_session()


batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
print(f'input_shape={input_shape}')

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')])

model.summary()

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size, epochs=epochs, verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.export('my_model_savedmodel')

loaded_model = tf.saved_model.load('my_model_savedmodel')
concrete_func = loaded_model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

# Convert variables to constants (freeze the graph)
frozen_func = convert_variables_to_constants_v2(concrete_func)
graphDef = frozen_func.graph.as_graph_def()
print("input node name:", graphDef.node[0].name)
print("output node name:", graphDef.node[-1].name)

# Save the frozen graph to a .pb file
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir='./',
                  name='frozen_graph.pb',
                  as_text=False)
