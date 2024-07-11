import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arrow

DATA_PATH = "Data.json"

def load_data(data_path):

    with open(data_path, "r") as fp:
        data = json.load(fp)
    
    X = np.array(data["mfcc"])
    y = np.array(data["labels"]) 
    return X, y

def prepare_datasets(test_size, validation_size):

    #load data
    X, y= load_data(DATA_PATH)

    #create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    #create train/ validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # 3D ARRAY -> (130,13, 1)
    X_train = X_train[..., np.newaxis] # 4d array ->(num_samples, 130 ,13 ,1)
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test

def build_model(input_shape):
    # create model

    model = keras.Sequential()

    #1st conv layer
    model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3,3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    
    #2nd conv layer
    model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3,3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    #3rd conv layer
    model.add(keras.layers.Conv2D(32, (2,2), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((2,2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    #flatten the output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    #output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model

def predict(model, X, y):

    X = X[np.newaxis, ...]

    
    prediction = model.predict(X) # X -> (1, 130,13, 1)

    #extract index with max value
    predicted_index = np.argmax(prediction, axis=1)
    print("Expected index: {}, Predicted index: {}".format(y, predicted_index))

def draw_neural_network(ax, left, right, bottom, top, layer_sizes):
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                            color='w', ec='k', zorder=4)
            ax.add_artist(circle)
            
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k', alpha=0.1)
                ax.add_artist(line)

def visualize_cnn_architecture():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    left, right, bottom, top = 0.1, 0.9, 0.1, 0.9
    
    input_layer = Rectangle((0.05, 0.3), 0.1, 0.4, fill=False)
    ax.add_patch(input_layer)
    ax.text(0.1, 0.25, 'Input\n(130, 13, 1)', ha='center')
    
    conv_layers = [
        ((0.2, 0.3), "Conv2D\n32 filters\n(3x3)"),
        ((0.3, 0.3), "Conv2D\n32 filters\n(3x3)"),
        ((0.4, 0.3), "Conv2D\n32 filters\n(2x2)")
    ]
    
    for pos, label in conv_layers:
        conv = Rectangle(pos, 0.08, 0.4, fill=False)
        ax.add_patch(conv)
        ax.text(pos[0] + 0.04, 0.25, label, ha='center')
        
        pool = Rectangle((pos[0] + 0.09, 0.4), 0.02, 0.2, fill=False)
        ax.add_patch(pool)
        ax.text(pos[0] + 0.1, 0.35, 'MaxPool', rotation=90, ha='center', va='center')
    
    flatten = Arrow(0.5, 0.5, 0.05, 0, width=0.3, color='k', alpha=0.3)
    ax.add_patch(flatten)
    ax.text(0.525, 0.7, 'Flatten', ha='center')
    
    dense_layers = [64, 10]
    draw_neural_network(ax, 0.6, 0.9, 0.2, 0.8, dense_layers)
    ax.text(0.7, 0.15, 'Dense (64)', ha='center')
    ax.text(0.85, 0.15, 'Output (10)', ha='center')
    
    plt.title('Convolutional Neural Network Architecture')
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":

    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    #build CNN network
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape)
    
    #compile network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])
    
    #train the CNN
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30, verbose=1)

    #evaluate the CNN
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))

    #make prediction on sample

    X = X_test[100]
    y = y_test[100]

    predict(model, X,y)

    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()