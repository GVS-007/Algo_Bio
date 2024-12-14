import os
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, concatenate
from allennlp.commands.elmo import ElmoEmbedder
from tqdm import tqdm


def rebuild_model(input_dim_tcr, input_dim_epi):
    print(f"Rebuilding model with input_dim_tcr={input_dim_tcr}, input_dim_epi={input_dim_epi}")
    
    inputA = Input(shape=(input_dim_tcr,), name="Input_TCR")
    inputB = Input(shape=(input_dim_epi,), name="Input_Epitope")

    
    x = Dense(2048, kernel_initializer='he_uniform', name="Dense_TCR")(inputA)
    x = BatchNormalization(name="BatchNorm_TCR")(x)
    x = Dropout(0.3, name="Dropout_TCR")(x)
    x = tf.nn.silu(x)
    x = Model(inputs=inputA, outputs=x, name="TCR_Model")
    y = Dense(2048, kernel_initializer='he_uniform', name="Dense_Epitope")(inputB)
    y = BatchNormalization(name="BatchNorm_Epitope")(y)
    y = Dropout(0.3, name="Dropout_Epitope")(y)
    y = tf.nn.silu(y)
    y = Model(inputs=inputB, outputs=y, name="Epitope_Model")

    combined = concatenate([x.output, y.output], name="Concatenate")
    z = Dense(1024, name="Dense_Combined")(combined)
    z = BatchNormalization(name="BatchNorm_Combined")(z)
    z = Dropout(0.3, name="Dropout_Combined")(z)
    z = tf.nn.silu(z)
    z = Dense(1, activation='sigmoid', name="Output")(z)

    model = Model(inputs=[x.input, y.input], outputs=z, name="Combined_Model")
    model.compile(loss='binary_crossentropy', optimizer='adam')
    print("Model summary:")
    model.summary()
    return model



def load_and_adjust_weights(model, weights_path):
    import h5py
    print(f"Loading weights from {weights_path} with adjustments...")
    
    with h5py.File(weights_path, 'r') as f:
        for layer in model.layers:
            if layer.name in f.keys():
                group = f[layer.name]
                weights = []
                for weight_name in group.keys():
                    weight_data = group[weight_name][()]
                    layer_weight_shape = layer.get_weights()[len(weights)].shape

                    if weight_data.shape != layer_weight_shape:
                        print(
                            f"Adjusting weights for layer {layer.name}: "
                            f"File shape {weight_data.shape}, Model shape {layer_weight_shape}"
                        )
                      
                        if len(weight_data.shape) == 2:  
                            weight_data = weight_data[:layer_weight_shape[0], :layer_weight_shape[1]]
                        elif len(weight_data.shape) == 1:  
                            weight_data = weight_data[:layer_weight_shape[0]]
                        else:
                            raise ValueError(f"Unexpected weight shape for layer {layer.name}")

                    weights.append(weight_data)
                layer.set_weights(weights)
    print("Weights loaded and adjusted successfully!")


print(f"Loading weights from {weights_path}...")
load_and_adjust_weights(model, weights_path)




def initialize_embedding_model():
    print("Initializing embedding model...")
    device = 0 if tf.config.list_physical_devices('GPU') else -1
    return ElmoEmbedder('options.json', 'weights.hdf5', cuda_device=device)



def generate_embeddings(sequences, model, batch_size=128):
    print("Generating embeddings...")
    embeddings = []
    for i in tqdm(range(0, len(sequences), batch_size)):
        batch_sequences = sequences[i: i + batch_size]
        batch_embeddings = model.embed_sentence(batch_sequences)
        batch_embeddings = np.mean(batch_embeddings, axis=0)  # Average across layers
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)



def load_test_data(csv_path, model, batch_size=128):
    print("Loading test data...")
    test_data = pd.read_csv(csv_path)
    epi_sequences = test_data.iloc[:, 0].tolist()  
    tcr_sequences = test_data.iloc[:, 1].tolist()  

    print("Generating embeddings for epitope...")
    epi_embeddings = generate_embeddings(epi_sequences, model, batch_size=batch_size)
    print("Generating embeddings for TCR...")
    tcr_embeddings = generate_embeddings(tcr_sequences, model, batch_size=batch_size)

    return epi_embeddings, tcr_embeddings, test_data


def ensure_csv_exists(file_path):
    if not os.path.exists(file_path):
        print(f"{file_path} does not exist. Creating an empty file...")
        df = pd.DataFrame(columns=["epi", "tcr", "binding"])
        df.to_csv(file_path, index=False)
        print(f"Empty file created at {file_path}.")


def predict_and_save_results(weights_path, X1_test, X2_test, test_data, output_file):
    ensure_csv_exists(output_file)

    print("Rebuilding the model...")
    input_dim_tcr = X1_test.shape[1]
    input_dim_epi = X2_test.shape[1]
    model = rebuild_model(input_dim_tcr, input_dim_epi)

    print(f"Loading weights from {weights_path}...")
    load_and_adjust_weights(model, weights_path)  

    print("Running predictions...")
    predictions = model.predict([X1_test, X2_test])  
    print(predictions)
    
    test_data['binding'] = predictions  

    test_data.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, required=True, help="Path to the test CSV file.")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to the model weights file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output predictions.")
    args = parser.parse_args()

    
    embedding_model = initialize_embedding_model()


    X1_test, X2_test, test_data = load_test_data(args.test_csv, embedding_model)

    predict_and_save_results(args.weights_path, X1_test, X2_test, test_data, args.output_file)
