import numpy as np
import tensorflow as tf
import h5py
import json
import logging
from typing import Optional, Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DTYPE = tf.float32
DTYPE_INT = tf.int64


class UnicodeCharsVocabulary:
    def __init__(self, vocab_file: str, max_word_length: int):
        """
        Initializes the vocabulary from a file.

        Args:
            vocab_file (str): Path to the vocabulary file.
            max_word_length (int): Maximum number of characters per token.
        """
        self.vocab = []
        self.max_word_length = max_word_length
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                self.vocab.append(word)
        self.size = len(self.vocab)
        self.word_to_id = {word: idx for idx, word in enumerate(self.vocab)}
    
    def id_to_word(self, idx: int) -> str:
        return self.vocab[idx]
    
    def word_to_id_func(self, word: str) -> int:
        return self.word_to_id.get(word, 0)  


class Batcher:
    def __init__(self, vocab_file: str, max_word_length: int):
        self.vocab = UnicodeCharsVocabulary(vocab_file, max_word_length)
        self.max_word_length = max_word_length
        self.char_to_id = self._build_char_to_id()
    
    def _build_char_to_id(self) -> Dict[str, int]:
        chars = sorted(list(set(''.join(self.vocab.vocab))))
        char_to_id = {char: idx + 1 for idx, char in enumerate(chars)}  # 0 reserved for padding
        return char_to_id
    
    def batch_sentences(self, sentences: List[List[str]]) -> np.ndarray:
        batch_size = len(sentences)
        max_seq_len = max(len(sentence) for sentence in sentences)
        batch = np.zeros((batch_size, max_seq_len, self.max_word_length), dtype=np.int32)
        
        for i, sentence in enumerate(sentences):
            for j, word in enumerate(sentence):
                for k, char in enumerate(word[:self.max_word_length]):
                    batch[i, j, k] = self.char_to_id.get(char, 0) 
        return batch


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_heads: int, ff_dim: int, dropout_rate: float, name: str = "transformer_encoder"):
        super(TransformerEncoder, self).__init__(name=name)
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),  
            tf.keras.layers.Dense(ff_dim)  
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training: bool = False):
        attn_output = self.mha(inputs, inputs, inputs)  
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output) 

        return out2


class TransformerLanguageModel(tf.keras.Model):
    """
    Transformer-based Bidirectional Language Model.
    """
    def __init__(
        self,
        options: dict,
        weight_file: str,
        use_character_inputs: bool = True,
        embedding_weight_file: Optional[str] = None,
        max_batch_size: int = 128,
    ):
        super(TransformerLanguageModel, self).__init__()
        self.options = options
        self.use_character_inputs = use_character_inputs
        self.max_batch_size = max_batch_size
        self.weight_file = weight_file
        self.embedding_weight_file = embedding_weight_file

        if self.use_character_inputs:
            self.char_embedding = tf.keras.layers.Embedding(
                input_dim=self.options['char_cnn']['n_characters'],
                output_dim=self.options['char_cnn']['embedding']['dim'],
                mask_zero=True,
                name="char_embed"
            )
            self.conv_layers = [
                tf.keras.layers.Conv2D(
                    filters=num,
                    kernel_size=(1, width),
                    activation=self._get_activation(self.options['char_cnn']['activation']),
                    padding='valid',
                    name=f"conv_{i}"
                )
                for i, (width, num) in enumerate(self.options['char_cnn']['filters'])
            ]
            self.global_max_pool = tf.keras.layers.GlobalMaxPooling2D()
            self.char_projection = tf.keras.layers.Dense(
                units=self.options['lstm']['projection_dim'],
                activation='relu',
                name="char_projection"
            )
            self.dropout = tf.keras.layers.Dropout(self.options['transformer']['dropout'])
        else:
            self.token_embedding = tf.keras.layers.Embedding(
                input_dim=self.options['vocab_size'] + 1, 
                output_dim=self.options['lstm']['projection_dim'],
                embeddings_initializer='uniform',
                mask_zero=True,
                name="token_embed"
            )

        self.pos_encoding = self._positional_encoding(
            self.options['transformer']['max_position_encoding'],
            self.options['lstm']['projection_dim']
        )

        self.transformer_encoders = [
            TransformerEncoder(
                num_heads=self.options['transformer']['num_heads'],
                ff_dim=self.options['transformer']['ff_dim'],
                dropout_rate=self.options['transformer']['dropout'],
                name=f"transformer_encoder_{i}"
            )
            for i in range(self.options['transformer']['num_layers'])
        ]

        self.final_dense = tf.keras.layers.Dense(
            units=self.options['transformer']['output_dim'],
            activation=None,
            name="final_projection"
        )

    def _get_activation(self, activation_name: str):
        if activation_name.lower() == 'relu':
            return 'relu'
        elif activation_name.lower() == 'tanh':
            return 'tanh'
        else:
            raise ValueError(f"Unsupported activation function: {activation_name}")

    def _positional_encoding(self, position: int, d_model: int):
        angle_rads = self._get_angles(np.arange(position)[:, np.newaxis],
                                      np.arange(d_model)[np.newaxis, :],
                                      d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=DTYPE)

    def _get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    def call(self, inputs, training: bool = False):
        if self.use_character_inputs:
            x = self.char_embedding(inputs)  
            x = tf.expand_dims(x, axis=-1)  
            x = tf.reshape(x, shape=(-1, x.shape[2], x.shape[3], 1))  
        
            for conv in self.conv_layers:
                x = conv(x) 
            x = self.global_max_pool(x) 
            x = self.char_projection(x)  
            batch_size = tf.shape(inputs)[0]
            seq_len = tf.shape(inputs)[1]
            x = tf.reshape(x, shape=(batch_size, seq_len, -1))  
            
            x += self.pos_encoding[:, :seq_len, :]
            x = self.dropout(x, training=training)
        else:
            x = self.token_embedding(inputs)
            seq_len = tf.shape(inputs)[1]
            x += self.pos_encoding[:, :seq_len, :]
            x = self.dropout(x, training=training)

        for encoder in self.transformer_encoders:
            x = encoder(x, training=training)

        lm_embeddings = self.final_dense(x)  

        return lm_embeddings


class BidirectionalLanguageModel:
    def __init__(
        self,
        options_file: str,
        weight_file: str,
        use_character_inputs: bool = True,
        embedding_weight_file: Optional[str] = None,
        max_batch_size: int = 128,
    ):
        with open(options_file, 'r') as fin:
            options = json.load(fin)

        if not use_character_inputs and embedding_weight_file is None:
            raise ValueError(
                "embedding_weight_file is required when use_character_inputs is False."
            )

        self._options = options
        self._weight_file = weight_file
        self._embedding_weight_file = embedding_weight_file
        self._use_character_inputs = use_character_inputs
        self._max_batch_size = max_batch_size

        self.model = TransformerLanguageModel(
            options=self._options,
            weight_file=self._weight_file,
            use_character_inputs=self._use_character_inputs,
            embedding_weight_file=self._embedding_weight_file,
            max_batch_size=self._max_batch_size
        )

        dummy_input = self._create_dummy_input()
        self.model(dummy_input, training=False)

        self._load_pretrained_weights()

    def _create_dummy_input(self):
        if self._use_character_inputs:
            return tf.constant(np.zeros((1, 1, self._options['char_cnn']['max_characters_per_token']), dtype=np.int32))
        else:
            return tf.constant(np.zeros((1, 1), dtype=np.int32))

    def _load_pretrained_weights(self):
        logger.info("Pretrained weights loading is not implemented. Please implement the weight loading logic.")
        pass  

    def __call__(self, inputs):
        return self.model(inputs, training=False)


def dump_token_embeddings(vocab_file: str, options_file: str, weight_file: str, outfile: str, batch_size: int = 64) -> None:
    with open(options_file, 'r') as fin:
        options = json.load(fin)
    max_word_length = options['char_cnn']['max_characters_per_token']

    vocab = UnicodeCharsVocabulary(vocab_file, max_word_length)
    batcher = Batcher(vocab_file, max_word_length)

    model = BidirectionalLanguageModel(
        options_file=options_file,
        weight_file=weight_file,
        use_character_inputs=True,  
        embedding_weight_file=None, 
        max_batch_size=batch_size
    )

    n_tokens = vocab.size
    embed_dim = options['transformer']['output_dim']
    embeddings = np.zeros((n_tokens, embed_dim), dtype=DTYPE.as_numpy_dtype)

    for start in range(0, n_tokens, batch_size):
        end = min(start + batch_size, n_tokens)
        batch_tokens = [vocab.id_to_word(k) for k in range(start, end)]
        char_ids = batcher.batch_sentences([[token] for token in batch_tokens])
        char_ids_tensor = tf.constant(char_ids, dtype=tf.int32)
        lm_embeddings = model(char_ids_tensor)
        embeddings_batch = lm_embeddings.numpy()[:, 0, :] 
        embeddings[start:end, :] = embeddings_batch[:end - start, :]

        logger.info(f"Processed tokens {start} to {end}")

    with h5py.File(outfile, 'w') as fout:
        fout.create_dataset('embedding', data=embeddings, dtype='float32', compression="gzip")
    logger.info(f"Token embeddings saved to {outfile}")


def dump_bilm_embeddings(vocab_file: str, dataset_file: str, options_file: str, weight_file: str, outfile: str, batch_size: int = 32) -> None:
    with open(options_file, 'r') as fin:
        options = json.load(fin)
    max_word_length = options['char_cnn']['max_characters_per_token']

    vocab = UnicodeCharsVocabulary(vocab_file, max_word_length)
    batcher = Batcher(vocab_file, max_word_length)

    model = BidirectionalLanguageModel(
        options_file=options_file,
        weight_file=weight_file,
        use_character_inputs=True, 
        embedding_weight_file=None,  
        max_batch_size=batch_size
    )

    embed_dim = options['transformer']['output_dim']

    with h5py.File(outfile, 'w') as fout:
        sentence_id = 0
        batch_sentences = []

        with open(dataset_file, 'r') as fin_dataset:
            for line in fin_dataset:
                sentence = line.strip().split()
                batch_sentences.append(sentence)
                if len(batch_sentences) == batch_size:
                    char_ids = batcher.batch_sentences(batch_sentences)
                    char_ids_tensor = tf.constant(char_ids, dtype=tf.int32)
                    lm_embeddings = model(char_ids_tensor)
                    embeddings_batch = lm_embeddings.numpy()  

                    for embedding in embeddings_batch:
                        fout.create_dataset(
                            f'{sentence_id}',
                            data=embedding,
                            dtype='float32',
                            compression="gzip"
                        )
                        sentence_id += 1

                    logger.info(f"Processed sentences up to ID {sentence_id}")
                    batch_sentences = []

            if batch_sentences:
                char_ids = batcher.batch_sentences(batch_sentences)
                char_ids_tensor = tf.constant(char_ids, dtype=tf.int32)
                lm_embeddings = model(char_ids_tensor)
                embeddings_batch = lm_embeddings.numpy()

                for embedding in embeddings_batch:
                    fout.create_dataset(
                        f'{sentence_id}',
                        data=embedding,
                        dtype='float32',
                        compression="gzip"
                    )
                    sentence_id += 1
                logger.info(f"Processed all sentences up to ID {sentence_id}")

    logger.info(f"Bidirectional LM embeddings saved to {outfile}")
