'''Train sentiment analysis model on IMDB dataset.'''

from transformer import ClassificationTransformer

import tensorflow as tf
tf.enable_eager_execution()
import tensorflow.keras.preprocessing.text as tx

import numpy as np


def get_data(max_vocab_size, seq_len, batch_size, num_epochs):
      
    train_data, train_labels, test_data, test_labels, val_data, val_labels = [], [], [], [], [], []
    
    with open("./data/opspam_train_reviews.txt", "r") as f:
        for line in f:
            train_data.append(line)
    with open("./data/opspam_train_labels.txt", "r") as f:
        for line in f:
            train_labels.append(int(line))
    with open("./data/opspam_test_reviews.txt", "r") as f:
        for line in f:
            test_data.append(line)
    with open("./data/opspam_test_labels.txt", "r") as f:
        for line in f:
            test_labels.append(int(line))
    with open("./data/opspam_val_reviews.txt", "r") as f:
        for line in f:
            val_data.append(line)
    with open("./data/opspam_val_labels.txt", "r") as f:
        for line in f:
            val_labels.append(int(line))
    
    all_data = train_data + val_data + test_data
    
    tokenizer = tx.Tokenizer(num_words=max_vocab_size,
                             filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                             lower=True,
                             split=" ",
                             char_level=False,
                             oov_token=None)
    
    tokenizer.fit_on_texts(all_data)
    
    train_data = np.array(tokenizer.texts_to_sequences(train_data))
    val_data = np.array(tokenizer.texts_to_sequences(val_data))
    test_data = np.array(tokenizer.texts_to_sequences(test_data))
     
    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)
    test_labels = np.array(test_labels)
     
    train_data = tf.keras.preprocessing.sequence.pad_sequences(
            train_data, value=0, padding='post', maxlen=seq_len)
    val_data = tf.keras.preprocessing.sequence.pad_sequences(
            val_data, value=0, padding='post', maxlen=seq_len)
    test_data = tf.keras.preprocessing.sequence.pad_sequences(
            test_data, value=0, padding='post', maxlen=seq_len)
          
    codes_to_words = dict()
    for word, raw_code in tokenizer.word_index.items():
        codes_to_words[3 + raw_code] = word
        # NB the first 3 slots are reserved: 0 -> <pad>, 1 -> <start>, 2 -> <unk>, 3 => <unused>
      
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    train_dataset = train_dataset.shuffle(2000).batch(batch_size).repeat(num_epochs)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
    val_dataset = val_dataset.shuffle(2000).batch(batch_size).repeat(num_epochs)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
    test_dataset = test_dataset.shuffle(2000).batch(batch_size)
  
    return train_dataset, val_dataset, test_dataset, codes_to_words, len(tokenizer.texts_to_sequences(all_data))


@tf.function
def train_step(model, loss_obj, optimizer, inputs, labels):
    with tf.GradientTape() as tape:
        probs = model(inputs, training=True)
        loss_val = loss_obj(labels, probs)
    gradients = tape.gradient(loss_val, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss_val


def validation_step(model, 
                    codes_to_words, 
                    val_dataset, 
                    test_dataset, 
                    loss_val,
                    num_data, 
                    epoch_index, 
                    num_epochs, 
                    batch_index, 
                    batch_size):
    
    # Compute accuracy score
    metric = tf.keras.metrics.Accuracy()
    
    # For val dataset
    for inputs, labels in val_dataset:
        probs = model(inputs, training=False)
        preds = tf.math.greater(probs, 0.5)
        metric(tf.cast(labels, 'bool'), preds)
    val_acc_score = metric.result()
    
    # For test dataset
    for inputs, labels in test_dataset:
        probs = model(inputs, training=False)
        preds = tf.math.greater(probs, 0.5)
        metric(tf.cast(labels, 'bool'), preds)
    test_acc_score = metric.result()
    
    # Print sample prediction.
    inputs, labels = next(iter(test_dataset.take(1)))
    probs = model(inputs, training=False)
    batch_size = tf.shape(inputs).numpy()[0]
    rand_index = np.random.randint(low=0, high=batch_size)
    sample_codes = inputs.numpy()[rand_index, :]
    sample_words = ' '.join([codes_to_words[code] for code in sample_codes if code >= 4])
    sample_prob = probs.numpy()[rand_index]
    sample_label = labels.numpy()[rand_index]
    
    print("\nEpoch: {}/{}".format(epoch_index, num_epochs))
    print('Batch: {}/{:.0f}'.format(batch_index + 1, num_data/batch_size*num_epochs))
    print("Loss value: {}".format(loss_val))
    print('Val accuracy: {:.3f}, Test accuracy: {:.3f}'.format(val_acc_score, test_acc_score))
    print('Sample sentence (prediction={:.3f}, actual label={}):\n{}\n'.format(
            sample_prob, sample_label, sample_words))
    

def main(max_vocab_size, seq_len, batch_size, num_epochs,
         num_layers, model_dims, attention_depth, num_heads, hidden_dims,
         num_batches_per_validation):
    '''
    :param max_vocab_size: maximum vocabulary size, only top words will be retained
    :param seq_len: maximum sentence length, sentences will be truncated or padded
    :param num_epochs: number of epochs
    :param num_layers: number of layers for transformer
    :param model_dims: embedding dimension in transformer
    :param attention_depth: depth of attention heads
    :param num_heads: number of heads per attention unit
    :param hidden_dims: number of hidden dimensions in transformer
    :param num_batches_per_validation: frequency at which we evaluate the model's performance
    '''
    train_dataset, val_dataset, test_dataset, codes_to_words, num_data = get_data(max_vocab_size, seq_len, batch_size, num_epochs)
    vocab_size = max_vocab_size + 4
    
    model = ClassificationTransformer(vocab_size, num_layers, model_dims,
                                      attention_depth, num_heads, hidden_dims)
    loss_obj = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(lr=0.0001) 
    
    epoch_index = 0
    for batch_index, (inputs, labels) in enumerate(train_dataset):
        loss_val = train_step(model, loss_obj, optimizer, inputs, labels)
        #print("\nbp{}\n".format(batch_index))
        if (batch_index*batch_size/num_data >= epoch_index):
            epoch_index = epoch_index + 1
            #validation_step(model, codes_to_words, val_dataset, test_dataset, num_data, epoch_index, num_epochs, batch_index, batch_size)
        if (batch_index + 1) % num_batches_per_validation == 0:
            validation_step(model, 
                            codes_to_words, 
                            val_dataset, 
                            test_dataset, 
                            loss_val,
                            num_data, 
                            epoch_index, 
                            num_epochs, 
                            batch_index, 
                            batch_size)
            

if __name__ == '__main__':
    main(max_vocab_size=10000,
         seq_len=256,
         batch_size=64,
         num_epochs=30,
         num_layers=4,
         model_dims=256,
         attention_depth=16,
         num_heads=8,
         hidden_dims=256,
         num_batches_per_validation=50
    )   
