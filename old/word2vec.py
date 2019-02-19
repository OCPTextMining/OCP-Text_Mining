from sklearn.manifold import TSNE
from config import Config
import matplotlib.pyplot as plt
import tensorflow as tf
import nltk.help
import numpy as np
import random
import collections
import glob


def input_data(text_file_directory=Config.TEXT_DATA_PATH):
    """
    Reads the text documents contained in text_file_directory
    :param text_file_directory: Path to directory (String)
    :return: List where one element in a word in the text. The text order is kept. (List of String)
    """
    list_of_words = []
    for file_path in glob.glob(text_file_directory):
        with open(file_path, 'r', errors="ignore") as f:
            content = ''.join(f.readlines())
            content.replace('  ', ' ')
            list_of_current_words = content.split(' ')
            list_of_current_words = [x.lower() for x in list_of_current_words if x != '']
            try:
                pos_tagged_words = nltk.pos_tag(list_of_current_words)
            except LookupError:
                nltk.download('averaged_perceptron_tagger')
            list_of_nouns = [word for (word, tag) in pos_tagged_words if 'NN' in tag]
            list_of_words.extend(list_of_nouns)
    return list_of_words


def build_dataset(words, vocabulary_size=10000):
    """
    Extract the top 10,000 words and assign them an primary key. The other words are grouped under the 'UNK' category
    :param words: list of words (list of Strings)
    :param vocabulary_size: Keep at most this number of words. Other will be classed at 'UNK'
    :return: data (list of integer keeping the text order), dictionary (mapping from words to index),
        reverse_dictionary (inverse mapping)
    """
    # Initialize count variable with the unique words representation
    count = [['UNK', -1]]
    # Counts all words
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    # Assign a primary key to each word
    for word, _ in count:
        dictionary[word] = len(dictionary)
    # List that will contain the index of each word
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    # Dict mapping index to the word
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


data_index = 0


def generate_batch(data, batch_size, num_skips, skip_window):
    """
    Generate batch training data (randomly chosen words and context words draw using the skip-gram method)
    :param data: list of integers representing words
    :param batch_size: size of the output (length of the list)
    :param num_skips: number of words drawn randomly from the surrounding context (int)
    :param skip_window: size of the window of context words to draw from around the input word (int)
    :return:
    """
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    context = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window input_word skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # input word at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]  # this is the input word
            context[i * num_skips + j, 0] = buffer[target]  # these are the context words
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, context


# Create Tensorflow model


def embedding_layer(x, embedding_shape):
    with tf.variable_scope("embedding"):
        embedding_init = tf.random_uniform(embedding_shape, -1.0, 1.0)
        embedding_matrix = tf.get_variable("E_M", initializer=embedding_init)
        return tf.nn.embedding_lookup(embedding_matrix, x), embedding_matrix


def noise_contrastive_loss(embedding_lookup, weight_shape, bias_shape, y_idx, negative_sampled_size, vocabulary_size):
    with tf.variable_scope("NCE"):
        nce_weight_init = tf.truncated_normal(weight_shape, stddev=1.0 / (weight_shape[1]) ** 0.5)
        nce_bias_init = tf.zeros(bias_shape)

        nce_weights = tf.get_variable("W", initializer=nce_weight_init)
        nce_biases = tf.get_variable("b", initializer=nce_bias_init)

        # Computes and returns the noise-contrastive estimation training loss.
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                             biases=nce_biases,
                                             labels=y_idx,
                                             inputs=embedding_lookup,
                                             num_sampled=negative_sampled_size,
                                             num_classes=vocabulary_size))

        return loss


def training(cost, global_step, learning_rate):
    with tf.variable_scope("training"):
        summary_op = tf.summary.scalar("cost", cost)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(cost, global_step=global_step)
        return train_op, summary_op


def validation(embedding_matrix, x_val):
    norm = tf.reduce_sum(embedding_matrix ** 2, 1, keep_dims=True) ** 0.5
    normalized = embedding_matrix / norm
    validation_embeddings = tf.nn.embedding_lookup(normalized, x_val)
    cosine_similarity = tf.matmul(validation_embeddings, normalized, transpose_b=True)
    return normalized, cosine_similarity


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)


def train_model(data, validation_examples, vocabulary_size, embedding_size=128, batch_size=128, num_skips=2, skip_window=2,
                training_epochs=500, display_step=10000, validation_step=10000, validation_size=20, top_match=10,
                plot_num=1000, just_for_plot=1000):
    """

    :param data:
    :param vocabulary_size:
    :param embedding_size: size of embedding vectors
    :param batch_size: length of training vectors
    :param num_skips: number of context words
    :param skip_window: windows size (max distance of the context words from the input words)
    :param validation_examples: validation data (indices)
    :param training_epochs: number of epochs for training
    :param display_step: print step
    :param validation_step: frequency to perform validation
    :param validation_size:
    :param top_match:
    :param plot_num:
    :param just_for_plot:
    :return:
    """
    # of batches per epoch
    batches_per_epoch = int(len(data) * num_skips / batch_size)

    model_path = 'model/'

    with tf.Graph().as_default():
        with tf.variable_scope("Skip_Gram_Model"):
            x = tf.placeholder(tf.int32, shape=[batch_size])
            y = tf.placeholder(tf.int32, [batch_size, 1])
            val = tf.constant(validation_examples, dtype=tf.int32)

            global_step = tf.Variable(0, name='global_step', trainable=False)
            e_lookup, e_matrix = embedding_layer(x, [vocabulary_size, embedding_size])

            tf.cast(e_lookup, tf.int32)
            cost = noise_contrastive_loss(e_lookup, [vocabulary_size, embedding_size], [vocabulary_size], y,
                                          negative_sampled_size=64, vocabulary_size=10000)
            train_op, summary_op = training(cost, global_step, learning_rate=0.05)

            validation_op = validation(e_matrix, val)

            sess = tf.Session()
            train_writer = tf.summary.FileWriter(model_path, graph=sess.graph)
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            step = 0
            avg_cost = 0

            for epoch in range(training_epochs):
                print('epoch:', epoch, 'batches_per_epoch:', batches_per_epoch)

                for minibatch in range(batches_per_epoch):
                    step += 1
                    minibatch_x, minibatch_y = generate_batch(data, batch_size, num_skips, skip_window)
                    feed_dict = {x: minibatch_x, y: minibatch_y}

                    _, new_cost, train_summary = sess.run([train_op, cost, summary_op], feed_dict=feed_dict)
                    train_writer.add_summary(train_summary, sess.run(global_step))
                    # Compute average loss
                    avg_cost += new_cost / display_step

                    if step % display_step == 0:
                        print("Elapsed:", str(step), " batches. Cost =", "{:.9f}".format(avg_cost))
                        avg_cost = 0

                    if step % validation_step == 0:
                        _, similarity = sess.run(validation_op)
                        for i in range(validation_size):
                            validation_word = reverse_dictionary[validation_examples[i]]
                            neighbors = (-similarity[i, :]).argsort()[1:top_match + 1]
                            print_str = "Nearest neighbor of %s:" % validation_word

                            for k in range(top_match):
                                print_str += " %s," % reverse_dictionary[neighbors[k]]
                            print(print_str[:-1])

            f_embeddings, _ = sess.run(validation_op)
    print(type(f_embeddings))
    tsne = TSNE(perplexity=30, n_components=2, early_exaggeration=12.0, learning_rate=200.0, n_iter=2000,
                n_iter_without_progress=300, init='pca')
    plot_embeddings = np.asfarray(f_embeddings[:plot_num, :], dtype='float')
    low_dim_embeddings = tsne.fit_transform(plot_embeddings)
    labels = [reverse_dictionary[i] for i in range(just_for_plot)]
    plot_with_labels(low_dim_embeddings, labels)


if __name__ == "__main__":
    # Nearest neighbors validation parameters
    validation_size = 20
    validation_dist_span = 500

    list_of_words = input_data(Config.TEXT_DATA_PATH)
    data, count, dictionary, reverse_dictionary = build_dataset(list_of_words)
    # batch, context = generate_batch(data, 20, 10, 10)
    train_model(data, np.random.choice(validation_dist_span, validation_size, replace=False), 10000)