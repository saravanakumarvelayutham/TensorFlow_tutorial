import tensorflow as tf
import numpy as np

from collections import OrderedDict

def create_dense_layer(inputs, n_nodes, activation=None, dropout=None):
    return tf.layers.dense(inputs=inputs,
                           units=n_nodes,
                           activation=activation)
    
    

def create_dropout(inputs, rate, mode):
    return tf.layers.dropout(inputs=inputs, 
                             rate=rate, 
                             training=mode == tf.estimator.ModeKeys.TRAIN)




# tf.logging.set_verbosity(tf.logging.INFO)


def deep_model_fn(features, labels, mode, params):
    """Model function for creating dense models"""
    from collections import OrderedDict
    
    layers_dict = OrderedDict()
    # Input Layer
    layers_dict['input_layer'] = features["x"]

    # create layers from layers param
    for i,layer in enumerate(params['layers']):
        layer['inputs'] = list(layers_dict.values())[-1]
        if layer == params['layers'][-1]:
            key = 'logits'
        else:
            key = 'layer' + str(i)
        layers_dict[key] = create_dense_layer(**layer)
        if layer['dropout']:
            layers_dict['dropout_l' + str(i)] = create_dropout(list(layers_dict.values())[-1],
                                                               rate=layer['dropout'],
                                                               mode=mode)
    print(list(layers_dict.keys()))


    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=layers_dict['logits'], axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(layers_dict['logits'], name="softmax_tensor")
    }
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=layers_dict['logits'])

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    

def main(model_params=None, mode='train', model_dir=None):
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
#     train_data = X['train']
#     train_labels = y['train']
    
#     eval_data = X['test']
#     eval_labels = y['test']

    # Create the Estimator
    if not model_dir:
        import os
        model_dir = '/tmp/mnist_deep_model'
        if mode == 'train':
            _ = os.system('rm -rf ' + model_dir)
    mnist_classifier = tf.estimator.Estimator(
        model_fn=deep_model_fn,
        params=model_params,
        model_dir=model_dir)


    # Train the model
    if mode == 'train':
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=100,
            num_epochs=10,
            shuffle=True)
        mnist_classifier.train(
            input_fn=train_input_fn)
    elif mode == 'predict':
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            num_epochs=1,
            shuffle=False)
        preds = mnist_classifier.predict(
            input_fn=predict_input_fn)
        return [p for p in preds], eval_labels
    elif mode == 'eval':
        # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)
        eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)

layers = [{'n_nodes': 1024, 'activation': tf.nn.relu, 'dropout': None},
          {'n_nodes': 512, 'activation': tf.nn.relu, 'dropout': 0.4},
          {'n_nodes': 256, 'activation': tf.nn.relu, 'dropout': None},
#           {'n_nodes': 256, 'activation': tf.nn.relu, 'dropout': 0.4},
          {'n_nodes': 128, 'activation': tf.nn.relu, 'dropout': 0.4},
#           {'n_nodes': 64, 'activation': tf.nn.relu, 'dropout': 0.2},
          {'n_nodes': 10, 'activation': None, 'dropout': None}]

model_params = {'learning_rate': 0.01,
                'layers': layers}

main(model_params, mode='train')

from sklearn.metrics import classification_report as crep, accuracy_score as acc

preds = main(model_params, mode='predict')

ypred3 = [c['classes'] for c in preds[0]]
ytrue = preds[1]

print('acc: ', acc(ytrue, ypred3), '\n')
print(crep(ytrue, ypred3))
