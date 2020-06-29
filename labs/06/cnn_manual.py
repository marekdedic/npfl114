#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from mnist import MNIST

class Convolution:
    def __init__(self, channels, kernel_size, stride, input_shape):
        # Create convolutional layer with the given arguments
        # and given input shape (e.g., [28, 28, 1]).

        self._channels = channels
        self._kernel_size = kernel_size
        self._stride = stride
        # Here the kernel and bias variables are created
        self._kernel = tf.Variable(
            tf.initializers.GlorotUniform(seed=42)([self._kernel_size, self._kernel_size, input_shape[2], self._channels]),
            trainable=True)
        self._bias = tf.Variable(tf.initializers.Zeros()([self._channels]), trainable=True)

    def forward(self, inputs):
        # TODO: Compute the forward propagation through the convolution
        # with `tf.nn.relu` activation and return the result.

        # In order for the computation to be reasonably fast, you cannot
        # manually iterate through the individual pixels or batch examples
        # (and ideally also not over input and output channels).
        # However, you can manually iterate through the kernel size.
        batch_size, input_h, input_w, _ = inputs.shape
        output_h = int(((input_h - self._kernel_size) / self._stride) + 1)
        output_w = int(((input_w - self._kernel_size) / self._stride) + 1)

        outputs = np.zeros((batch_size, output_h, output_w, self._channels))
        for kernel_x in range(self._kernel_size):
            for kernel_y in range(self._kernel_size):
                max_y = input_h - self._kernel_size + kernel_y + 1
                max_x = input_w - self._kernel_size + kernel_x + 1
                outputs += inputs[:, kernel_y:max_y:self._stride, kernel_x:max_x:self._stride, :] @ self._kernel[kernel_y, kernel_x]

        outputs += self._bias[tf.newaxis, tf.newaxis, tf.newaxis, :]
        return tf.nn.relu(outputs)

    def backward(self, inputs, outputs, outputs_gradient):
        # TODO: Given the inputs of the layer, outputs of the layer
        # (computed in forward pass) and the gradient of the loss
        # with respect to layer outputs, return a list with the
        # following three elements:
        # - gradient of inputs with respect to the loss
        # - list of variables in the layer, e.g.,
        #     [self._kernel, self._bias]
        # - list of gradients of the layer variables with respect
        #   to the loss (in the same order as the previous argument)

        batch_size, input_h, input_w, input_channels = inputs.shape
        inputs_gradient = np.zeros(inputs.shape, dtype=np.float32)
        kernel_gradient = np.zeros(self._kernel.shape, dtype=np.float32)

        relu_gradient = np.where(outputs > 0, outputs_gradient, 0)
        bias_gradient = tf.reduce_sum(relu_gradient, axis=[0, 1, 2])

        for kernel_x in range(self._kernel_size):
            for kernel_y in range(self._kernel_size):
                max_y = input_h - self._kernel_size + kernel_y + 1
                max_x = input_w - self._kernel_size + kernel_x + 1

                kernel_gradient[kernel_y, kernel_x] = tf.reduce_sum(inputs[:, kernel_y:max_y:self._stride, kernel_x:max_x:self._stride, :, tf.newaxis] * relu_gradient[:, :, :, tf.newaxis, :], axis=[0, 1, 2])
                inputs_gradient[:, kernel_y:max_y:self._stride, kernel_x:max_x:self._stride, :] += relu_gradient @ tf.transpose(self._kernel[kernel_y, kernel_x, :])

        return inputs_gradient, [self._kernel, self._bias], [tf.constant(kernel_gradient, dtype=tf.float32), tf.constant(bias_gradient, dtype=tf.float32)]

class Network:
    def __init__(self, args):
        # Create the convolutional layers according to args.cnn.
        input_shape = [MNIST.H, MNIST.W, MNIST.C]
        self._convolutions = []
        for layer in args.cnn.split(","):
            channels, kernel_size, stride = map(int, layer.split("-"))
            self._convolutions.append(Convolution(channels, kernel_size, stride, input_shape))
            input_shape = [(input_shape[0] - kernel_size) // stride + 1,
                           (input_shape[1] - kernel_size) // stride + 1, channels]

        # Create the classification head
        self._flatten = tf.keras.layers.Flatten(input_shape=input_shape)
        self._classifier = tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax)

        # Create the loss, metric and the optimizer
        self._loss = tf.losses.SparseCategoricalCrossentropy()
        self._accuracy = tf.metrics.SparseCategoricalAccuracy()
        self._optimizer = tf.optimizers.Adam(args.learning_rate)

    def train_epoch(self, dataset):
        for batch in dataset.batches(args.batch_size):
            # Forward pass through the convolutions
            hidden = tf.constant(batch["images"])
            convolution_values = [hidden]
            for convolution in self._convolutions:
                hidden = convolution.forward(hidden)
                convolution_values.append(hidden)

            # Run the classification head and compute its gradient
            with tf.GradientTape() as tape:
                tape.watch(hidden)

                predictions = self._flatten(hidden)
                predictions = self._classifier(predictions)
                loss = self._loss(batch["labels"], predictions)

            variables = self._classifier.trainable_variables
            hidden_gradient, *gradients = tape.gradient(loss, [hidden] + variables)

            # Backpropagate the gradient throug the convolutions
            for convolution, inputs, outputs in reversed(list(zip(self._convolutions, convolution_values[:-1], convolution_values[1:]))):
                hidden_gradient, convolution_variables, convolution_gradients =convolution.backward(inputs, outputs, hidden_gradient)
                variables.extend(convolution_variables)
                gradients.extend(convolution_gradients)

            # Update the weights
            self._optimizer.apply_gradients(zip(gradients, variables))

    def evaluate(self, dataset):
        self._accuracy.reset_states()
        for batch in dataset.batches(args.batch_size):
            hidden = batch["images"]
            for convolution in self._convolutions:
                hidden = convolution.forward(hidden)
            hidden = self._flatten(hidden)
            predictions = self._classifier(hidden)
            self._accuracy(batch["labels"], predictions)
        return self._accuracy.result().numpy()


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--cnn", default="5-3-2,10-3-2", type=str, help="CNN architecture.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate.")
    parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.recodex:
        tf.keras.utils.get_custom_objects()["glorot_uniform"] = tf.initializers.GlorotUniform(seed=args.seed)
        tf.keras.utils.get_custom_objects()["orthogonal"] = tf.initializers.Orthogonal(seed=args.seed)
        tf.keras.utils.get_custom_objects()["uniform"] = tf.initializers.RandomUniform(seed=args.seed)

    # Report only errors by default
    if not args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Load data, using only 5000 training images
    mnist = MNIST()
    mnist.train._size = 5000

    # Create the model
    network = Network(args)

    for epoch in range(args.epochs):
        network.train_epoch(mnist.train)

        accuracy = network.evaluate(mnist.dev)
        print("Dev accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * accuracy), flush=True)

    accuracy = network.evaluate(mnist.test)
    print("Test accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * accuracy), flush=True)


    # Save the test accuracy in percents rounded to two decimal places.
    with open("cnn_manual.out", "w") as out_file:
        print("{:.2f}".format(100 * accuracy), file=out_file)
