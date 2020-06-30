#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from cifar10 import CIFAR10

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    # TODO: Define reasonable defaults and optionally more parameters
    parser.add_argument("--batch_size", default=None, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=None, type=int, help="Number of epochs.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Report only errors by default
    if not args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))
    os.makedirs(args.logdir, exist_ok=True)

    # Load data
    cifar = CIFAR10()

    # TODO: Create the model and train it
    inputs = tf.keras.layers.Input([CIFAR10.H, CIFAR10.W, CIFAR10.C])
    hidden = inputs
    hidden = tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation=tf.nn.relu)(hidden)
    hidden = tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation=tf.nn.relu)(hidden)
    hidden = tf.keras.layers.MaxPool2D(pool_size=2)(hidden)
    hidden = tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation=tf.nn.relu)(hidden)
    hidden = tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation=tf.nn.relu)(hidden)
    hidden = tf.keras.layers.MaxPool2D(pool_size=2)(hidden)
    #hidden = tf.keras.layers.Conv2D(128, kernel_size=5, padding="same", activation=tf.nn.relu, use_bias=False)(hidden)
    #hidden = tf.keras.layers.Conv2D(128, kernel_size=5, padding="same", activation=tf.nn.relu, use_bias=False)(hidden)
    #hidden = tf.keras.layers.MaxPool2D(pool_size=2)(hidden)
    hidden = tf.keras.layers.Flatten()(hidden)
    outputs = tf.keras.layers.Dense(CIFAR10.LABELS, activation=tf.nn.softmax)(hidden)
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(
        optimizer = tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="Dev set accuracy")]
    )
    model.fit(
        cifar.train.data["images"], cifar.train.data["labels"],
        batch_size=args.batch_size,
        validation_data=(cifar.dev.data["images"], cifar.dev.data["labels"]),
        epochs=args.epochs
    )

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as out_file:
        for probs in model.predict(cifar.test.data["images"], batch_size=args.batch_size):
            print(np.argmax(probs), file=out_file)
