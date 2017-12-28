import keras
import keras.backend as backend
import numpy as np
import tensorflow as tf
from keras.engine.training import Model
from keras.initializers import he_normal
from keras.layers import Dense
from keras.layers.core import Lambda
from keras.layers.merge import concatenate
from keras.models import Sequential
from keras.utils import np_utils


def _get_available_devices():
    return [x.name for x in backend.get_session().list_devices()]


def _normalize_device_name(name):
    name = '/' + ':'.join(name.lower().replace('/', '').split(':')[-2:])
    return name


def multi_gpu_model(model, gpus):
    if backend.backend() != 'tensorflow':
        raise ValueError('`multi_gpu_model` is only available '
                         'with the TensorFlow backend.')
    if isinstance(gpus, (list, tuple)):
        if len(gpus) <= 1:
            raise ValueError('For multi-gpu usage to be effective, '
                             'call `multi_gpu_model` with `len(gpus) >= 2`. '
                             'Received: `gpus=%s`' % gpus)
        num_gpus = len(gpus)
        target_gpu_ids = gpus
    else:
        if gpus <= 1:
            raise ValueError('For multi-gpu usage to be effective, '
                             'call `multi_gpu_model` with `gpus >= 2`. '
                             'Received: `gpus=%d`' % gpus)
        num_gpus = gpus
        target_gpu_ids = range(num_gpus)

    import tensorflow as tf

    target_devices = ['/cpu:0'] + ['/gpu:%d' % i for i in target_gpu_ids]
    available_devices = _get_available_devices()
    available_devices = [_normalize_device_name(name) for name in available_devices]
    for device in target_devices:
        if device not in available_devices:
            raise ValueError(
                'To call `multi_gpu_model` with `gpus=%d`, '
                'we expect the following devices to be available: %s. '
                'However this machine only has: %s. '
                'Try reducing `gpus`.' % (gpus,
                                          target_devices,
                                          available_devices))

    def get_slice(data, i, parts):
        shape = tf.shape(data)
        batch_size = shape[:1]
        input_shape = shape[1:]
        step = batch_size // parts
        if i == num_gpus - 1:
            size = batch_size - step * i
        else:
            size = step
        size = tf.concat([size, input_shape], axis=0)
        stride = tf.concat([step, input_shape * 0], axis=0)
        start = stride * i
        return tf.slice(data, start, size)

    all_outputs = []
    for i in range(len(model.outputs)):
        all_outputs.append([])

    # Place a copy of the model on each GPU,
    # each getting a slice of the inputs.
    for i, gpu_id in enumerate(target_gpu_ids):
        with tf.device('/gpu:%d' % gpu_id):
            with tf.name_scope('replica_%d' % gpu_id):
                inputs = []
                # Retrieve a slice of the input.
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_i = Lambda(get_slice,
                                     output_shape=input_shape,
                                     arguments={'i': i,
                                                'parts': num_gpus})(x)
                    inputs.append(slice_i)

                # Apply model on slice
                # (creating a model replica on the target device).
                outputs = model(inputs)
                if not isinstance(outputs, list):
                    outputs = [outputs]

                # Save the outputs for merging back together later.
                for o in range(len(outputs)):
                    all_outputs[o].append(outputs[o])

    # Merge outputs on CPU.
    with tf.device('/cpu:0'):
        merged = []
        for name, outputs in zip(model.output_names, all_outputs):
            merged.append(concatenate(outputs,
                                      axis=0, name=name))
        return Model(model.inputs, merged)


if __name__ == "__main__":
    data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    label = np.array([0, 1, 1, 1])
    label = np_utils.to_categorical(label)

    with tf.device("/cpu:0"):
        model = Sequential()

        model.add(Dense(units=5, activation='relu', kernel_initializer=he_normal(), input_shape=(2,), name="dense1"))
        model.add(Dense(units=2, activation='softmax', name="dense2"))

    parallel_model = multi_gpu_model(model, 4)

    parallel_model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adam(lr=1e-5),
                           metrics=['accuracy'])

    parallel_model.fit(data, label,
                       verbose=2,
                       batch_size=10,
                       epochs=10,
                       shuffle=True)
