#!/usr/bin/python
# -*- coding: utf-8 -*-
# %% imports we know we'll need
from __future__ import print_function  # bringing the print from 3 to 2.6
from scipy import special  # special functions for mathematics and physics
import numpy as np
import tensorflow as tf
import BER_calc
from sklearn.utils import shuffle
import os
import channel_model as ch
from importlib import reload
import matplotlib.pyplot as plt
from tensorflow import keras

reload(ch)  # reloads the ch module in case of we have edited it externally
# the below block is to make GPU 0 default, in case of distributed or GPU 1 doesn't work again
os.environ[
    "CUDA_DEVICE_ORDER"
] = "PCI_BUS_ID"  # Arrange GPU devices from 0 in the order of PCI_BUS_ID
os.environ[
    "CUDA_VISIBLE_DEVICES"
] = "1"  # Set the currently used GPU device to be only the device name of the 0/gpu:0'
print(tf.config.list_physical_devices("GPU"))


def create_dataset_m(
    x_pol_in_complex,
    y_pol_in_complex,
    x_pol_des_complex,
    n_sym1,
):
    raw_size = x_pol_in_complex.shape[0]
    dataset_size = raw_size - 2 * n_sym1
    dataset_range = n_sym1 + np.arange(dataset_size)
    dataset_x_pol__des = np.empty([dataset_size, 2], dtype="float64")
    dataset_x_pol__des[:, 0] = np.real(x_pol_des_complex[dataset_range])
    dataset_x_pol__des[:, 1] = np.imag(x_pol_des_complex[dataset_range])
    dataset_x_pol__in = np.empty([dataset_size, n_sym1, 4], dtype="float64")
    dataset_x_pol__in[:] = np.nan
    bnd_vec = int(np.floor(n_sym1 / 2))

    for vec_idx, center_vec in enumerate(dataset_range):
        local_range = center_vec + np.arange(-bnd_vec, bnd_vec + 1)
        n1 = np.arange(0, n_sym1)

        if np.any(local_range < 0) or np.any(local_range > raw_size):
            raise ValueError(
                "Local range steps out of the data range during dataset creation!!!"
            )
        else:
            dataset_x_pol__in[vec_idx, n1, 0] = np.real(x_pol_in_complex[local_range])
            dataset_x_pol__in[vec_idx, n1, 1] = np.imag(x_pol_in_complex[local_range])
            dataset_x_pol__in[vec_idx, n1, 2] = np.real(y_pol_in_complex[local_range])
            dataset_x_pol__in[vec_idx, n1, 3] = np.imag(y_pol_in_complex[local_range])

    if np.any(np.isnan(dataset_x_pol__in)):
        raise ValueError("Dataset matrix wasn't fully filled by data!!!")
    # shuffle avoids overfitting
    dataset_x_pol__in, dataset_x_pol__des = shuffle(
        dataset_x_pol__in, dataset_x_pol__des
    )
    return dataset_x_pol__in, dataset_x_pol__des


def BER_est(x_in, x_ref):
    QAM_order = QAM
    return BER_calc.QAM_BER_gray(x_in, x_ref, np.array(QAM_order))


# fiber parameters
QAM = 16
n_taps = 70
n_sym = 2 * n_taps + 1
lr1 = 0.001  # learning rate
training_epochs = 27
batches = 2000
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = \
    tf.data.experimental.AutoShardPolicy.OFF
guard_band = 1000
D_random = 16.8
gamma_random = 1.2
alpha_random = 0.21
roll_off_random = 0.1
NF_random = 4.5
fiber_length = 50
power = 6
symbol_rate = 40
n_span = 54

model_w_aug = keras.models.load_model('onlySR40_span54_273.h5')
model_w_aug.summary() #parameters

# #  NN
# inputs = tf.keras.Input(shape=(n_sym, 4), name="digits")
# x = tf.keras.layers.Bidirectional(
#     tf.keras.layers.LSTM(
#         100, return_sequences=True
#     )
# )(inputs)
# x = tf.keras.layers.Bidirectional(
#     tf.keras.layers.LSTM(
#         100, return_sequences=True
#     )
# )(x)
# x = tf.keras.layers.Bidirectional(
#     tf.keras.layers.LSTM(
#         100, return_sequences=True
#     )
# )(x)
# x = tf.keras.layers.Bidirectional(
#     tf.keras.layers.LSTM(
#         100, return_sequences=True
#     )
# )(x)
# x = tf.keras.layers.Flatten()(x)
# outputs = tf.keras.layers.Dense(2)(x)
# model_w_aug = tf.keras.Model(inputs=inputs, outputs=outputs)
# model_w_aug.compile(
#     loss=tf.keras.losses.mean_squared_error,
#     optimizer=tf.keras.optimizers.Adam(
#         learning_rate=lr1
#     ),
#     metrics=["accuracy"]
# )


exception_train = []
for epoch in range(training_epochs):
    channel = ch.create_channel_parameters(
        n_spans=n_span,
        z_span=fiber_length,
        alpha_db=alpha_random,
        gamma=gamma_random,
        noise_figure_db=NF_random,
        dispersion_parameter=D_random,
        dz=1,
    )
    wdm = ch.create_wdm_parameters(
        p_ave_dbm=power,
        n_symbols=2 ** 18,
        m_order=QAM,
        roll_off=roll_off_random,
        upsampling=8,
        downsampling_rate=4,
        symb_freq=symbol_rate* 1e9,
        np_filter=0,
        n_channels=1,
        channel_spacing=0,
    )

    result = ch.full_line_model(channel, wdm)
    Sca = np.sqrt((10 ** (wdm["p_ave_dbm"] / 10) * 1e-3) / 2)
    In_raw_complex_train = result["points_x_shifted"][2048:-2048] / Sca
    Out_raw_complex_train = result["points_orig_x"][2048:-2048] / Sca
    In_raw_complexy_train = result["points_y_shifted"][2048:-2048] / Sca
    Out_raw_complexy_train = result["points_orig_y"][2048:-2048] / Sca

    length_raw_complex_train = int(len(In_raw_complex_train))
    train_range = range(guard_band, int(length_raw_complex_train - guard_band))
    # training
    x_pol_in_raw_complex_train = In_raw_complex_train[train_range]
    y_pol_in_raw_complex_train = In_raw_complexy_train[train_range]
    x_pol_des_raw_complex_train = Out_raw_complex_train[train_range]
    y_pol_des_raw_complex_train = Out_raw_complexy_train[train_range]
    dataset_x_pol_in_train, dataset_x_pol_des_train = create_dataset_m(
        x_pol_in_raw_complex_train,
        y_pol_in_raw_complex_train,
        x_pol_des_raw_complex_train,
        n_sym
    )
    # Wrap data in Dataset objects.
    train_data = tf.data.Dataset.from_tensor_slices(
        (dataset_x_pol_in_train, dataset_x_pol_des_train)
    )
    # The batch size must now be set on the Dataset objects.
    train_data = train_data.batch(batches)
    train_data = train_data.with_options(options)
    model_w_aug.fit(train_data, epochs=1, verbose=0)
    Prediction_train = model_w_aug.predict(train_data, verbose=0)
    Xin_train = (
        dataset_x_pol_in_train[:, n_taps, 0] + 1j * dataset_x_pol_in_train[:, n_taps, 1]
    )
    Xeq_train = Prediction_train[:, 0] + 1j * Prediction_train[:, 1]
    Xref_train = dataset_x_pol_des_train[:, 0] + 1j * dataset_x_pol_des_train[:, 1]
    BER_train = BER_est(Xeq_train, Xref_train)
    BER_train_ref = BER_est(Xin_train, Xref_train)
    Q_train_predict = 20 * np.log10(np.sqrt(2) * special.erfcinv(2 * BER_train))
    Q_train_ref = 20 * np.log10(np.sqrt(2) * special.erfcinv(2 * BER_train_ref))
    print("Epoch now", epoch, "SR now", symbol_rate, "Span now", n_span)
    print("Q_train Sim now", Q_train_predict, "Q_train Sim ref now", Q_train_ref)
    model_w_aug.save("onlySR40_span54_300.h5")


Q_prop_ref3, Q_prop_ref4 = np.zeros(1), np.zeros(1)
# exception_test = []
for epoch in range(1):
    channel = ch.create_channel_parameters(
        n_spans=n_span,
        z_span=fiber_length,
        alpha_db=alpha_random,
        gamma=gamma_random,
        noise_figure_db=NF_random,
        dispersion_parameter=D_random,
        dz=1,
    )
    wdm = ch.create_wdm_parameters(
        p_ave_dbm=power,
        n_symbols=2 ** 18,
        m_order=QAM,
        roll_off=roll_off_random,
        upsampling=8,
        downsampling_rate=4,
        symb_freq=symbol_rate * 1e9,
        np_filter=0,
        n_channels=1,
        channel_spacing=0,
    )

    try:
        result = ch.full_line_model(channel, wdm)
    except:
        pass
    Sca = np.sqrt((10 ** (wdm["p_ave_dbm"] / 10) * 1e-3) / 2)
    In_raw_complex_test = result["points_x_shifted"][2048:-2048] / Sca
    Out_raw_complex_test = result["points_orig_x"][2048:-2048] / Sca
    In_raw_complexy_test = result["points_y_shifted"][2048:-2048] / Sca
    Out_raw_complexy_test = result["points_orig_y"][2048:-2048] / Sca

    length_raw_complex_test = int(len(In_raw_complex_test))
    test_range = range(guard_band, int(length_raw_complex_test - guard_band))

    # training
    x_pol_in_raw_complex_test = In_raw_complex_test[test_range]
    y_pol_in_raw_complex_test = In_raw_complexy_test[test_range]
    x_pol_des_raw_complex_test = Out_raw_complex_test[test_range]
    y_pol_des_raw_complex_test = Out_raw_complexy_test[test_range]
    dataset_x_pol_in_test, dataset_x_pol_des_test = create_dataset_m(
        x_pol_in_raw_complex_test,
        y_pol_in_raw_complex_test,
        x_pol_des_raw_complex_test,
        n_sym
    )
    # Wrap data in Dataset objects.
    n_test = np.floor(len(dataset_x_pol_in_test) / batches)
    range_test = np.arange(int(batches * n_test))
    test_data = tf.data.Dataset.from_tensor_slices((dataset_x_pol_in_test[range_test]))
    # The batch size must now be set on the Dataset objects.
    test_data = test_data.batch(batches)
    test_data = test_data.with_options(options)

    savedModel = keras.models.load_model("onlySR40_span54_300.h5")
    Prediction_test = savedModel.predict(test_data, verbose=0)
    Xeq_test = Prediction_test[:, 0] + 1j * Prediction_test[:, 1]
    Xref_test = (
        dataset_x_pol_des_test[range_test, 0]
        + 1j * dataset_x_pol_des_test[range_test, 1]
    )

    # test_index = symbol_rate_test.index(sr)
    BER_test = BER_est(Xeq_test, Xref_test)
    Best_BER_no_NN = BER_est(
        dataset_x_pol_in_test[range_test, n_taps, 0]
        + 1j * dataset_x_pol_in_test[range_test, n_taps, 1],
        dataset_x_pol_des_test[range_test, 0]
        + 1j * dataset_x_pol_des_test[range_test, 1],
    )
    Q_prop_ref3[epoch] = 20 * np.log10(np.sqrt(2) * special.erfcinv(2 * BER_test))
    Q_prop_ref4[epoch] = 20 * np.log10(np.sqrt(2) *
                                       special.erfcinv(2 * Best_BER_no_NN))
    print(
        "##########################################"
        "###################################################"
    )
    print("SR now", symbol_rate, "Span now")
    print("Q_test Exp now", Q_prop_ref3[epoch],
          "Q-factor  Exp ref", Q_prop_ref4[epoch])

np.save("onlySR40_span54_300", Q_prop_ref3)
np.save("onlySR40_span54_300_ref", Q_prop_ref4)
plt.bar(symbol_rate, Q_prop_ref3, label="Exp now NN", color="blue")
plt.bar(symbol_rate, Q_prop_ref4, label="Exp ref no NN", color="red")
plt.legend()
plt.title("onlySR40_span54_300")
plt.show()
