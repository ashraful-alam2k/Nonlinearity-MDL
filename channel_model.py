import numpy as np
from scipy.fft import fft, ifft, fftshift
import tensorflow as tf
# from datetime import datetime
import commpy
from datetime import datetime


# SSFM CPU version


def ssfm_dispersive_step(
    signal, t_span, dispersion=None, w=None, delta_z=0.001, alpha=0, beta2=1, beta3=0
):
    # F+
    temp_freq = fft(signal)

    if dispersion is None:
        if w is None:
            # w is frequencies in Fourier-space for Split-step method
            # w is defined as w = -W/2 : dw : W/2
            # W is 1 / dt, where dt is initial signal time step
            # dw = W / N, where N is number of point in initial signal

            n = len(signal)
            # dw = band / n
            # w = [dw * (i - n / 2) for i in range(n)]
            w = fftshift([(i - n / 2) * (2.0 * np.pi / t_span) for i in range(n)])
            # w = np.array([(i - n / 2) * (2. * np.pi / t_span) for i in range(n)])
            # w = np.fft.fftfreq(K, d=t_span/n) * 2. * np.pi # Probably better way

        dispersion = np.exp(
            (0.5j * beta2 * w ** 2 + 1.0 / 6.0 * beta3 * w ** 3 - alpha / 2.0) * delta_z
        )

    # F-
    # print(np.mean(dispersion))
    temp_signal = ifft(temp_freq * dispersion)

    return temp_signal


def ssfm_nonlinear_step(signal, gamma, delta_z):
    temp_signal = signal * np.exp(
        1.0j * delta_z * gamma * np.power(np.absolute(signal), 2)
    )

    return temp_signal


def fiber_propogate(
    initial_signal, t_span, fiber_length, n_span, gamma, beta2, alpha=0, beta3=0
):

    if abs(fiber_length) < 1e-15:
        return initial_signal

    dz = fiber_length / n_span

    # D/2
    signal = ssfm_dispersive_step(
        initial_signal, t_span, delta_z=dz / 2.0, beta2=beta2, alpha=alpha, beta3=beta3
    )

    for n in range(n_span):
        signal = ssfm_nonlinear_step(signal, gamma, dz)
        signal = ssfm_dispersive_step(
            signal, t_span, delta_z=dz, beta2=beta2, alpha=alpha, beta3=beta3
        )

    # -D/2
    signal = ssfm_dispersive_step(
        signal, t_span, delta_z=-dz / 2.0, beta2=beta2, alpha=alpha, beta3=beta3
    )

    return signal


def fiber_propogate_high_order(
    initial_signal, t_span, fiber_length, n_span, gamma, beta2, alpha=0, beta3=0
):
    # TODO: check dz and n_power for calculation

    dz = fiber_length / (6 * n_span)

    signal = initial_signal
    # One step gives z + 6dz
    for step in range(n_span):
        # (D/2 N)^4
        for n in range(4):
            signal = ssfm_dispersive_step(signal, t_span, delta_z=dz / 2.0, beta2=beta2)
            signal = ssfm_nonlinear_step(signal, gamma, dz)

        # -D/2
        signal = ssfm_dispersive_step(signal, t_span, delta_z=-dz / 2.0, beta2=beta2)

        # -2N
        signal = ssfm_nonlinear_step(signal, gamma, -2.0 * dz)

        # -D/2
        signal = ssfm_dispersive_step(signal, t_span, delta_z=-dz / 2.0, beta2=beta2)

        # N
        signal = ssfm_nonlinear_step(signal, gamma, dz)

        # (D/2 N)^3
        for n in range(3):
            signal = ssfm_dispersive_step(signal, t_span, delta_z=dz / 2.0, beta2=beta2)
            signal = ssfm_nonlinear_step(signal, gamma, dz)

        # D/2
        signal = ssfm_dispersive_step(signal, t_span, delta_z=dz / 2.0, beta2=beta2)

    return signal


# SSFM GPU version

# NLSE


def tf_ssfm_dispersive_step(signal, dispersion):

    return tf.signal.ifft(tf.signal.fft(signal) * dispersion)


def tf_ssfm_nonlinear_step(signal, gamma, delta_z):

    # signal * np.exp(1.0j * delta_z * gamma * np.power(np.absolute(signal), 2))

    # return signal * tf.math.exp(tf.dtypes.complex(0.0, delta_z * gamma) * tf.math.abs(signal) * tf.math.abs(signal))
    # return tf.math.exp(tf.dtypes.complex(0.0, delta_z * gamma) * tf.math.abs(signal) * tf.math.abs(signal))
    abs_signal = tf.cast(tf.math.abs(signal), tf.complex128)
    return signal * tf.math.exp(
        tf.cast(1.0j * delta_z * gamma, tf.complex128) * abs_signal * abs_signal
    )


def tf_fiber_propogate(
    initial_signal, t_span, fiber_length, n_span, gamma, beta2, alpha=0, beta3=0
):

    if abs(fiber_length) < 1e-15:
        return initial_signal

    dz = fiber_length / n_span

    n = len(initial_signal)
    w = tf.signal.fftshift(
        np.array(
            [(i - n / 2) * (2.0 * np.pi / t_span) for i in range(n)], dtype=np.complex
        )
    )
    w2 = tf.math.pow(w, 2)
    w3 = tf.math.pow(w, 3)
    # dispersion = tf.dtypes.complex(0.0, 0.5) * tf.dtypes.complex(beta2, 0.0) * w2
    # dispersion = tf.math.exp(0.5j * beta2 * tf.math.pow(w, 2))
    dispersion = tf.math.exp(
        (0.5j * beta2 * w2 + 1.0 / 6.0 * beta3 * w3 - alpha / 2.0) * dz
    )
    dispersion_half = tf.math.exp(
        (0.5j * beta2 * w2 + 1.0 / 6.0 * beta3 * w3 - alpha / 2.0) * dz / 2.0
    )
    dispersion_mhalf = tf.math.exp(
        (0.5j * beta2 * w2 + 1.0 / 6.0 * beta3 * w3 - alpha / 2.0) * -dz / 2.0
    )

    # D/2
    signal = tf_ssfm_dispersive_step(initial_signal, dispersion_half)

    for n in range(n_span):
        signal = tf_ssfm_nonlinear_step(signal, gamma, dz)
        signal = tf_ssfm_dispersive_step(signal, dispersion)
        # ...

    # -D/2
    signal = tf_ssfm_dispersive_step(signal, dispersion_mhalf)

    return signal


def tf_ssfm_manakov_dispersive_step(first, second, dispersion):

    first_new = tf.signal.ifft(tf.signal.fft(first) * dispersion)
    second_new = tf.signal.ifft(tf.signal.fft(second) * dispersion)
    return first_new, second_new


def tf_ssfm_manakov_nonlinear_step(first, second, gamma, delta_z):

    abs_first = tf.cast(tf.math.abs(first), tf.complex128)
    abs_second = tf.cast(tf.math.abs(second), tf.complex128)
    first_new = first * tf.math.exp(
        tf.cast(1.0j * delta_z * 8.0 / 9.0 * gamma, tf.complex128)
        * (abs_first * abs_first + abs_second * abs_second)
    )
    second_new = second * tf.math.exp(
        tf.cast(1.0j * delta_z * 8.0 / 9.0 * gamma, tf.complex128)
        * (abs_first * abs_first + abs_second * abs_second)
    )
    return first_new, second_new


def tf_manakov_fiber_propogate(
    initial_first,
    initial_second,
    t_span,
    fiber_length,
    n_span,
    gamma,
    beta2,
    alpha=0,
    beta3=0,
):

    if abs(fiber_length) < 1e-15:
        return initial_first, initial_second

    dz = fiber_length / n_span
    # print(dz)

    if len(initial_first) != len(initial_second):
        print(
            "[tf_manakov_fiber_propogate] Error: sizes of first and second polarisation have to be the same!"
        )
        return initial_first, initial_second

    n = len(initial_first)
    w = tf.signal.fftshift(
        np.array(
            [(i - n / 2) * (2.0 * np.pi / t_span) for i in range(n)], dtype=np.complex
        )
    )
    w2 = tf.math.pow(w, 2)
    w3 = tf.math.pow(w, 3)
    # dispersion = tf.dtypes.complex(0.0, 0.5) * tf.dtypes.complex(beta2, 0.0) * w2
    # dispersion = tf.math.exp(0.5j * beta2 * tf.math.pow(w, 2))
    dispersion = tf.math.exp(
        (0.5j * beta2 * w2 + 1.0 / 6.0 * beta3 * w3 - alpha / 2.0) * dz
    )
    dispersion_half = tf.math.exp(
        (0.5j * beta2 * w2 + 1.0 / 6.0 * beta3 * w3 - alpha / 2.0) * dz / 2.0
    )
    dispersion_mhalf = tf.math.exp(
        (0.5j * beta2 * w2 + 1.0 / 6.0 * beta3 * w3 - alpha / 2.0) * -dz / 2.0
    )

    # D/2
    first, second = tf_ssfm_manakov_dispersive_step(
        initial_first, initial_second, dispersion_half
    )

    for n in range(n_span):
        first, second = tf_ssfm_manakov_nonlinear_step(first, second, gamma, dz)
        first, second = tf_ssfm_manakov_dispersive_step(first, second, dispersion)
        # ...

    # -D/2
    first, second = tf_ssfm_manakov_dispersive_step(first, second, dispersion_mhalf)

    return first, second


# Additional functions


def check_energy(signal, t_span, spectrum):
    # energy_signal = np.mean(np.power(np.absolute(signal), 2)) * t_span
    # energy_spectrum = np.mean(np.power(np.absolute(spectrum), 2)) * (2.0 * np.pi / t_span * len(signal))
    energy_signal = np.mean(np.power(np.absolute(signal), 2))
    energy_spectrum = np.mean(np.power(np.absolute(spectrum), 2)) / len(signal)
    if abs(energy_signal - energy_spectrum) > 1e-14:
        print("Error, energy is different: ", abs(energy_signal - energy_spectrum))

    return energy_signal, energy_spectrum


def get_energy(signal, t_span):
    return np.mean(np.power(np.absolute(signal), 2)) * t_span


def get_gauss_pulse(amplitude, t, tau, z=0, beta2=0):
    z_ld = z / tau ** 2 * abs(beta2)
    a_z = amplitude / np.sqrt(1 - 1.0j * z_ld * np.sign(beta2))

    return a_z * np.exp(
        -0.5 / (1 + z_ld ** 2) * np.power(t / tau, 2) * (1.0 + 1.0j * z_ld)
    )

    # return amplitude * np.exp(-0.5 * np.power(t / tau, 2))


def get_pulse_nonlinear(signal, gamma, z):
    return signal * np.exp(1.0j * gamma * np.power(np.abs(signal), 2) * z)


def get_soliton_pulse(t, tau, soliton_order, beta2, gamma):
    if beta2 > 0:
        print("Error: beta2 > 0")
        beta2 = -beta2
    return np.sqrt(-beta2 * soliton_order ** 2 / (gamma * tau ** 2)) / np.cosh(t / tau)


# WDM signal generation

# Filter shape

# def rrcosfilter_our(N, alpha, Ts, Fs):
#     """
#     Generates a root raised cosine (RRC) filter (FIR) impulse response.
#
#     Parameters
#     ----------
#     N : int
#         Length of the filter in samples.
#
#     alpha : float
#         Roll off factor (Valid values are [0, 1]).
#
#     Ts : float
#         Symbol period in seconds.
#
#     Fs : float
#         Sampling Rate in Hz.
#
#     Returns
#     ---------
#
#     time_idx : 1-D ndarray of floats
#         Array containing the time indices, in seconds, for
#         the impulse response.
#
#     h_rrc : 1-D ndarray of floats
#         Impulse response of the root raised cosine filter.
#     """
#
#     T_delta = 1/float(Fs)
#     time_idx = ((np.arange(N)-N/2))*T_delta
#     sample_num = np.arange(N)
#     h_rrc = np.zeros(N, dtype=float)
#
#     for x in sample_num:
#         t = (x-N/2)*T_delta
#         if t == 0.0:
#             h_rrc[x] = 1.0 - alpha + (4*alpha/np.pi)
#         elif alpha != 0 and t == Ts/(4*alpha):
#             h_rrc[x] = (alpha/np.sqrt(2))*(((1+2/np.pi)* \
#                     (np.sin(np.pi/(4*alpha)))) + ((1-2/np.pi)*(np.cos(np.pi/(4*alpha)))))
#         elif alpha != 0 and t == -Ts/(4*alpha):
#             h_rrc[x] = (alpha/np.sqrt(2))*(((1+2/np.pi)* \
#                     (np.sin(np.pi/(4*alpha)))) + ((1-2/np.pi)*(np.cos(np.pi/(4*alpha)))))
#         else:
#             h_rrc[x] = (np.sin(np.pi*t*(1-alpha)/Ts) +  \
#                     4*alpha*(t/Ts)*np.cos(np.pi*t*(1+alpha)/Ts))/ \
#                     (np.pi*t*(1-(4*alpha*t/Ts)*(4*alpha*t/Ts))/Ts)
#
#     return h_rrc

# def rrcosfilter_our(N, alpha, Ts, Fs):
#     """
#     Generates a root raised cosine (RRC) filter (FIR) impulse response.
#     Parameters
#     ----------
#     N : int
#         Length of the filter in samples.
#     alpha : float
#         Roll off factor (Valid values are [0, 1]).
#     Ts : float
#         Symbol period in seconds.
#     Fs : float
#         Sampling Rate in Hz.
#     Returns
#     ---------
#     time_idx : 1-D ndarray of floats
#         Array containing the time indices, in seconds, for
#         the impulse response.
#     h_rrc : 1-D ndarray of floats
#         Impulse response of the root raised cosine filter.
#     """
#     T_delta = 1/float(Fs)
#     time_idx = ((np.arange(N)-N/2))*T_delta
#     sample_num = np.arange(N)
#     h_rrc = np.zeros(N, dtype=float)
#
#     for x in sample_num:
#         t = (x-N/2)*T_delta
#         if np.isclose(t, 0.0, atol=1e-16, rtol=1e-15):
#             h_rrc[x] = 1.0 - alpha + (4*alpha/np.pi)
#         elif alpha != 0 and np.isclose(t, Ts/(4*alpha), atol=1e-16, rtol=1e-15):
#             h_rrc[x] = (alpha/np.sqrt(2))*(((1+2/np.pi)* \
#                     (np.sin(np.pi/(4*alpha)))) + ((1-2/np.pi)*(np.cos(np.pi/(4*alpha)))))
#         elif alpha != 0 and np.isclose(t, -Ts/(4*alpha), atol=1e-16, rtol=1e-15):
#             h_rrc[x] = (alpha/np.sqrt(2))*(((1+2/np.pi)* \
#                     (np.sin(np.pi/(4*alpha)))) + ((1-2/np.pi)*(np.cos(np.pi/(4*alpha)))))
#         else:
#             h_rrc[x] = (np.sin(np.pi*t*(1-alpha)/Ts) +  \
#                     4*alpha*(t/Ts)*np.cos(np.pi*t*(1+alpha)/Ts)) / \
#                     (np.pi*t*(1-(4*alpha*t/Ts)*(4*alpha*t/Ts))/Ts)
#
#     return h_rrc
def rrcosfilter_new(nt, beta, t_symb, sample_rate):

    one_over_ts = 1.0 / t_symb
    dt = 1.0 / float(sample_rate)
    t = (np.arange(nt) - nt / 2.0) * dt
    rrc = np.zeros(nt, dtype=np.float)

    # found ranges for conditions
    zero_pos = np.where(np.isclose(t, 0.0, atol=1e-16, rtol=1e-15))
    if beta != 0:
        nodes_pos = np.where(
            np.isclose(abs(t), 0.25 * t_symb / beta, atol=1e-16, rtol=1e-15)
        )
        all_pos = np.where(
            ~(
                np.isclose(abs(t), 0.25 * t_symb / beta, atol=1e-16, rtol=1e-15)
                | np.isclose(t, 0.0, atol=1e-16, rtol=1e-15)
            )
        )

    else:
        all_pos = np.where(~np.isclose(t, 0.0, atol=1e-16, rtol=1e-15))

    if beta != 0 and np.shape(nodes_pos)[1] != 0:
        nodes_values = (
            np.ones(len(t[nodes_pos]), dtype=float)
            * beta
            * one_over_ts
            / np.sqrt(2)
            * (
                (1.0 + 2.0 / np.pi) * np.sin(0.25 * np.pi / beta)
                + (1.0 - 2.0 / np.pi) * np.cos(0.25 * np.pi / beta)
            )
        )
        rrc[nodes_pos] = nodes_values

    if np.shape(zero_pos)[1] != 0:
        rrc[zero_pos] = one_over_ts * (1.0 + beta * (4.0 / np.pi - 1))

    all_values = (
        np.sin(np.pi * (1.0 - beta) * t[all_pos] * one_over_ts)
        + 4.0
        * beta
        * t[all_pos]
        * one_over_ts
        * np.cos(np.pi * (1.0 + beta) * t[all_pos] * one_over_ts)
    ) / (
        np.pi * t[all_pos] * (1.0 - np.power(4.0 * beta * t[all_pos] * one_over_ts, 2))
    )
    rrc[all_pos] = all_values

    return rrc


def rrcosfilter_our(N, alpha, Ts, Fs):

    return rrcosfilter_new(N, alpha, Ts, Fs) * Ts


def tf_convolution(signal, filter_val):

    np_signal_orig = len(signal)
    # conv_out = tf.zeros((len(signal)), dtype=tf.complex128)
    # conv_out = np.zeros((len(signal)), dtype=complex)
    np_filter = len(filter_val)
    add_zeros = np.array([0.0 for i in range(np_filter)])
    signal_ext = tf.concat((add_zeros, signal, add_zeros), axis=0)
    # print(np.shape(signal), np.shape(signal_ext), np.shape(filter_val))
    np_signal = len(signal_ext)
    # for n in range(np_filter, np_signal + np_filter):
    #     # tf.reduce_sum(filter_val * signal_ext[n - np_filter: n])
    #     # print(np.shape())
    #     conv_out[n - np_filter] = tf.reduce_sum(filter_val * signal_ext[n - np_filter : n])

    conv_out = np.array(
        [
            tf.reduce_sum(filter_val * signal_ext[n - np_filter : n])
            for n in range(np_filter, np_signal_orig + np_filter)
        ]
    )

    conv_out = tf.cast(conv_out, tf.complex128)
    return conv_out


def filter_shaper(signal, filter_val):

    spectrum = tf.signal.fftshift(tf.signal.fft(signal))
    return tf.signal.ifft(tf.signal.ifftshift(spectrum * filter_val))

    # return tf_convolution(signal, filter_val)
    # return np.convolve(signal, filter_val)


def matched_filter(signal, filter_val):
    return filter_shaper(signal, filter_val) / tf.cast(
        tf.reduce_sum(tf.math.abs(filter_val)), tf.complex128
    )


# Coding / decodong


def Gray_alphabet(bm):
    import numpy as np

    gseq = np.empty((2 ** bm, bm), dtype=int)
    for i in range(2 ** bm):
        buf = i ^ (i >> 1)
        buf = np.asarray([int(x) for x in bin(buf)[2:]])
        gseq[i, :] = np.append(np.zeros(bm - buf.size, dtype=int), buf)
    return gseq


def Gray_QAM_bit_abc(m):
    import numpy as np

    bm = int(m / 2)
    gseq = Gray_alphabet(bm)
    gabc = np.concatenate(
        (np.tile(gseq, reps=(2 ** bm, 1)), np.repeat(gseq, repeats=2 ** bm, axis=0)),
        axis=1,
    )
    return gabc


def Gray_QAM_sym_abc(m, norm=True):
    import numpy as np

    ms = int(np.sqrt(2 ** m))
    abc_side = np.arange(0, ms) * 2 - (ms - 1)
    QAM_abc = np.tile(abc_side, reps=(ms)) + 1j * np.repeat(
        np.flip(abc_side, axis=0), repeats=ms, axis=0
    )
    if norm:
        QAM_abc = QAM_abc / np.std(QAM_abc)
    return QAM_abc


def hard_slice(QAMsyms, m, norm=True):
    import numpy as np

    alphabet = Gray_QAM_sym_abc(m, norm)
    sym_indices = list(map(lambda sym: np.argmin(np.abs(sym - alphabet)), QAMsyms))
    return alphabet[sym_indices], sym_indices


def QAM2gray_bits(QAMsyms, QAM_order, norm=True):
    # Converts vector QAM complex-valued symbols to the Gray coded bits
    # QAMsyms - QAM symbol vector to convert
    # QAM_order - order of the QAM target alphabet (e.g. 16 for 16QAM)
    # norm - whether the targer QAM alphabet has unitary power
    import numpy as np

    m = np.log2(QAM_order)  # Number of bits per QAM symbol

    # Popular error tracking
    if np.mod(m, 1.0) != 0.0:
        raise ValueError("Given QAM order should be some power of 2.")
    if np.mod(m, 2.0) != 0.0:
        raise ValueError(
            "Non-square constellations are not supported (e.g. 32QAM, 128QAM)"
        )
    if QAMsyms.ndim != 1:
        raise ValueError("Input array of QAM symbols must be an array")

    m = int(m)  # Convert bit number to integer after checking its value
    QAM_indices = hard_slice(QAMsyms, m, norm)[
        1
    ]  # Hard slice the input QAM sequence and return its
    bit_alphabet = Gray_QAM_bit_abc(
        m
    )  # Bit patterns, corresponding to every symbol from QAM alphabet
    bit_seq = np.concatenate(
        tuple((bit_alphabet[QAM_ind] for QAM_ind in QAM_indices)), axis=0
    )
    return bit_seq


def QAM_BER_gray(QAMsyms_chk, QAMsyms_ref, QAM_order, norm=True):
    # Calculates BER between the two QAM symbol vectors in input data
    # QAMsyms - QAM symbol vector to convert
    # QAM_order - order of the QAM target alphabet (e.g. 16 for 16QAM)
    # norm - whether the targer QAM alphabet has unitary power
    import numpy as np

    bits_chk = QAM2gray_bits(QAMsyms_chk, QAM_order, norm)
    bits_ref = QAM2gray_bits(QAMsyms_ref, QAM_order, norm)
    BER = np.mean(np.logical_xor(bits_ref, bits_chk))
    return BER


def BER_est(m_order, x, x_ref):
    return QAM_BER_gray(x, x_ref, m_order)


####### Gray code ##########

# Helper function to xor two characters
def xor_c(a, b):
    return int(0) if (a == b) else int(1)


# Helper function to flip the bit
def flip(c):
    return int(1) if (c == int(0)) else int(0)


# function to convert binary string
# to gray string
def binarytoGray(binary, num_bits_symbol):
    gray = binary
    # gray = np.zeros((len(binary),), dtype=int)
    # # MSB of gray code is same as
    # # binary code
    # NN = int(len(binary) / num_bits_symbol)
    # count = 0
    # for k in range(0, NN):
    #     gray[count] = binary[count]
    #     for j in range(count + 1, count + num_bits_symbol):
    #         gray[j] = xor_c(binary[j - 1], binary[j])
    #     count = count + num_bits_symbol
    return gray


# function to convert gray code
# string to binary string
def graytoBinary(gray, num_bits):
    binary = gray
    # NN = int(len(gray) / num_bits)
    # binary = np.zeros((len(gray),), dtype=int)
    # count = 0
    # for k in range(0, NN):
    #     binary[count] = gray[count]
    #     for j in range(count + 1, count + num_bits):
    #         if gray[j] == 0:
    #             binary[j] = binary[j - 1]
    #         else:
    #             binary[j] = flip(binary[j - 1])
    #     count = count + num_bits
    return binary


# Channel parameters


def get_default_channel_parameters():

    channel = {}
    channel["n_spans"] = 12  # Number of spans
    channel["z_span"] = 80  # Span Length [km]
    channel["alpha_db"] = 0.225  # Attenuation coefficient [dB km^-1]
    channel["alpha"] = channel["alpha_db"] / (10 * np.log10(np.exp(1)))
    channel["gamma"] = 1.2  # Non-linear Coefficient [W^-1 km^-1]. Default = 1.2
    channel["noise_figure_db"] = 4.5  # Noise Figure [dB]. Default = 4.5
    channel["noise_figure"] = 10 ** (channel["noise_figure_db"] / 10)
    channel["gain"] = np.exp(channel["alpha"] * channel["z_span"])  # gain for one span
    channel["dispersion_parameter"] = 16.8  #  [ps nm^-1 km^-1]  dispersion parameter
    channel["beta2"] = (
        -(1550e-9 ** 2) * (channel["dispersion_parameter"] * 1e-3) / (2 * np.pi * 3e8)
    )  # conversion to beta2 - Chromatic Dispersion Coefficient [s^2 km^−1]
    channel["beta3"] = 0
    channel["h_planck"] = 6.6256e-34  # Planck's constant [J/s]
    channel["fc"] = 299792458 / 1550e-9  # carrier frequency
    channel["dz"] = 1.0  # length of the step for SSFM [km]
    channel["nz"] = int(
        channel["z_span"] / channel["dz"]
    )  # number of steps per each span
    channel["noise_density"] = (
        channel["h_planck"]
        * channel["fc"]
        * (channel["gain"] - 1)
        * channel["noise_figure"]
    )

    return channel


def create_channel_parameters(
    n_spans, z_span, alpha_db, gamma, noise_figure_db, dispersion_parameter, dz
):

    alpha = alpha_db / (10 * np.log10(np.exp(1)))
    noise_figure = 10 ** (noise_figure_db / 10)
    gain = np.exp(alpha * z_span)  # gain for one span
    beta2 = (
        -(1550e-9 ** 2) * (dispersion_parameter * 1e-3) / (2 * np.pi * 3e8)
    )  # conversion to beta2 - Chromatic Dispersion Coefficient [s^2 km^−1]
    beta3 = 0
    h_planck = 6.6256e-34  # Planck's constant [J/s]
    # nu = 299792458 / 1550e-9  # light frequency carrier [Hz]
    fc = 299792458 / 1550e-9  # carrier frequency
    nz = int(z_span / dz)  # number of steps per each span
    noise_density = h_planck * fc * (gain - 1) * noise_figure

    channel = {}
    channel["n_spans"] = n_spans  # Number of spans
    channel["z_span"] = z_span  # Span Length [km]
    channel["alpha_db"] = alpha_db  # Attenuation coefficient [dB km^-1]
    channel["alpha"] = alpha
    channel["gamma"] = gamma  # Non-linear Coefficient [W^-1 km^-1]. Default = 1.2
    channel["noise_figure_db"] = noise_figure_db  # Noise Figure [dB]. Default = 4.5
    channel["noise_figure"] = noise_figure
    channel["gain"] = gain  # gain for one span
    channel[
        "dispersion_parameter"
    ] = dispersion_parameter  # [ps nm^-1 km^-1]  dispersion parameter
    channel[
        "beta2"
    ] = beta2  # conversion to beta2 - Chromatic Dispersion Coefficient [s^2 km^−1]
    channel["beta3"] = beta3
    channel["h_planck"] = h_planck  # Planck's constant [J/s]
    channel["fc"] = h_planck  # carrier frequency
    channel["dz"] = dz  # length of the step for SSFM [km]
    channel["nz"] = nz  # number of steps per each span
    channel["noise_density"] = noise_density

    return channel


def create_wdm_parameters(
    n_channels,
    p_ave_dbm,
    n_symbols,
    m_order,
    roll_off,
    upsampling,
    downsampling_rate,
    symb_freq,
    channel_spacing,
    n_polarisations=2,
    np_filter=0,
):
    wdm = {}
    wdm["n_channels"] = n_channels
    wdm["channel_spacing"] = channel_spacing
    wdm["n_polarisations"] = n_polarisations
    wdm["p_ave_dbm"] = p_ave_dbm
    wdm["n_symbols"] = n_symbols
    wdm["m_order"] = m_order
    wdm["roll_off"] = roll_off
    wdm["upsampling"] = upsampling
    wdm["downsampling_rate"] = downsampling_rate
    wdm["symb_freq"] = symb_freq
    wdm["np_filter"] = np_filter
    wdm["p_ave"] = (10 ** (wdm["p_ave_dbm"] / 10)) / 1000

    return wdm


def get_default_wdm_parameters():

    wdm = {}
    wdm["n_channels"] = 1
    wdm["channel_spacing"] = 50
    wdm["n_polarisations"] = 2
    wdm["p_ave_dbm"] = 0
    wdm["n_symbols"] = 2 ** 15
    wdm["m_order"] = 16
    wdm["roll_off"] = 0.1
    wdm["upsampling"] = 8
    wdm["downsampling_rate"] = 1
    wdm["symb_freq"] = 64e9
    wdm["np_filter"] = 2 ** 12
    wdm["p_ave"] = (10 ** (wdm["p_ave_dbm"] / 10)) / 1000

    return wdm


def generate_wdm_base(wdm, bits=None):

    symb_freq = int(wdm["symb_freq"])  # symbol frequency
    sample_freq = int(
        symb_freq * wdm["upsampling"]
    )  # sampling frequency used for the discrete simulation of analog signals
    # dt = 1 / sample_freq
    t_s = 1 / symb_freq  # symbol spacing
    # bandwidth = 1 / (2 * t_s)  # Nyquist bandwidth of the base band signal
    ups = int(t_s * sample_freq)  # Number of samples per second in the analog domain
    # np_filter = 2 ** 12  # Filter length in symbols

    modem = commpy.QAMModem(wdm["m_order"])
    n_bits = int(modem.num_bits_symbol * wdm["n_symbols"])
    if bits is None:
        bits = np.random.randint(0, 2, n_bits, int)  # Random bit stream
    else:
        if len(bits) != n_bits:
            print(
                "[generate_wdm_base] Error: length of input bits does not correspond to the parameters"
            )
    gray = binarytoGray(bits, modem.num_bits_symbol)  # after gray code
    points = modem.modulate(gray) / np.sqrt(
        modem.Es
    )  # Modulated baud points sQ = mod1.modulate(sB)/np.sqrt(mod1.Es)
    points = points * np.sqrt(wdm["p_ave"])

    points_sequence = np.zeros(ups * wdm["n_symbols"], dtype="complex")
    points_sequence[
        ::ups
    ] = points  # every ups samples, the value of sQ is inserted into the sequence
    points_sequence = tf.cast(points_sequence, tf.complex128)

    np_sequence = len(points_sequence)

    ft_filter_values = tf.signal.fftshift(
        tf.signal.fft(rrcosfilter_our(np_sequence, wdm["roll_off"], t_s, sample_freq))
    )
    ft_filter_values = tf.cast(ft_filter_values, tf.complex128)
    signal_x = filter_shaper(points_sequence, ft_filter_values)

    additional = {
        "ft_filter_values": ft_filter_values,
        "bits": bits,
        "gray": gray,
        "points": points,
    }

    return tf.cast(signal_x, tf.complex128), additional


def generate_wdm_new(wdm):

    # n_symbols - Number of Symbols transmitted
    # m_order - Modulation Level
    # roll_off
    # upsampling
    # downsampling_rate

    # Check input parameters
    if not (wdm["n_polarisations"] == 1 or wdm["n_polarisations"] == 2):
        print("[generate_wdm] Error: wrong number of polarisations")
        return -1

    start_time = datetime.now()

    symb_freq = int(wdm["symb_freq"])  # symbol frequency
    sample_freq = int(
        symb_freq * wdm["upsampling"]
    )  # sampling frequency used for the discrete simulation of analog signals
    dt = 1.0 / sample_freq
    dw = wdm["channel_spacing"]

    bits = []
    gray = []
    points = []
    ft_filter_values = []

    if wdm["n_polarisations"] == 2:
        wdm["p_ave"] = wdm["p_ave"] / 2

    for wdm_index in range(wdm["n_channels"]):
        if wdm["n_polarisations"] == 1:
            signal_temp, additional = generate_wdm_base(wdm)
            if wdm_index == 0:
                signal = signal_temp
                np_signal = len(signal)
                t = np.array([dt * (k - np_signal / 2) for k in range(np_signal)])
            else:
                signal += signal_temp

            bits.append(additional["bits"])
            gray.append(additional["gray"])
            points.append(additional["points"])
            ft_filter_values.append(additional["ft_filter_values"])

        elif wdm["n_polarisations"] == 2:
            signal_x_temp, additional_x = generate_wdm_base(wdm)
            signal_y_temp, additional_y = generate_wdm_base(wdm)

            if wdm_index == 0:
                signal_x = signal_x_temp
                signal_y = signal_y_temp
                np_signal = len(signal_x)
                t = np.array([dt * (k - np_signal / 2) for k in range(np_signal)])
            else:
                signal_x += signal_x_temp * np.exp(
                    1.0j * dw * t
                )  # TODO: finish multichannel
                signal_y += signal_y_temp * np.exp(1.0j * dw * t)

            bits.append(additional_x["bits"])
            bits.append(additional_y["bits"])
            gray.append(additional_x["gray"])
            gray.append(additional_y["gray"])
            points.append(additional_x["points"])
            points.append(additional_y["points"])
            ft_filter_values.append(additional_x["ft_filter_values"])
            ft_filter_values.append(additional_y["ft_filter_values"])

    end_time = datetime.now()
    time_diff = end_time - start_time
    execution_time = time_diff.total_seconds() * 1000
    # print("Signal generation took", execution_time, "ms")

    additional_all = {
        "ft_filter_values": ft_filter_values,
        "bits": bits,
        "gray": gray,
        "points": points,
    }

    if wdm["n_polarisations"] == 1:
        return tf.cast(signal, tf.complex128), additional_all
    else:
        return (
            tf.cast(signal_x, tf.complex128),
            tf.cast(signal_y, tf.complex128),
            additional_all,
        )


def generate_wdm(wdm, bits_x=None, bits_y=None, points_x=None, points_y=None):

    # n_symbols - Number of Symbols transmitted
    # m_order - Modulation Level
    # roll_off
    # upsampling
    # downsampling_rate

    symb_freq = int(wdm["symb_freq"])  # symbol frequency
    sample_freq = int(
        symb_freq * wdm["upsampling"]
    )  # sampling frequency used for the discrete simulation of analog signals
    dt = 1 / sample_freq
    t_s = 1 / symb_freq  # symbol spacing
    bandwidth = 1 / (2 * t_s)  # Nyquist bandwidth of the base band signal
    ups = int(t_s * sample_freq)  # Number of samples per second in the analog domain
    # np_filter = 2 ** 12  # Filter length in symbols

    start_time = datetime.now()

    #  ######## INITIATE I, Q and noise components of polarization X #########
    p_ave_x_dbm = wdm["p_ave_dbm"]  # dBm
    p_ave_x = (10 ** (p_ave_x_dbm / 10)) / 1000 / 2
    modem_x = commpy.QAMModem(wdm["m_order"])
    n_bits_x = int(modem_x.num_bits_symbol * wdm["n_symbols"])

    if bits_x is None:
        bits_x = np.random.randint(0, 2, n_bits_x, int)  # Random bit stream
    else:
        if len(bits_x) != n_bits_x:
            print(
                "[generate_wdm_old] Error: length of input bits does not correspond to the parameters"
            )

    # bits_x = np.random.randint(0, 2, n_bits_x, int)  # Random bit stream
    gray_x = binarytoGray(bits_x, modem_x.num_bits_symbol)  # after gray code
    if points_x is None:
        points_x = modem_x.modulate(gray_x) / np.sqrt(
            modem_x.Es
        )  # Modulated baud points sQ = mod1.modulate(sB)/np.sqrt(mod1.Es)
        points_x = points_x * np.sqrt(p_ave_x)

    #  ######## INITIATE I, Q and noise components of polarization Y #########
    p_ave_y_dbm = wdm["p_ave_dbm"]  # dBm
    p_ave_y = (10 ** (p_ave_y_dbm / 10)) / 1000 / 2
    modem_y = commpy.QAMModem(wdm["m_order"])
    n_bits_y = int(modem_y.num_bits_symbol * wdm["n_symbols"])

    if bits_y is None:
        bits_y = np.random.randint(0, 2, n_bits_y, int)  # Random bit stream
    else:
        if len(bits_y) != n_bits_y:
            print(
                "[generate_wdm_old] Error: length of input bits does not correspond to the parameters"
            )

    # bits_y = np.random.randint(0, 2, n_bits_y, int)  # Random bit stream
    gray_y = binarytoGray(bits_y, modem_y.num_bits_symbol)  # after gray code
    if points_y is None:
        points_y = modem_y.modulate(gray_y) / np.sqrt(
            modem_y.Es
        )  # Modulated baud points sQ = mod1.modulate(sB)/np.sqrt(mod1.Es)
        points_y = points_y * np.sqrt(p_ave_y)

    #  ######## Turning the Discrete signal in Countinuous  X #########

    points_sequence_x = np.zeros(ups * wdm["n_symbols"], dtype="complex")
    points_sequence_x[
        ::ups
    ] = points_x  # every ups samples, the value of sQ is inserted into the sequence
    points_sequence_x = tf.cast(points_sequence_x, tf.complex128)

    #  ######## Turning the Discrete signal in Countinuous  Y #########

    points_sequence_y = np.zeros(ups * wdm["n_symbols"], dtype="complex")
    points_sequence_y[
        ::ups
    ] = points_y  # every ups samples, the value of sQ is inserted into the sequence
    points_sequence_y = tf.cast(points_sequence_y, tf.complex128)

    #  ######## Root Raised Cosine Filter X #########

    np_sequence = len(points_sequence_x)
    # add_zeros = np.array([0. for  i in range(int((np_xxx - filtlen) / 2))])

    # filtlen = np_xxx
    ft_filter_values = tf.signal.fftshift(
        tf.signal.fft(rrcosfilter_our(np_sequence, wdm["roll_off"], t_s, sample_freq))
    )
    # filter_values = rrcosfilter_our(np_filter, roll_off, t_s, sample_freq * int(np_sequence / np_filter))
    # print(np.shape(filter_values))
    # filter_values_ups = np.zeros((np_sequence), dtype=complex)
    # filter_values_ups[::int(np_sequence / np_filter)] = filter_values
    # print(len(filter_values_ups), np_sequence)
    # filter_values = np.concatenate((add_zeros, filter_values, add_zeros), axis=0)

    # filter_values_ups = tf.cast(filter_values_ups, tf.complex128)
    # ft_filter_values = tf.signal.fftshift(tf.signal.fft(filter_values_ups))

    ft_filter_values = tf.cast(ft_filter_values, tf.complex128)
    signal_x = filter_shaper(points_sequence_x, ft_filter_values)
    signal_y = filter_shaper(points_sequence_y, ft_filter_values)

    end_time = datetime.now()
    time_diff = end_time - start_time
    execution_time = time_diff.total_seconds() * 1000
    # print("Signal generation took", execution_time, "ms")

    additional = {
        "ft_filter_values": ft_filter_values,
        "bits_y": bits_y,
        "gray_y": gray_y,
        "points_y": points_y,
        "bits_x": bits_x,
        "gray_x": gray_x,
        "points_x": points_x,
    }

    return (
        tf.cast(signal_x, tf.complex128),
        tf.cast(signal_y, tf.complex128),
        additional,
    )


def propagate_manakov(channel, signal_x, signal_y, sample_freq):

    dt = 1 / sample_freq
    nt = len(signal_x)
    # print(nt)
    t_span = dt * nt
    start_time = datetime.now()

    sq_gain = tf.cast(tf.math.sqrt(channel["gain"]), tf.complex128)
    std = tf.cast(tf.math.sqrt(channel["noise_density"] * sample_freq), tf.complex128)
    one_over_sq_2 = tf.cast(1.0 / tf.math.sqrt(2.0), tf.complex128)

    for span_ind in range(channel["n_spans"]):
        signal_x, signal_y = tf_manakov_fiber_propogate(
            signal_x,
            signal_y,
            t_span,
            channel["z_span"],
            channel["nz"],
            channel["gamma"],
            channel["beta2"],
            alpha=channel["alpha"],
            beta3=channel["beta3"],
        )
        #
        # noise_x = (np.random.normal(0, 1, size=nt) + 1.0j * np.random.normal(0, 1, size=nt)) * one_over_sq_2
        # noise_y = (np.random.normal(0, 1, size=nt) + 1.0j * np.random.normal(0, 1, size=nt)) * one_over_sq_2

        noise_x = (
            tf.complex(
                tf.random.normal([nt], 0, 1, dtype=tf.float64),
                tf.random.normal([nt], 0, 1, dtype=tf.float64),
            )
            * one_over_sq_2
        )
        noise_y = (
            tf.complex(
                tf.random.normal([nt], 0, 1, dtype=tf.float64),
                tf.random.normal([nt], 0, 1, dtype=tf.float64),
            )
            * one_over_sq_2
        )

        signal_x = sq_gain * signal_x + noise_x * std
        signal_y = sq_gain * signal_y + noise_y * std

    end_time = datetime.now()
    time_diff = end_time - start_time
    execution_time = time_diff.total_seconds() * 1000
    # print("Signal propagation took", execution_time, "ms")

    return signal_x, signal_y


def receiver(signal_x, signal_y, ft_filter_values, downsampling_rate):

    start_time = datetime.now()
    signal_x = matched_filter(signal_x, ft_filter_values)
    signal_y = matched_filter(signal_y, ft_filter_values)

    signal_x = signal_x[::downsampling_rate]  # downsample
    signal_y = signal_y[::downsampling_rate]

    end_time = datetime.now()
    time_diff = end_time - start_time
    execution_time = time_diff.total_seconds() * 1000
    # print("Matched filter took", execution_time, "ms")

    return signal_x, signal_y


def dispersion_compensation(channel, signal_x, signal_y, dt):

    #  Dispersion compensation #
    nt_cdc = len(signal_x)
    t_span = nt_cdc * dt
    w = tf.signal.fftshift(
        np.array(
            [(i - nt_cdc / 2) * (2.0 * np.pi / t_span) for i in range(nt_cdc)],
            dtype=np.complex,
        )
    )
    w2 = tf.math.pow(w, 2)
    w3 = tf.math.pow(w, 3)
    dispersion = tf.math.exp(
        (0.5j * channel["beta2"] * w2 + 1.0 / 6.0 * channel["beta3"] * w3)
        * (-channel["z_span"] * channel["n_spans"])
    )
    signal_cdc_x, signal_cdc_y = tf_ssfm_manakov_dispersive_step(
        tf.cast(signal_x, tf.complex128), tf.cast(signal_y, tf.complex128), dispersion
    )

    return signal_cdc_x, signal_cdc_y


def nonlinear_shift(points, points_orig):

    return np.dot(np.transpose(np.conjugate(points_orig)), points_orig) / np.dot(
        np.transpose(np.conjugate(points_orig)), points
    )


def full_line_model_default():

    # Specify channel parameters

    n_spans = 12
    z_span = 80
    alpha_db = 0.2
    gamma = 1.2
    noise_figure_db = 4.5
    dispersion_parameter = 16.8
    dz = 1
    channel = create_channel_parameters(
        n_spans, z_span, alpha_db, gamma, noise_figure_db, dispersion_parameter, dz
    )

    # or you can use default parameters
    # channel = get_default_channel_parameters()

    # Specify signal parameters

    wdm = create_wdm_parameters()

    return full_line_model(channel, wdm)


def full_line_model(
    channel, wdm, bits_x=None, bits_y=None, points_x=None, points_y=None
):

    sample_freq = int(wdm["symb_freq"] * wdm["upsampling"])  # in our case , it's 2^21
    dt = 1.0 / sample_freq

    signal_x, signal_y, wdm_info = generate_wdm(
        wdm, bits_x=bits_x, bits_y=bits_y, points_x=points_x, points_y=points_y
    )
    points_orig_x = wdm_info["points_x"]
    points_orig_y = wdm_info["points_y"]
    ft_filter_values = wdm_info["ft_filter_values"]
    np_signal = len(signal_x)

    e_signal_x = get_energy(signal_x, dt * np_signal)
    e_signal_y = get_energy(signal_y, dt * np_signal)

    signal_x, signal_y = propagate_manakov(channel, signal_x, signal_y, sample_freq)

    e_signal_x_prop = get_energy(signal_x, dt * np_signal)
    e_signal_y_prop = get_energy(signal_y, dt * np_signal)

    # print("Signal energy before propagation (x / y):", e_signal_x, e_signal_y)
    # print("Signal energy after propagation (x / y):", e_signal_x_prop, e_signal_y_prop)
    # print("Signal energy difference (x / y):",
    #       np.absolute(e_signal_x - e_signal_x_prop),
    #       np.absolute(e_signal_y - e_signal_y_prop))

    samples_x, samples_y = receiver(
        signal_x, signal_y, ft_filter_values, wdm["downsampling_rate"]
    )
    samples_x, samples_y = dispersion_compensation(
        channel, samples_x, samples_y, dt * wdm["downsampling_rate"]
    )

    sample_step = int(wdm["upsampling"] / wdm["downsampling_rate"])
    points_x = samples_x[::sample_step].numpy()
    points_y = samples_y[::sample_step].numpy()

    nl_shift_x = nonlinear_shift(points_x, points_orig_x)
    points_x_shifted = points_x * nl_shift_x

    nl_shift_y = nonlinear_shift(points_y, points_orig_y)
    points_y_shifted = points_y * nl_shift_y

    # print("BER (x / y):", BER_est(wdm['m_order'], points_x_shifted, points_orig_x), BER_est(wdm['m_order'], points_y_shifted, points_orig_y))

    result = {
        "points_x": points_x,
        "points_orig_x": points_orig_x,
        "points_x_shifted": points_x_shifted,
        "points_y": points_y,
        "points_orig_y": points_orig_y,
        "points_y_shifted": points_y_shifted,
    }

    return result
