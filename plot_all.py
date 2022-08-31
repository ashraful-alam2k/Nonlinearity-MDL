import numpy as np
import matplotlib.pyplot as plt

symbol_rate_test = [x for x in range(40, 100, 1)]
with_sr_feature = np.load("with_SR_feature.npy")
without_sr_feature = np.load("without_SR_feature.npy")
only40 = np.load("only40_power.npy")
only99 = np.load("only99_1000.npy")
ref = np.load("with_SR_feature_ref.npy")
x = np.arange(40, 105, 5)
y = np.arange(0, 11, 1)

plt.plot(symbol_rate_test, with_sr_feature, label="With SR feature", c="green")
plt.plot(symbol_rate_test, without_sr_feature, label="Without SR feature", c="fuchsia")
plt.plot(symbol_rate_test, only40, label="With only SR 40", c="teal")
plt.plot(symbol_rate_test, only99, label="With only SR 99", c="gold")
plt.plot(symbol_rate_test, ref, label="Exp ref no NN", c="red")
plt.legend(prop={"size": 6})
plt.xlabel("Symbol Rate [GBd]")
plt.ylabel("Q-Factor [dB]")
plt.xticks(x)
plt.yticks(y)
plt.title("Q-Factor for different methods")
plt.grid()
plt.show()
