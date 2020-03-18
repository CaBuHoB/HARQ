import numpy as np
import numpy.polynomial.polynomial as poly


def coder(m_x, g_x):
    r = g_x.size - 1
    c_x = np.mod(poly.polydiv(np.append(np.zeros(r), m_x), g_x), 2)[1]
    return np.append(c_x, m_x) if c_x.size == r else np.concatenate((c_x, np.zeros(r - c_x.size), m_x))


def decoder(b_x, g_x):
    return np.mod(poly.polydiv(b_x, g_x), 2)[1].max() == 0


k = 8
g_x = np.flip(np.array([1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1]))

messages = np.array([np.flip([int(j) for j in np.binary_repr(i, width=k)]) for i in np.arange(2 ** k)])
codewords_bits = np.array([coder(mes, g_x) for mes in messages])
codewords = codewords_bits * 2 - 1

np.save('./../Tables/messages_2^{}.npy'.format(k), messages)
np.save('./../Tables/codewords_bits_2^{}.npy'.format(k), codewords_bits)
np.save('./../Tables/codewords_2^{}.npy'.format(k), codewords)
