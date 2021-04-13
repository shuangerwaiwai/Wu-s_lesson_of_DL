import numpy as np
import h5py
import matplotlib.pyplot as plt

#%matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#%load_ext autoreload
#%autoreload 2

np.random.seed(1)

def zere_pad(X, pad):
    X_paded = np.pad(X,(
        (0,0),
        (pad, pad),
        (pad, pad),
        (0, 0)),
        'constant', constant_values=0
    )
    return X_paded

# np.random.seed(1)
# x = np.random.randn(4, 3, 3, 2)
# x_paded = zere_pad(x, 2)
#
# print("x.shape = ", x.shape)
# print("x_paded.shape = ", x_paded.shape)
# print("x[1, 1] = ", x[1,1])
# print("x_paded[1, 1] = ", x_paded[1, 1])
#
# fig, axarr = plt.subplots(1, 2)
# axarr[0].set_title('x')
# axarr[0].imshow(x[0,:,:,0])
# axarr[1].set_title('x_paded')
# axarr[1].imshow(x_paded[0,:,:,0])
# plt.show()

def conv_single_step(a_slice_prev, W, b):
    s = np.multiply(a_slice_prev, W) + b
    Z = np.sum(s)
    return Z

# np.random.seed(1)
#
# a_slice_prev = np.random.randn(4,4,3)
# W = np.random.randn(4,4,3)
# b = np.random.randn(1,1,1)
#
# Z = conv_single_step(a_slice_prev, W, b)
#
# print("Z=" + str(Z))

def conv_forward(A_prev, W, b, hparameters):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    stride = hparameters["stride"]
    pad = hparameters["pad"]

    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1

    Z = np.zeros((m, n_H, n_W, n_C))

    A_prev_pad = zere_pad(A_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h*stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    a_slice_prev = a_prev_pad[vert_start: vert_end, horiz_start:horiz_end,:]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:,:,:,c], b[0,0,0,c])

    assert(Z.shape == (m, n_H, n_W, n_C))

    cache = (A_prev, W, b, hparameters)

    return (Z, cache)

# np.random.seed(1)
# A_prev = np.random.randn(10, 4, 4, 3)
# W = np.random.randn(2, 2, 3, 8)
# b = np.random.randn(1,1,1,8)
#
# hparameters = {"pad": 2, "stride": 1}
#
# Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
# print("np.mean(Z) = ", np.mean(Z))
# print("cache_conv[0][1][2][3] = ", cache_conv[0][1][2][3])

def pool_forward(A_prev, hparameters, mode="max"):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    f = hparameters["f"]
    stride = hparameters["stride"]

    n_H = int((n_H_prev - f) / stride) + 1
    n_W = int((n_W_prev - f) / stride) + 1
    n_C = n_C_prev

    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h*stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    a_slice_prev = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]

                    if mode == "max":
                        A[i, h, w, c] = np.max(a_slice_prev)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_slice_prev)

    assert(A.shape == (m, n_H, n_W, n_C))
    cache = (A_prev, hparameters)

    return A,cache

# np.random.seed(1)
# A_prev = np.random.randn(2, 4, 4, 3)
# hparameters = {"f":4, "stride":1}
#
# A, cache = pool_forward(A_prev, hparameters, mode="max")
# A, cache = pool_forward(A_prev, hparameters)
# print("mode=max")
# print("A=",A)
#
# A, cache = pool_forward(A_prev, hparameters, mode="average")
# print("mode=average")
# print("A=",A)

def conv_backward(dZ, cache):
    (A_prev, W, b, hparameters) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (m, n_H, n_W, n_C) = dZ.shape
    (f, f, n_C_prev, n_C) = W.shape

    pad = hparameters["pad"]
    stride = hparameters["stride"]

    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1,1,1,n_C))

    A_prev_pad = zere_pad(A_prev, pad)
    dA_prev_pad = zere_pad(dA_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h
                    vert_end = vert_start + f
                    horiz_start = w
                    horiz_end = horiz_start + f

                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end,:]
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end,:] += W[:,:,:,c]+dZ[i, h, w, c]
                    dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                    db[:,:,:,c] += dZ[i, h, w, c]

        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad,:]

    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

    return (dA_prev, dW, db)

# np.random.seed(1)
# A_prev = np.random.randn(10, 4, 4, 3)
# W = np.random.randn(2,2,3,8)
# b = np.random.randn(2,2,3,8)
# hparameters = {"pad":2, "stride":1}
#
# Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
#
# dA, dW, db = conv_backward(Z, cache_conv)

def create_mask_from_window(x):
    mask = x == np.max(x)
    return mask

# np.random.seed(1)
# x = np.random.randn(2,3)
# mask = create_mask_from_window(x)
# print("x=", str(x))
# print("mask=", str(mask))

def distribute_value(dz, shape):
    (n_H, n_W) = shape
    average = dz / (n_H * n_W)
    a = np.ones(shape)*average
    return a

# dz = 2
# shape = (2,2)
# a = distribute_value(dz, shape)
# print("a=" + str(a))

def pool_backward(dA, cache, mode="max"):
    (A_prev, hparameters) = cache
    f = hparameters["f"]
    stride = hparameters["stride"]

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (m, n_H, n_W, n_C) = dA.shape
    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h
                    vert_end = vert_start + f
                    horiz_start = w
                    horiz_end = horiz_end + f

                    if mode == "max":
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end,c]
                        mask = create_mask_from_window(a_prev_slice)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end,c] += np.multiply(mask, dA[i, h, w, c])
                    elif mode == "average":
                        da = dA[i,h,w,c]
                        shape = (f,f)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] + distribute_value(da, shape)

    assert(dA_prev.shape == A_prev.shape)
    return dA_prev


