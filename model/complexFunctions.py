#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: spopoff
"""

import torch
from torch.nn.functional import (
    avg_pool2d,
    dropout,
    dropout2d,
    interpolate,
    max_pool2d,
    relu,
    gelu,
    sigmoid,
    tanh,
)


from torch.nn.functional import max_pool2d, avg_pool2d, dropout, dropout2d, interpolate
from torch import tanh, relu, sigmoid

def complex2real(inp, dim):
    return torch.cat((inp.real,inp.imag),dim)

def complex_matmul(A, B):
    """
    Performs the matrix product between two complex matrices
    """

    outp_real = torch.matmul(A.real, B.real) - torch.matmul(A.imag, B.imag)
    outp_imag = torch.matmul(A.real, B.imag) + torch.matmul(A.imag, B.real)

    return outp_real.type(torch.complex64) + 1j * outp_imag.type(torch.complex64)


def complex_avg_pool2d(inp, *args, **kwargs):
    """
    Perform complex average pooling.
    """
    absolute_value_real = avg_pool2d(inp.real, *args, **kwargs)
    absolute_value_imag = avg_pool2d(inp.imag, *args, **kwargs)

    return absolute_value_real.type(torch.complex64) + 1j * absolute_value_imag.type(
        torch.complex64
    )


def complex_normalize(inp):
    """
    Perform complex normalization
    """
    real_value, imag_value = inp.real, inp.imag
    real_norm = (real_value - real_value.mean()) / real_value.std()
    imag_norm = (imag_value - imag_value.mean()) / imag_value.std()
    return real_norm.type(torch.complex64) + 1j * imag_norm.type(torch.complex64)


def complex_relu(inp):
    return relu(inp.real).type(torch.complex64) + 1j * relu(inp.imag).type(
        torch.complex64
    )

def complex_gelu(inp):
    return gelu(inp.real).type(torch.complex64) + 1j * gelu(inp.imag).type(
        torch.complex64
    )

def complex_sigmoid(inp):
    return sigmoid(inp.real).type(torch.complex64) + 1j * sigmoid(inp.imag).type(
        torch.complex64
    )


def complex_tanh(inp):
    return tanh(inp.real).type(torch.complex64) + 1j * tanh(inp.imag).type(
        torch.complex64
    )


def complex_opposite(inp):
    return -inp.real.type(torch.complex64) + 1j * (-inp.imag.type(torch.complex64))


def complex_stack(inp, dim):
    inp_real = [x.real for x in inp]
    inp_imag = [x.imag for x in inp]
    return torch.stack(inp_real, dim).type(torch.complex64) + 1j * torch.stack(
        inp_imag, dim
    ).type(torch.complex64)


def _retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=-2)
    output = flattened_tensor.gather(
        dim=-1, index=indices.flatten(start_dim=-2)
    ).view_as(indices)
    return output


def complex_upsample(
    inp,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    recompute_scale_factor=None,
):
    """
    Performs upsampling by separately interpolating the real and imaginary part and recombining
    """
    outp_real = interpolate(
        inp.real,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
    )
    outp_imag = interpolate(
        inp.imag,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
    )

    return outp_real.type(torch.complex64) + 1j * outp_imag.type(torch.complex64)


def complex_upsample2(
    inp,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    recompute_scale_factor=None,
):
    """
    Performs upsampling by separately interpolating the amplitude and phase part and recombining
    """
    outp_abs = interpolate(
        inp.abs(),
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
    )
    angle = torch.atan2(inp.imag, inp.real)
    outp_angle = interpolate(
        angle,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
    )

    return outp_abs * (
        torch.cos(outp_angle).type(torch.complex64)
        + 1j * torch.sin(outp_angle).type(torch.complex64)
    )


def complex_max_pool2d(
    inp,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
):
    """
    Perform complex max pooling by selecting on the absolute value on the complex values.
    """
    absolute_value, indices = max_pool2d(
        inp.abs(),
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=True,
    )
    # performs the selection on the absolute values
    # absolute_value = absolute_value.type(torch.complex64)
    # # retrieve the corresponding phase value using the indices
    # # unfortunately, the derivative for 'angle' is not implemented
    # angle = torch.atan2(inp.imag, inp.real)
    # # get only the phase values selected by max pool
    # angle = _retrieve_elements_from_indices(angle, indices)

    result = _retrieve_elements_from_indices(inp, indices)
    return result
        #absolute_value * (torch.cos(angle).type(torch.complex64)+ 1j * torch.sin(angle).type(torch.complex64))


def complex_dropout(inp, p=0.5, training=True):
    # need to have the same dropout mask for real and imaginary part,
    # this not a clean solution!
    mask = torch.ones(*inp.shape, dtype=torch.float32, device=inp.device)
    mask = dropout(mask, p, training) * 1 / (1 - p)
    mask.type(inp.dtype)
    return mask * inp


def complex_dropout2d(inp, p=0.5, training=True):
    # need to have the same dropout mask for real and imaginary part,
    # this not a clean solution!
    mask = torch.ones(*inp.shape, dtype=torch.float32, device=inp.device)
    mask = dropout2d(mask, p, training) * 1 / (1 - p)
    mask.type(inp.dtype)
    return mask * inp


## new
def c_norm(
    input_centred,
    Vrr,
    Vii,
    Vri,
    beta,
    gamma_rr,
    gamma_ri,
    gamma_ii,
    scale=True,
    center=True,
    layernorm=False,
    dim=-1,
):

    """This function is used to apply the complex normalization
    as introduced by "Deep Complex Networks", Trabelsi C. et al.

    Arguments
    ---------
    input_centred : torch.Tensor
        It is the tensor to be normalized. The features
        dimension is divided by 2 with the first half
        corresponding to the real-parts and the second half
        to the imaginary parts.
    Vrr : torch.Tensor
        It is a tensor that contains the covariance between real-parts.
    Vii : torch.Tensor
        It is a tensor that contains the covariance between imaginary-parts.
    Vri : torch.Tensor
        It is a tensor that contains the covariance between real-parts and
        imaginary-parts.
    beta : torch.Tensor
        It is a tensor corresponding to the beta parameter on the real-valued
        batch-normalization, but in the complex-valued space.
    gamma_rr : torch.Tensor
        It is a tensor that contains the gamma between real-parts.
    gamma_ii : torch.Tensor
        It is a tensor that contains the gamma between imaginary-parts.
    gamma_ri : torch.Tensor
        It is a tensor that contains the gamma between real-parts and
        imaginary-parts.
    scale : bool, optional
        It defines if scaling should be used or not. It is
        equivalent to the real-valued batchnormalization
        scaling (default True).
    center : bool, optional,
        It defines if centering should be used or not. It is
        equivalent to the real-valued batchnormalization centering
        (default True).
    layernorm : bool, optional
        It defines is c_standardization is called from a layernorm or a
        batchnorm layer (default False).
    dim : int, optional
        It defines the axis that should be considered as the complex-valued
        axis (divided by 2 to get r and i) (default -1).
    """

    ndim = input_centred.dim()
    input_dim = input_centred.size(dim) // 2
    if scale:
        gamma_broadcast_shape = [1] * ndim
        gamma_broadcast_shape[dim] = input_dim
    if center:
        broadcast_beta_shape = [1] * ndim
        broadcast_beta_shape[dim] = input_dim * 2

    if scale:
        standardized_output = c_standardization(
            input_centred, Vrr, Vii, Vri, layernorm, dim=dim
        )

        # Now we perform the scaling and Shifting of the normalized x using
        # the scaling parameter
        #           [  gamma_rr gamma_ri  ]
        #   Gamma = [  gamma_ri gamma_ii  ]
        # and the shifting parameter
        #    Beta = [beta_real beta_imag].T
        # where:
        # x_real_BN = gamma_rr * x_real_normed +
        #             gamma_ri * x_imag_normed + beta_real
        # x_imag_BN = gamma_ri * x_real_normed +
        #             gamma_ii * x_imag_normed + beta_imag

        broadcast_gamma_rr = gamma_rr.view(gamma_broadcast_shape)
        broadcast_gamma_ri = gamma_ri.view(gamma_broadcast_shape)
        broadcast_gamma_ii = gamma_ii.view(gamma_broadcast_shape)

        cat_gamma_4_real = torch.cat(
            [broadcast_gamma_rr, broadcast_gamma_ii], dim=dim
        )
        cat_gamma_4_imag = torch.cat(
            [broadcast_gamma_ri, broadcast_gamma_ri], dim=dim
        )
        if dim == 0:
            centred_real = standardized_output[:input_dim]
            centred_imag = standardized_output[input_dim:]
        elif dim == 1 or (dim == -1 and ndim == 2):
            centred_real = standardized_output[:, :input_dim]
            centred_imag = standardized_output[:, input_dim:]
        elif dim == -1 and ndim == 3:
            centred_real = standardized_output[:, :, :input_dim]
            centred_imag = standardized_output[:, :, input_dim:]
        elif dim == -1 and ndim == 4:
            centred_real = standardized_output[:, :, :, :input_dim]
            centred_imag = standardized_output[:, :, :, input_dim:]
        else:
            centred_real = standardized_output[:, :, :, :, :input_dim]
            centred_imag = standardized_output[:, :, :, :, input_dim:]

        rolled_standardized_output = torch.cat(
            [centred_imag, centred_real], dim=dim
        )
        if center:
            broadcast_beta = beta.view(broadcast_beta_shape)
            a = cat_gamma_4_real * standardized_output
            b = cat_gamma_4_imag * rolled_standardized_output
            return a + b + broadcast_beta
        else:
            return (
                cat_gamma_4_real * standardized_output
                + cat_gamma_4_imag * rolled_standardized_output
            )
    else:
        if center:
            broadcast_beta = beta.view(broadcast_beta_shape)
            return input_centred + broadcast_beta
        else:
            return input_centred


def c_standardization(input_centred, Vrr, Vii, Vri, layernorm=False, dim=-1):
    """This function is used to standardize a centred tensor of
    complex numbers (mean of the set must be 0).

    Arguments
    ---------
    input_centred : torch.Tensor
        It is the tensor to be normalized. The features
        dimension is divided by 2 with the first half
        corresponding to the real-parts and the second half
        to the imaginary parts.
    Vrr : torch.Tensor
        It is a tensor that contains the covariance between real-parts.
    Vii : torch.Tensor
        It is a tensor that contains the covariance between imaginary-parts.
    Vri : torch.Tensor
        It is a tensor that contains the covariance between real-parts and
        imaginary-parts.
    layernorm : bool, optional
        It defines is c_standardization is called from a layernorm or a
        batchnorm layer (default False).
    dim : int, optional
        It defines the axis that should be considered as the complex-valued
        axis (divided by 2 to get r and i) (default -1).
    """
    ndim = input_centred.dim()
    input_dim = input_centred.size(dim) // 2
    variances_broadcast = [1] * ndim
    variances_broadcast[dim] = input_dim

    if layernorm:
        variances_broadcast[0] = input_centred.size(0)

    # We require the covariance matrix's inverse square root. That requires
    # square rooting, followed by inversion (During the computation of square
    # root we compute the determinant we'll need for inversion as well).

    # tau = Vrr + Vii = Trace. Guaranteed >=0 because Positive-definite matrix
    tau = Vrr + Vii

    # delta = (Vrr * Vii) - (Vri ** 2) = Determinant
    delta = (Vrr * Vii) - (Vri ** 2)

    s = delta.sqrt()
    t = (tau + 2 * s).sqrt()

    # The square root matrix could now be explicitly formed as
    #       [ Vrr+s Vri   ]
    # (1/t) [ Vir   Vii+s ]
    # https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
    # but we don't need to do this immediately since we can also simultaneously
    # invert. We can do this because we've already computed the determinant of
    # the square root matrix, and can thus invert it using the analytical
    # solution for 2x2 matrices
    #      [ A B ]             [  D  -B ]
    # inv( [ C D ] ) = (1/det) [ -C   A ]
    # http://mathworld.wolfram.com/MatrixInverse.html
    # Thus giving us
    #           [  Vii+s  -Vri   ]
    # (1/s)(1/t)[ -Vir     Vrr+s ]
    # So we proceed as follows:

    inverse_st = 1.0 / (s * t)
    Wrr = (Vii + s) * inverse_st
    Wii = (Vrr + s) * inverse_st
    Wri = -Vri * inverse_st

    # And we have computed the inverse square root matrix W = sqrt(V)!
    # Normalization. We multiply, x_normalized = W.x.

    # The returned result will be a complex standardized input
    # where the real and imaginary parts are obtained as follows:
    # x_real_normed = Wrr * x_real_centred + Wri * x_imag_centred
    # x_imag_normed = Wri * x_real_centred + Wii * x_imag_centred

    broadcast_Wrr = Wrr.view(variances_broadcast)
    broadcast_Wri = Wri.view(variances_broadcast)
    broadcast_Wii = Wii.view(variances_broadcast)

    cat_W_4_real = torch.cat([broadcast_Wrr, broadcast_Wii], dim=dim)
    cat_W_4_imag = torch.cat([broadcast_Wri, broadcast_Wri], dim=dim)

    if dim == 0:
        centred_real = input_centred[:input_dim]
        centred_imag = input_centred[input_dim:]
    elif dim == 1 or (dim == -1 and ndim == 2):
        centred_real = input_centred[:, :input_dim]
        centred_imag = input_centred[:, input_dim:]
    elif dim == -1 and ndim == 3:
        centred_real = input_centred[:, :, :input_dim]
        centred_imag = input_centred[:, :, input_dim:]
    elif dim == -1 and ndim == 4:
        centred_real = input_centred[:, :, :, :input_dim]
        centred_imag = input_centred[:, :, :, input_dim:]
    else:
        centred_real = input_centred[:, :, :, :, :input_dim]
        centred_imag = input_centred[:, :, :, :, input_dim:]

    rolled_input = torch.cat([centred_imag, centred_real], dim=dim)

    output = cat_W_4_real * input_centred + cat_W_4_imag * rolled_input

    #   Wrr * x_real_centered | Wii * x_imag_centered
    # + Wri * x_imag_centered | Wri * x_real_centered
    # -----------------------------------------------
    # = output

    return output



def multi_mean(input, axes, keepdim=False):
    """
    Performs `torch.mean` over multiple dimensions of `input`.
    """
    axes = sorted(axes)
    m = input
    for axis in reversed(axes):
        m = m.mean(axis, keepdim)
    return m