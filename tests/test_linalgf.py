# -*- coding: utf-8 -*-
"""
    tests.linalgf
    ~~~~~~~~~~~~~

    Test the linalgf module.
"""
import numpy as np
from scipy.linalg import block_diag
import pytest
import limetr.linalgf as linalgf


@pytest.mark.parametrize(("n", "k"),
                         [(6, 3), (3, 6)])
def test_block_lsvd(n, k):
    z = np.random.randn(n, k)
    tr_u, tr_s, tr_vt = np.linalg.svd(z, full_matrices=False)
    my_u = np.zeros(tr_u.size)
    my_s = np.zeros(tr_s.size)
    linalgf.izmat.block_lsvd(z, my_u, my_s)

    assert np.linalg.norm(my_u.reshape(min(n, k), n).T - tr_u) < 1e-8
    assert np.linalg.norm(my_s - tr_s) < 1e-8


@pytest.mark.parametrize(("n", "k"),
                         [(6, 3), (3, 6)])
def test_block_izmv(n, k):
    m = min(n, k)

    z = np.random.randn(n, k)
    x = np.random.randn(n)

    my_u = np.zeros(n*m)
    my_s = np.zeros(m)
    linalgf.izmat.block_lsvd(z, my_u, my_s)

    tr_y = x + z.dot(z.T.dot(x))
    my_y = np.zeros(n)
    linalgf.izmat.block_izmv(my_u, my_s**2, x, my_y)

    assert np.linalg.norm(my_y - tr_y) < 1e-8

@pytest.mark.parametrize(("n", "k"),
                         [(6, 3), (3, 6)])
def test_block_izmm(n, k):
    m = min(n, k)

    z = np.random.randn(n, k)
    x = np.random.randn(n, 5)

    my_u = np.zeros(n*m)
    my_s = np.zeros(m)
    linalgf.izmat.block_lsvd(z, my_u, my_s)

    tr_y = x + z.dot(z.T.dot(x))
    my_y = np.zeros((n, 5), order='F')
    linalgf.izmat.block_izmm(my_u, my_s**2, x, my_y)

    assert np.linalg.norm(my_y - tr_y) < 1e-8


@pytest.mark.parametrize(("n", "k"),
                         [(6, 3), (3, 6)])
def test_block_izdiag(n, k):
    m = min(n, k)

    z = np.random.randn(n, k)
    my_u = np.zeros(n*m)
    my_s = np.zeros(m)
    linalgf.izmat.block_lsvd(z, my_u, my_s)

    tr_d = np.diag(np.eye(n) + z.dot(z.T))
    my_d = np.zeros(n)
    linalgf.izmat.block_izdiag(my_u, my_s**2, my_d)

    assert np.linalg.norm(my_d - tr_d) < 1e-8


@pytest.mark.parametrize(("n", "k"),
                         [([5, 2, 4], 3)])
def test_lzvd(n, k):
    z_full = np.random.randn(np.sum(n), k)
    z_list = np.split(z_full, np.cumsum(n)[:-1])

    tr_u_list = []
    tr_s_list = []
    for i in range(len(n)):
        u, s, vt = np.linalg.svd(z_list[i], full_matrices=False)
        tr_u_list.append(u)
        tr_s_list.append(s)
    tr_u = np.hstack([u.reshape(u.size, order='F') for u in tr_u_list])
    tr_s = np.hstack(tr_s_list)

    my_u = np.zeros(tr_u.size)
    my_s = np.zeros(tr_s.size)

    nz = [z.shape[0] for z in z_list]
    nu = [u.size for u in tr_u_list]
    ns = [s.size for s in tr_s_list]

    linalgf.izmat.lsvd(nz, nu, ns, z_full, my_u, my_s)

    assert np.linalg.norm(my_u - tr_u) < 1e-8
    assert np.linalg.norm(my_s - tr_s) < 1e-8


@pytest.mark.parametrize(("n", "k"),
                         [([5, 2, 4], 3)])
def test_izmv(n, k):
    z_full = np.random.randn(np.sum(n), k)
    z_list = np.split(z_full, np.cumsum(n)[:-1])
    x_list = [np.random.randn(n[i]) for i in range(len(n))]
    x_full = np.hstack(x_list)

    tr_y = np.hstack([
        x_list[i] + z_list[i].dot(z_list[i].T.dot(x_list[i]))
        for i in range(len(n))
    ])

    ns = np.minimum(n, k)
    nu = ns * n
    nx = n
    nz = n

    my_u = np.zeros(nu.sum())
    my_s = np.zeros(ns.sum())

    linalgf.izmat.lsvd(nz, nu, ns, z_full, my_u, my_s)
    my_y = linalgf.izmat.izmv(nu, ns, nx, my_u, my_s**2, x_full)

    assert np.linalg.norm(my_y - tr_y) < 1e-8


@pytest.mark.parametrize(("n", "k"),
                         [([5, 2, 4], 3)])
def test_izmm(n, k):
    z_full = np.random.randn(np.sum(n), k)
    z_list = np.split(z_full, np.cumsum(n)[:-1])
    x_list = [np.random.randn(n[i], 5) for i in range(len(n))]
    x_full = np.vstack(x_list)

    tr_y = np.vstack([
        x_list[i] + z_list[i].dot(z_list[i].T.dot(x_list[i]))
        for i in range(len(n))
    ])

    ns = np.minimum(n, k)
    nu = ns*n
    nx = n
    nz = n

    my_u = np.zeros(nu.sum())
    my_s = np.zeros(ns.sum())

    linalgf.izmat.lsvd(nz, nu, ns, z_full, my_u, my_s)
    my_y = linalgf.izmat.izmm(nu, ns, nx, my_u, my_s**2, x_full)

    assert np.linalg.norm(my_y - tr_y) < 1e-8


@pytest.mark.parametrize(("n", "k"),
                         [([5, 2, 4], 3)])
def test_izdiag(n, k):
    z_full = np.random.randn(np.sum(n), k)
    z_list = np.split(z_full, np.cumsum(n)[:-1])

    tr_d = np.hstack([
        np.diag(np.eye(n[i]) + z_list[i].dot(z_list[i].T))
        for i in range(len(n))
    ])

    ns = np.minimum(n, k)
    nu = ns*n
    nx = n
    nz = n

    my_u = np.zeros(nu.sum())
    my_s = np.zeros(ns.sum())

    linalgf.izmat.lsvd(nz, nu, ns, z_full, my_u, my_s)
    my_d = linalgf.izmat.izdiag(sum(n), nu, ns, nx, my_u, my_s**2)

    assert np.linalg.norm(my_d - tr_d) < 1e-8


@pytest.mark.parametrize(("n", "k"),
                         [([5, 2, 4], 3)])
def test_izeig(n, k):
    z_full = np.random.randn(np.sum(n), k)
    z_list = np.split(z_full, np.cumsum(n)[:-1])

    tr_d = np.hstack([
        np.diag(np.eye(n[i]) + z_list[i].dot(z_list[i].T))
        for i in range(len(n))
    ])

    ns = np.minimum(n, k)
    nu = ns * n
    nz = n

    my_u = np.zeros(nu.sum())
    my_s = np.zeros(ns.sum())

    linalgf.izmat.lsvd(nz, nu, ns, z_full, my_u, my_s)
    my_eig_val = linalgf.izmat.izeig(sum(n), n, ns, my_s**2)
    tr_eig_val, tr_eig_vec = np.linalg.eig(block_diag(*[
        np.eye(n[i]) + z_list[i].dot(z_list[i].T)
        for i in range(len(n))
    ]))

    assert np.linalg.norm(my_eig_val - tr_eig_val)
