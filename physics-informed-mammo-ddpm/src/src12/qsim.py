# qsim.py - exact state-vector simulator for the poster's Variational Quantum Circuit (VQC), batched, written once for
# both numpy (local tests) and torch (training with autograd). Poster: angle encoding -> L x [Rx Ry Rz per qubit
# (trainable) + entangling CNOT ring] -> measure <Z_i> of every qubit.
import numpy as np

class Backend:
    def __init__(self, lib):
        self.lib = lib; self.torch = lib.__name__ == "torch"
    def cplx(self, re, im):
        return self.lib.complex(re, im) if self.torch else re + 1j * im
    def stack(self, L, dim):
        return self.lib.stack(L, dim) if self.torch else np.stack(L, dim)
    def zeros_like_real(self, x):
        return self.lib.zeros_like(x)
    def index(self, perm):
        return self.lib.as_tensor(perm, dtype=self.lib.long) if self.torch else np.asarray(perm)

def rot(bk, theta, kind):
    """theta: (...,) real -> (...,2,2) complex rotation matrix RX / RY / RZ"""
    L = bk.lib; c, s = L.cos(theta / 2), L.sin(theta / 2); z = bk.zeros_like_real(theta)
    if kind == "x":
        rows = [[bk.cplx(c, z), bk.cplx(z, -s)], [bk.cplx(z, -s), bk.cplx(c, z)]]
    elif kind == "y":
        rows = [[bk.cplx(c, z), bk.cplx(-s, z)], [bk.cplx(s, z), bk.cplx(c, z)]]
    else:
        rows = [[bk.cplx(c, -s), bk.cplx(z, z)], [bk.cplx(z, z), bk.cplx(c, s)]]
    return bk.stack([bk.stack(r, -1) for r in rows], -2)

def apply_1q(bk, state, U, q, n):
    """state (B, 2^n); U (2,2) shared or (B,2,2) per sample; qubit q = most-significant-first (PennyLane wire order)"""
    B = state.shape[0]; s = state.reshape(B, 2 ** q, 2, 2 ** (n - q - 1))
    s = bk.lib.einsum("ij,akjb->akib", U, s) if U.ndim == 2 else bk.lib.einsum("aij,akjb->akib", U, s)
    return s.reshape(B, 2 ** n)

def cnot_ring_perm(n):
    """basis permutation of CNOT(0,1) CNOT(1,2) ... CNOT(n-1,0) applied in this order (n >= 3; n = 2 -> one CNOT)"""
    pairs = [(i, (i + 1) % n) for i in range(n)] if n > 2 else [(0, 1)]
    idx = np.arange(2 ** n)
    bit = lambda v, q: (v >> (n - 1 - q)) & 1
    for c, t in pairs:                       # new_state[i] = old_state[perm(i)], composed gate by gate
        src = np.array([i ^ (1 << (n - 1 - t)) if bit(i, c) else i for i in range(2 ** n)])
        idx = idx[src]
    return idx

def zsign(n):
    return np.array([[1 - 2 * ((i >> (n - 1 - q)) & 1) for q in range(n)] for i in range(2 ** n)], dtype=np.float32)

def angle_state(bk, angles):
    """product state RY(theta_q)|0> on every wire: (B,n) -> (B,2^n) (exact; identical to applying RY gates)"""
    B, n = angles.shape; L = bk.lib
    st = bk.cplx(L.ones((B, 1)) if bk.torch else np.ones((B, 1)), L.zeros((B, 1)) if bk.torch else np.zeros((B, 1)))
    for q in range(n):
        a = angles[:, q]; v = bk.stack([bk.cplx(L.cos(a / 2), L.zeros_like(a)), bk.cplx(L.sin(a / 2), L.zeros_like(a))], 1)
        st = (st.unsqueeze(-1) * v.unsqueeze(1)).reshape(B, -1) if bk.torch else (st[:, :, None] * v[:, None, :]).reshape(B, -1)
    return st

def vqc_state(bk, angles, W, perm, init=None, reupload=False, entangle=True):
    """angles (B,n): encoding angles; W (L,n,3) trainable (Rx,Ry,Rz) -> final state (B,2^n).
    init      : optional prepared state (B,2^n) (amplitude / basis encoding); default = angle encoding RY(angles)|0>
    reupload  : re-apply the encoding RY(angles) before every layer after the first (data re-uploading)
    entangle  : False removes the CNOT ring (no-entanglement ablation)"""
    n = angles.shape[1]
    st = angle_state(bk, angles) if init is None else init
    for l in range(W.shape[0]):
        if reupload and l > 0:
            Ry = rot(bk, angles, "y")                                                                # (B,n,2,2)
            for q in range(n):
                st = apply_1q(bk, st, Ry[:, q], q, n)
        U = rot(bk, W[l, :, 2], "z") @ rot(bk, W[l, :, 1], "y") @ rot(bk, W[l, :, 0], "x")       # (n,2,2): Rx first, then Ry, Rz
        for q in range(n):
            st = apply_1q(bk, st, U[q], q, n)
        if entangle:
            st = st[:, perm]
    return st

def feature_map_state(bk, angles, reps=2):
    """fixed (non-trainable) entangling feature map for the quantum kernel: reps x [RY(x) on all wires, CNOT ring,
    RZ(x) on all wires] starting from |0...0> (in the spirit of Havlicek et al. 2019)"""
    B, n = angles.shape; perm = cnot_ring_perm(n)
    st = angle_state(bk, angles)
    for r in range(reps):
        if r > 0:
            Ry = rot(bk, angles, "y")
            for q in range(n):
                st = apply_1q(bk, st, Ry[:, q], q, n)
        st = st[:, perm]
        Rz = rot(bk, angles, "z")
        for q in range(n):
            st = apply_1q(bk, st, Rz[:, q], q, n)
    return st

def expz(bk, st, zs):
    p = st.real ** 2 + st.imag ** 2
    return p @ zs

def bloch(bk, st, q, n):
    """Bloch vector (<X>,<Y>,<Z>) of qubit q (reduced state) -> (B,3)"""
    B = st.shape[0]; s = st.reshape(B, 2 ** q, 2, 2 ** (n - q - 1))
    a0, a1 = s[:, :, 0, :], s[:, :, 1, :]
    r01 = (a0 * a1.conj()).sum((1, 2)) if bk.torch else (a0 * a1.conj()).sum((1, 2))   # rho_01 = sum psi_0 psi_1*
    pz = (abs(a0) ** 2).sum((1, 2)) - (abs(a1) ** 2).sum((1, 2))
    return bk.stack([2 * r01.real, -2 * r01.imag, pz], 1)

def n_params(n, L):
    return 3 * n * L
