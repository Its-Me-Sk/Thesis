import sys, numpy as np; import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "src12"))
import qsim as Q
bk = Q.Backend(np); r = np.random.default_rng(0)
I2 = np.eye(2); X = np.array([[0,1],[1,0]]); Y = np.array([[0,-1j],[1j,0]]); Z = np.diag([1,-1])
def RX(t): return np.cos(t/2)*I2 - 1j*np.sin(t/2)*X
def RY(t): return np.cos(t/2)*I2 - 1j*np.sin(t/2)*Y
def RZ(t): return np.cos(t/2)*I2 - 1j*np.sin(t/2)*Z
def kron(L):
    M = np.array([[1.0]])
    for m in L: M = np.kron(M, m)
    return M
def on(U, q, n): return kron([U if i == q else I2 for i in range(n)])
def cnot(c, t, n):
    P0 = np.diag([1,0]); P1 = np.diag([0,1])
    return kron([P0 if i == c else I2 for i in range(n)]) + kron([P1 if i == c else (X if i == t else I2) for i in range(n)])
for n, L in ((2, 1), (3, 2), (4, 3), (5, 2)):
    B = 3; ang = r.uniform(-3, 3, (B, n)); W = r.uniform(-3, 3, (L, n, 3))
    st = Q.vqc_state(bk, ang, W, Q.cnot_ring_perm(n)); ez = Q.expz(bk, st, Q.zsign(n))
    pairs = [(i, (i + 1) % n) for i in range(n)] if n > 2 else [(0, 1)]
    for b in range(B):
        psi = np.zeros(2 ** n, complex); psi[0] = 1
        for q in range(n): psi = on(RY(ang[b, q]), q, n) @ psi
        for l in range(L):
            for q in range(n): psi = on(RZ(W[l,q,2]) @ RY(W[l,q,1]) @ RX(W[l,q,0]), q, n) @ psi
            for c, t in pairs: psi = cnot(c, t, n) @ psi
        assert np.allclose(psi, st[b], atol=1e-10), (n, L, b)
        ref = [np.real(psi.conj() @ on(Z, q, n) @ psi) for q in range(n)]
        assert np.allclose(ref, ez[b])
        for q in range(n):
            bl = Q.bloch(bk, st[b:b+1], q, n)[0]
            refb = [np.real(psi.conj() @ on(P, q, n) @ psi) for P in (X, Y, Z)]
            assert np.allclose(bl, refb, atol=1e-10), (bl, refb)
print("qsim exact vs dense-matrix reference: OK (states, <Z>, Bloch vectors)")
