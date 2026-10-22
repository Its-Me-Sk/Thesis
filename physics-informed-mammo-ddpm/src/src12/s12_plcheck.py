# s12_plcheck.py - cross-check of qsim.py (numpy backend) against PennyLane default.qubit (run in its own process).
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qsim as Q
out = sys.argv[1]
import pennylane as qml
res = []
for n, L in ((4, 3), (8, 4)):
    r = np.random.default_rng(n); ang = r.uniform(-3, 3, (5, n)); W = r.uniform(-3, 3, (L, n, 3))
    dv = qml.device("default.qubit", wires=n)
    @qml.qnode(dv)
    def circ(x):
        for q in range(n): qml.RY(x[q], wires=q)
        for l in range(L):
            for q in range(n):
                qml.RX(W[l, q, 0], wires=q); qml.RY(W[l, q, 1], wires=q); qml.RZ(W[l, q, 2], wires=q)
            for q in range(n): qml.CNOT(wires=[q, (q + 1) % n])
        return [qml.expval(qml.PauliZ(q)) for q in range(n)]
    ref = np.array([np.array(circ(x), dtype=float) for x in ang])
    mine = Q.expz(None, Q.vqc_state(Q.Backend(np), ang, W, Q.cnot_ring_perm(n)), Q.zsign(n))
    res.append(float(np.abs(ref - mine).max()))
err = max(res); print(f"PennyLane {qml.__version__} cross-check (4 and 8 qubits): max |<Z> difference| = {err:.2e}", flush=True)
json.dump({"pennylane": qml.__version__, "max_abs_diff": err}, open(os.path.join(out, "pennylane_check.json"), "w"))
sys.exit(0 if err < 1e-5 else 3)
