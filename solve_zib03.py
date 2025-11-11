import time
import os
os.environ["JAX_LOG_COMPILES"] = "1"
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import gurobipy as gp
import jax
jax.config.update("jax_enable_x64", True)
from mpax.mp_io import create_qp_from_gurobi
from mpax.r2hpdhg import r2HPDHG
import time

log = []
path ="F:/LP/MPAX-main/zib03_pre.mps"
fname = "zib03.mps"

# ===== Warm-up để kích hoạt JIT (nếu có) =====
gurobi_model = gp.read(path)
gurobi_model.setParam("Seed", 36)
gurobi_model.setParam("Threads", 1)
gurobi_model.setParam("OutputFlag", 0)
gurobi_model = gurobi_model.relax()
lp = create_qp_from_gurobi(gurobi_model)


solver_warmup = r2HPDHG(iteration_limit=100,eps_abs=1e-6, eps_rel=1e-6)
_ = solver_warmup.optimize(lp)

print(f"🚀 Warm-up xong {fname}")

solver = r2HPDHG(eps_abs=1e-6, eps_rel=1e-6,verbose= True,l2_norm_rescaling=True)
start = time.time()
result = solver.optimize(lp)
elapsed = time.time() - start

status = int(result.termination_status)
obj_val = float(getattr(result, "primal_objective", float("nan")))


print(f"✅ {fname} | Obj: {obj_val:.6f} | Time: {elapsed:.2f}s | Status: {status}")
log.append((fname, obj_val, elapsed))

del solver

# ===== Ghi log ra file =====
with open("solve_log_r2hpdhg.txt", "a") as f:
    for fname, obj, t in log:
        f.write(f"{fname}\tObj: {obj}\tTime: {t}\n")