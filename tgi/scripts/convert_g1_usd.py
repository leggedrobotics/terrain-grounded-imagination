"""Newton-compatible copy of the base env's G1 USD -> tgi/assets/g1_newton.usd (run once after installation).

Newton only creates collision shapes for gprims that carry PhysicsCollisionAPI, so the copy moves the API from the
instanceable collision Xforms onto the capsules.

    python scripts/convert_g1_usd.py
"""

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

from fpm.assets import G1_USD_PATH as src  # noqa: E402
from tgi.constants import G1_NEWTON_USD as dst  # noqa: E402

stage = Usd.Stage.Open(src)
flat = Usd.Stage.Open(stage.Flatten())
for p in flat.Traverse():
    if p.IsInstanceable():
        p.SetInstanceable(False)
n = 0
for p in list(flat.Traverse()):
    if p.HasAPI(UsdPhysics.CollisionAPI) and not p.IsA(UsdGeom.Gprim):
        for c in Usd.PrimRange(p):
            if c.IsA(UsdGeom.Gprim):
                UsdPhysics.CollisionAPI.Apply(c)
                n += 1
        p.RemoveAPI(UsdPhysics.CollisionAPI)
flat.GetRootLayer().Export(dst)
print(f"collision gprims: {n} -> {dst}")
app.close()
