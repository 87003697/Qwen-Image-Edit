# 可调整的相机与分辨率参数
W=1024
H=1024
RADIUS=2.0
FOVY=50.0
ELEV=0.0
AZI=0.0

python scripts/render_mesh.py \
    --mesh @dataset/meshes/01.glb \
    --out outputs/render_mesh_normal.png \
    --out_mode normal \
    --W ${W} \
    --H ${H} \
    --radius ${RADIUS} \
    --fovy ${FOVY} \
    --elevation ${ELEV} \
    --azimuth ${AZI}