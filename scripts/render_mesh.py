import argparse
import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import nvdiffrast.torch as dr
import envlight
from kiui.mesh import Mesh
from kiui.cam import OrbitCamera
from kiui.op import safe_normalize

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", default="@dataset/meshes/01.glb")
    parser.add_argument("--hdr", default="_reference_codes/kiuikit/kiui/assets/lights/mud_road_puresky_1k.hdr")
    parser.add_argument("--fg_lut", default="_reference_codes/kiuikit/kiui/assets/lights/bsdf_256_256.bin")
    parser.add_argument("--out", default="outputs/render_mesh.png")
    parser.add_argument("--out_mode", default="normal", choices=["normal", "rgb"])
    parser.add_argument("--W", type=int, default=1024)
    parser.add_argument("--H", type=int, default=1024)
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--fovy", type=float, default=50.0)
    parser.add_argument("--elevation", type=float, default=0.0)
    parser.add_argument("--azimuth", type=float, default=0.0)
    parser.add_argument("--env_scale", type=float, default=2.0)
    parser.add_argument("--front_dir", type=str, default="+z")
    return parser.parse_args()

def resolve_paths(args):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return {
        "mesh": os.path.join(repo_root, args.mesh),
        "hdr": os.path.join(repo_root, args.hdr),
        "fg_lut": os.path.join(repo_root, args.fg_lut),
        "out": os.path.join(repo_root, args.out),
    }

def setup_camera(args, W, H):
    cam = OrbitCamera(W, H, r=args.radius, fovy=args.fovy)  # 相机轨道半径/视场
    cam.from_angle(elevation=args.elevation, azimuth=args.azimuth)
    return cam

def load_resources(paths, device, front_dir, env_scale):
    mesh = Mesh.load(paths["mesh"], device=device, front_dir=front_dir)  # 读取网格
    light = envlight.EnvLight(paths["hdr"], scale=env_scale, device=device)
    FG_LUT = torch.from_numpy(np.fromfile(paths["fg_lut"], dtype=np.float32).reshape(1, 256, 256, 2)).to(device)  # shape: [1, 256, 256, 2]
    return mesh, light, FG_LUT


def render_normal(mesh, cam, device, H, W, bg_color):
    glctx = dr.RasterizeCudaContext()

    pose = torch.from_numpy(cam.pose.astype(np.float32)).to(device)  # shape: [4, 4]
    proj = torch.from_numpy(cam.perspective.astype(np.float32)).to(device)  # shape: [4, 4]

    v_homo = F.pad(mesh.v, (0, 1), value=1.0)  # shape: [V, 4]
    v_cam = torch.matmul(v_homo, torch.inverse(pose).T)  # shape: [V, 4]
    v_cam = v_cam.unsqueeze(0)  # shape: [1, V, 4]
    v_clip = torch.matmul(v_cam, proj.T)  # shape: [1, V, 4]

    rast, rast_db = dr.rasterize(glctx, v_clip, mesh.f, (H, W))  # shape: [1, H, W, 4]

    alpha = (rast[..., 3:] > 0).float()  # shape: [1, H, W, 1]
    alpha = dr.antialias(alpha, rast, v_clip, mesh.f).clamp(0, 1)  # shape: [1, H, W, 1]

    normal, _ = dr.interpolate(mesh.vn.unsqueeze(0).contiguous(), rast, mesh.fn)  # shape: [1, H, W, 3]
    normal = safe_normalize(normal)  # shape: [1, H, W, 3]

    R_wc = torch.inverse(pose)[:3, :3]  # shape: [3, 3]
    normal_view = torch.einsum("bhwc,cd->bhwd", normal, R_wc)  # shape: [1, H, W, 3]
    normal_view = safe_normalize(normal_view)  # shape: [1, H, W, 3]

    # OpenCV/OpenGL 坐标差异：翻转 X 轴与参考实现保持一致
    opencv_to_opengl = torch.eye(3, device=device, dtype=normal_view.dtype)  # shape: [3, 3]
    opencv_to_opengl[0, 0] = -1
    normal_view = torch.einsum("bhwc,cd->bhwd", normal_view, opencv_to_opengl)  # shape: [1, H, W, 3]

    normal_img = (normal_view * 0.5 + 0.5).clamp(0, 1)  # shape: [1, H, W, 3]
    bg = bg_color.view(1, 1, 1, 3)  # shape: [1, 1, 1, 3]
    normal_img = torch.where(alpha > 0, normal_img, bg)  # shape: [1, H, W, 3]

    normal_buffer = normal_img[0].detach().cpu().numpy()  # shape: [H, W, 3]
    normal_buffer = np.clip(normal_buffer, 0, 1)

    return normal_buffer  # RGB float [H, W, 3] in [0,1]

def render_pbr(mesh, light, FG_LUT, cam, device, H, W, bg_color):
    glctx = dr.RasterizeCudaContext()

    pose = torch.from_numpy(cam.pose.astype(np.float32)).to(device)  # shape: [4, 4]
    proj = torch.from_numpy(cam.perspective.astype(np.float32)).to(device)  # shape: [4, 4]

    v_homo = F.pad(mesh.v, (0, 1), value=1.0)  # shape: [V, 4]
    v_cam = torch.matmul(v_homo, torch.inverse(pose).T)  # shape: [V, 4]
    v_cam = v_cam.unsqueeze(0)  # shape: [1, V, 4]
    v_clip = torch.matmul(v_cam, proj.T)  # shape: [1, V, 4]

    rast, rast_db = dr.rasterize(glctx, v_clip, mesh.f, (H, W))  # shape: [1, H, W, 4]

    alpha = (rast[..., 3:] > 0).float()  # shape: [1, H, W, 1]
    alpha = dr.antialias(alpha, rast, v_clip, mesh.f).clamp(0, 1)  # shape: [1, H, W, 1]

    if mesh.vt is not None and mesh.ft is not None and mesh.albedo is not None:
        texc, _ = dr.interpolate(mesh.vt.unsqueeze(0).contiguous(), rast, mesh.ft)  # shape: [1, H, W, 2]
        albedo = dr.texture(mesh.albedo.unsqueeze(0), texc, filter_mode="linear")  # shape: [1, H, W, 3]
    else:
        albedo = torch.ones(1, H, W, 3, device=device)  # shape: [1, H, W, 3]
        texc = None  # 未使用

    normal, _ = dr.interpolate(mesh.vn.unsqueeze(0).contiguous(), rast, mesh.fn)  # shape: [1, H, W, 3]
    normal = safe_normalize(normal)  # shape: [1, H, W, 3]
    R_wc = torch.inverse(pose)[:3, :3]  # shape: [3, 3]
    normal_view = torch.einsum("bhwc,cd->bhwd", normal, R_wc)  # shape: [1, H, W, 3]
    normal_view = safe_normalize(normal_view)  # shape: [1, H, W, 3]
    normal_img = (normal_view * 0.5 + 0.5).clamp(0, 1)  # shape: [1, H, W, 3]
    normal_img = torch.where(alpha > 0, normal_img, torch.tensor(1.0, device=normal_img.device))  # shape: [1, H, W, 3]

    if mesh.metallicRoughness is not None and texc is not None:
        metallic_roughness = dr.texture(mesh.metallicRoughness.unsqueeze(0), texc, filter_mode="linear")  # shape: [1, H, W, 3]
        metallic = metallic_roughness[..., 2:3]  # shape: [1, H, W, 1]
        roughness = metallic_roughness[..., 1:2]  # shape: [1, H, W, 1]
    else:
        metallic = torch.ones_like(albedo[..., :1])  # shape: [1, H, W, 1]
        roughness = torch.ones_like(albedo[..., :1])  # shape: [1, H, W, 1]

    xyzs, _ = dr.interpolate(mesh.v.unsqueeze(0), rast, mesh.f)  # shape: [1, H, W, 3]
    viewdir = safe_normalize(xyzs - pose[:3, 3])  # shape: [1, H, W, 3]

    n_dot_v = (normal * viewdir).sum(-1, keepdim=True).clamp(min=1e-4)  # shape: [1, H, W, 1]
    reflective = n_dot_v * normal * 2 - viewdir  # shape: [1, H, W, 3]

    diffuse_albedo = (1 - metallic) * albedo  # shape: [1, H, W, 3]
    fg_uv = torch.cat([n_dot_v, roughness], -1).clamp(0, 1)  # shape: [1, H, W, 2]
    fg = dr.texture(FG_LUT, fg_uv.reshape(1, -1, 1, 2).contiguous(), filter_mode="linear", boundary_mode="clamp").reshape(1, H, W, 2)  # shape: [1, H, W, 2]
    F0 = (1 - metallic) * 0.04 + metallic * albedo  # shape: [1, H, W, 3]
    specular_albedo = F0 * fg[..., 0:1] + fg[..., 1:2]  # shape: [1, H, W, 3]

    diffuse_light = light(normal)  # shape: [1, H, W, 3]
    specular_light = light(reflective, roughness)  # shape: [1, H, W, 3]

    color = diffuse_albedo * diffuse_light + specular_albedo * specular_light  # shape: [1, H, W, 3]
    color = color * alpha + bg_color * (1 - alpha)  # shape: [1, H, W, 3]

    color_buffer = color[0].detach().cpu().numpy()  # shape: [H, W, 3]
    color_buffer = np.clip(color_buffer, 0, 1)

    return color_buffer  # RGB float [H, W, 3] in [0,1]

def save_image(buffer_rgb, out_path):
    image = (buffer_rgb * 255).astype(np.uint8)  # shape: [H, W, 3]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # shape: [H, W, 3]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, image)
    print(f"saved to {out_path}")

def main():
    args = build_args()
    paths = resolve_paths(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("需要 CUDA 以使用 nvdiffrast 进行光栅化")

    cam = setup_camera(args, args.W, args.H)
    mesh, light, FG_LUT = load_resources(paths, device, args.front_dir, args.env_scale)

    if args.out_mode == "normal":
        bg_color = torch.tensor([0.5, 0.5, 1.0], device=device)  # shape: [3]
        buffer = render_normal(mesh, cam, device, args.H, args.W, bg_color)  # shape: [H, W, 3]
    else:
        bg_color = torch.tensor([1.0, 1.0, 1.0], device=device)  # shape: [3]
        buffer = render_pbr(mesh, light, FG_LUT, cam, device, args.H, args.W, bg_color)  # shape: [H, W, 3]

    save_image(buffer, paths["out"])

if __name__ == "__main__":
    main()