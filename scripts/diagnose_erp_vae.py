"""Build and analyze a no-denoising ERP VAE boundary experiment.

The synthetic control is sampled from continuous functions of a unit sphere.
It has no projection seam or longitude-dependent value at either pole.
No model, LoRA, prompt or sampler is involved in these workflows.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspect_native_video import perspective


def build(args):
    args.output.mkdir(parents=True, exist_ok=True)
    args.input_dir.mkdir(parents=True, exist_ok=True)
    h, w = 512, 1024
    lon = ((np.arange(w) + .5) / w - .5) * (2 * np.pi)
    lat = (.5 - (np.arange(h) + .5) / h) * np.pi
    lon, lat = np.meshgrid(lon, lat)
    x, y, z = np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)
    rgb = np.stack((.5 + .22*x + .12*np.sin(13*z + 4*y),
                    .5 + .22*y + .12*np.sin(11*x - 3*z),
                    .5 + .22*z + .12*np.sin(9*y + 5*x)), -1)
    synthetic = (rgb.clip(0, 1) * 255).round().astype(np.uint8)
    photo = np.array(Image.open(args.image).convert("RGB").resize((w, h), Image.Resampling.LANCZOS))
    for name, pixels in (("sphere", synthetic), ("pano13", photo)):
        Image.fromarray(pixels).save(args.input_dir / f"erp_probe_{name}.png")
        # Reference-only wrapped pixels: keep all original pixels unchanged.
        padded = np.concatenate((pixels[:, -256:], pixels, pixels[:, :256]), axis=1)
        Image.fromarray(padded).save(args.input_dir / f"erp_probe_{name}_padded.png")
        Image.fromarray(pixels).save(args.output / f"{name}_source.png")

    workflows = []
    for version in ("23", "25"):
        graph = {}
        if version == "23":
            graph["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {
                "ckpt_name": "ltx-2.3-22b-dev-fp8.safetensors"}}
            vae = ["1", 2]
        else:
            graph["1"] = {"class_type": "VAELoader", "inputs": {
                "vae_name": "ltx-2.5-video-vae-bf16.safetensors"}}
            vae = ["1", 0]
        def node(cls, **inputs):
            key = str(len(graph) + 1)
            graph[key] = {"class_type": cls, "inputs": inputs}
            return [key, 0]
        for name in ("sphere", "pano13"):
            pixels = node("LoadImage", image=f"erp_probe_{name}.png")
            latent = node("VAEEncode", pixels=pixels, vae=vae)
            stock = node("VAEDecode", samples=latent, vae=vae)
            circular = node("EquirectCircularVAEDecodeLTX", samples=latent, vae=vae,
                            pad_w=8, target_width=1024)
            padded = node("LoadImage", image=f"erp_probe_{name}_padded.png")
            padded_latent = node("VAEEncode", pixels=padded, vae=vae)
            padded_decode = node("VAEDecode", samples=padded_latent, vae=vae)
            both = node("ImageCrop", image=padded_decode, width=1024, height=512, x=256, y=0)
            for mode, image in (("stock", stock), ("decode_wrap", circular), ("both_wrap", both)):
                node("SaveImage", images=image,
                     filename_prefix=f"erp-boundary-20261019/ltx{version}_{name}_{mode}")
        file = args.output / f"vae_ltx{version}.json"
        file.write_text(json.dumps(graph, indent=2), encoding="utf-8")
        workflows.append(str(file.resolve()))
    print(json.dumps({"workflows": workflows}))


def analyze(args):
    rows = []
    canvas = Image.new("RGB", (4*320, 4*348), "#222222")
    draw = ImageDraw.Draw(canvas)
    for row, (version, name) in enumerate((("23", "sphere"), ("25", "sphere"),
                                          ("23", "pano13"), ("25", "pano13"))):
        source = np.array(Image.open(args.output / f"{name}_source.png").convert("RGB"))
        for col, mode in enumerate(("source", "stock", "decode_wrap", "both_wrap")):
            if mode == "source":
                image = source
            else:
                matches = sorted(args.raw.glob(f"ltx{version}_{name}_{mode}_*.png"))
                if len(matches) != 1:
                    raise ValueError(f"Expected exactly one output for {version}/{name}/{mode}: {matches}")
                image = np.array(Image.open(matches[0]).convert("RGB"))
            f = image.astype(np.float32)
            jumps = np.abs(f - np.roll(f, 1, axis=1)).mean((0, 2))
            error = f - source.astype(np.float32)
            rows.append({"version": version, "input": name, "mode": mode,
                         "edge_mae": float(jumps[0]),
                         "edge_ratio": float(jumps[0] / max(np.median(jumps), 1e-8)),
                         "image_mae": float(np.abs(error).mean()),
                         "boundary_residual_mae": float(np.abs(error[:, 0]-error[:, -1]).mean())})
            view = perspective(image, np.pi, 0, size=320)
            canvas.paste(Image.fromarray(view), (col*320, row*348+28))
            draw.text((col*320+5, row*348+6), f"LTX{version} {name} {mode}", fill="white")
    canvas.save(args.output / "vae_join_comparison.jpg")
    (args.output / "vae_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


def dual(args):
    graph = json.loads((args.output / "vae_ltx25.json").read_text(encoding="utf-8"))
    # Retain the identical encoding inputs but only save the new decoder controls.
    graph = {k: v for k, v in graph.items() if v["class_type"] != "SaveImage"}
    encodes = [(k, v) for k, v in graph.items() if v["class_type"] == "VAEEncode"]
    key = max(map(int, graph)) + 1
    for latent_id, encode in encodes:
        file = graph[encode["inputs"]["pixels"][0]]["inputs"]["image"]
        if "_padded" in file:
            continue
        name = file.removeprefix("erp_probe_").removesuffix(".png")
        for blend in (0, 1, 4):
            graph[str(key)] = {"class_type": "EquirectLTX25VAEDecode", "inputs": {
                "samples": [latent_id, 0], "vae": ["1", 0], "blend_columns": blend}}
            graph[str(key+1)] = {"class_type": "SaveImage", "inputs": {
                "images": [str(key), 0],
                "filename_prefix": f"erp-boundary-20261019/ltx25_{name}_dual{blend}"}}
            key += 2
    reachable = set()
    def visit(node_id):
        if node_id in reachable:
            return
        reachable.add(node_id)
        for value in graph[node_id]["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and str(value[0]) in graph:
                visit(str(value[0]))
    for node_id, value in graph.items():
        if value["class_type"] == "SaveImage":
            visit(node_id)
    graph = {k: v for k, v in graph.items() if k in reachable}
    path = args.output / "vae_ltx25_dual.json"
    path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    print(path)


def analyze_dual(args):
    rows = []
    canvas = Image.new("RGB", (1280, 696), "#222222")
    draw = ImageDraw.Draw(canvas)
    for row, name in enumerate(("sphere", "pano13")):
        source = np.array(Image.open(args.output / f"{name}_source.png").convert("RGB"))
        for col, mode in enumerate(("source", "dual0", "dual1", "dual4")):
            if mode == "source":
                image = source
            else:
                files = sorted(args.raw.glob(f"ltx25_{name}_{mode}_*.png"))
                if len(files) != 1:
                    raise ValueError(files)
                image = np.array(Image.open(files[0]).convert("RGB"))
            f = image.astype(np.float32)
            jumps = np.abs(f - np.roll(f, 1, axis=1)).mean((0, 2))
            residual = f - source.astype(np.float32)
            rows.append({"input": name, "mode": mode, "edge_mae": float(jumps[0]),
                         "boundary_residual_mae": float(np.abs(residual[:, 0]-residual[:, -1]).mean()),
                         "image_mae": float(np.abs(residual).mean())})
            canvas.paste(Image.fromarray(perspective(image, np.pi, 0, 320)),
                         (col*320, row*348+28))
            draw.text((col*320+5, row*348+6), f"LTX25 {name} {mode}", fill="white")
    canvas.save(args.output / "vae_dual_comparison.jpg")
    (args.output / "vae_dual_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("build", "analyze", "dual", "analyze-dual"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--input-dir", type=Path)
    p.add_argument("--image", type=Path)
    p.add_argument("--raw", type=Path)
    args = p.parse_args()
    {"build": build, "analyze": analyze, "dual": dual,
     "analyze-dual": analyze_dual}[args.action](args)
