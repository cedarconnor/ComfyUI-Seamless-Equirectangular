"""Run original SpheRoPE LTX 2.3 with existing Comfy-packaged local weights.

No ComfyUI modules are imported. Requires the standalone native LTX environment.
The adapter renames Gemma keys, extracts its embedded tokenizer, and rejects
non-default streaming requests unsupported by the pinned native LTX revision.
"""
import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
import subprocess

for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ[name] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"


def export_frames(frames, folder, case):
    from PIL import Image
    import torch
    if not torch.isfinite(frames).all():
        raise ValueError("Decoder returned non-finite pixels")
    for i, frame in enumerate(frames):
        if frame.dtype == torch.uint8:
            pixels = frame.numpy()
        else:
            pixels = (frame.float().clamp(0, 1) * 255).round().to(torch.uint8).numpy()
        Image.fromarray(pixels).save(folder / f"{case}_{i:03d}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument("--ltx-repo", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", choices=("stock", "full", "rope_decode"), default="full")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--disable-triton-fusion", action="store_true",
                        help="Use native deterministic PyTorch LoRA fusion when Triton cannot compile")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=80084)
    parser.add_argument("--gamma", type=float, default=3.0)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--prompt", default="A modern sunlit living room with cream walls, warm oak floorboards, a gray fabric sofa, low wooden coffee table, tall windows with sheer curtains, a floor lamp and potted plants, realistic architectural photography. A stationary camera, subtle light changes.")
    args = parser.parse_args()
    for path in (args.reference_repo, args.ltx_repo / "packages/ltx-core/src", args.ltx_repo / "packages/ltx-pipelines/src"):
        sys.path.insert(0, str(path.resolve()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import torch
    from safetensors import safe_open
    from transformers import GemmaTokenizer
    from ltx_core.loader import LoraPathStrengthAndSDOps
    from ltx_core.loader.sd_ops import SDOps, KeyValueOperationResult
    from ltx_core.loader.module_ops import ModuleOps
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.text_encoders.gemma.encoders.encoder_configurator import GemmaTextEncoderConfigurator, GEMMA_MODEL_OPS
    from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer
    from ltx_core.quantization.fp8_cast import build_policy
    from ltx_core.quantization import fp8_cast
    from ltx_pipelines.utils.blocks import PromptEncoder, DiffusionStage
    from ltx_pipelines.utils.args import ImageConditioningInput
    from ltx_pipelines.utils.denoisers import FactoryGuidedDenoiser
    from ltx_pipelines.utils.types import DenoisedLatentResult
    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_core.model.video_vae import TilingConfig, TemporalTilingConfig
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP
    import panorama_diffusers.pipelines.pipeline_ltx23_erp as erp

    gemma = args.models / "text_encoders/gemma_3_12B_it.safetensors"
    checkpoint = args.models / "checkpoints/ltx-2.3-22b-dev-fp8.safetensors"
    lora = args.models / "loras/ltx-2.3-22b-distilled-lora-384.safetensors"
    upscaler = args.models / "latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
    for path in (gemma, checkpoint, lora, upscaler):
        if not path.is_file():
            raise FileNotFoundError(path)

    def remap(key, value):
        if key.startswith("model."):
            dest = "model.model.language_model." + key[len("model."):]
        elif key.startswith("vision_model."):
            dest = "model.model.vision_tower." + key
        elif key.startswith("multi_modal_projector."):
            dest = "model.model." + key
        else:
            raise ValueError(f"Unexpected Gemma key: {key}")
        out = [KeyValueOperationResult(dest, value)]
        if key == "model.embed_tokens.weight":
            out.append(KeyValueOperationResult("model.lm_head.weight", value))
        return out

    gemma_ops = (SDOps("LOCAL_GEMMA3_NAMES")
                 .with_matching(prefix="model.")
                 .with_matching(prefix="vision_model.")
                 .with_matching(prefix="multi_modal_projector.")
                 .with_kv_operation(operation=remap))
    meta = GemmaTextEncoderConfigurator.from_config({})
    expected = {k: tuple(v.shape) for k, v in meta.state_dict().items()}
    mapped = {}
    tokenizer_dir = args.output_dir / "tokenizer"
    tokenizer_dir.mkdir(exist_ok=True)
    with safe_open(str(gemma), framework="pt") as f:
        (tokenizer_dir / "tokenizer.model").write_bytes(f.get_tensor("spiece_model").numpy().tobytes())
        for key in f.keys():
            if key == "spiece_model":
                continue
            value = torch.empty(f.get_slice(key).get_shape(), device="meta")
            for dest, tensor in remap(key, value):
                mapped[dest] = tuple(tensor.shape)
    assert mapped == expected, {"missing": sorted(expected.keys() - mapped.keys()), "extra": sorted(mapped.keys() - expected.keys()), "shape_mismatch": [k for k in mapped.keys() & expected.keys() if mapped[k] != expected[k]]}
    del meta
    # Native wrapper uses AutoTokenizer locally; save only metadata plus the
    # SentencePiece model already present inside the user's encoder checkpoint.
    tokenizer = GemmaTokenizer(vocab_file=str(tokenizer_dir / "tokenizer.model"), add_bos_token=True, add_eos_token=False)
    tokenizer.save_pretrained(str(tokenizer_dir))
    native_tokenizer = LTXVGemmaTokenizer(str(tokenizer_dir), 1024)
    import sentencepiece
    sp = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer_dir / "tokenizer.model"))
    for prompt in (args.prompt, erp.ERP_VIDEO_NEGATIVE, args.prompt + ". " + erp.ERP_VIDEO_PROMPT):
        tokens = [int(t) for t, mask in native_tokenizer.tokenize_with_weights(prompt)["gemma"] if mask]
        assert tokens == [sp.bos_id()] + sp.encode(prompt.strip()), "Tokenizer parity failed"
    quantization = build_policy(checkpoint)
    if args.disable_triton_fusion:
        fp8_cast.TRITON_AVAILABLE = False
        # Exercise the exact fallback before loading the 22B transformer.
        weight = torch.tensor([0.25, -0.5], device="cuda").to(torch.float8_e4m3fn)
        delta = torch.tensor([0.125, 0.25], device="cuda", dtype=torch.bfloat16)
        fused = fp8_cast.fuse_cast_fp8_weight(delta.clone(), weight)
        assert torch.equal(fused.float(), (weight.bfloat16() + delta).to(weight.dtype).float())
    commits = {name: subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
               for name, path in (("spherope", args.reference_repo), ("ltx", args.ltx_repo))}
    probe = {"gemma_shapes_matched": len(mapped), "tokenizer_parity": True, "weights_downloaded": False,
             "torch": torch.__version__, "checkpoint": str(checkpoint), "gemma": str(gemma), "commits": commits,
             "quantization": "native fp8_cast policy with checkpoint scale folding",
             "disable_triton_fusion": args.disable_triton_fusion}
    (args.output_dir / "adapter_probe.json").write_text(json.dumps(probe, indent=2))
    print(json.dumps(probe), flush=True)
    if args.probe_only:
        return

    tokenizer_op = ModuleOps("LocalTokenizer", matcher=lambda m: hasattr(m, "tokenizer") and m.tokenizer is None,
                             mutator=lambda m: (setattr(m, "tokenizer", native_tokenizer) or m))
    builder = SingleGPUModelBuilder(model_class_configurator=GemmaTextEncoderConfigurator,
                                    model_path=str(gemma), model_sd_ops=gemma_ops,
                                    module_ops=(GEMMA_MODEL_OPS, tokenizer_op))

    class LocalPromptEncoder(PromptEncoder):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw, text_encoder_builder=builder)

        def __call__(self, *a, streaming_prefetch_count=None, **kw):
            if streaming_prefetch_count is not None or kw.get("enhance_first_prompt"):
                raise ValueError("This local adapter supports default streaming and no prompt enhancement")
            return super().__call__(*a, **kw)

    class LocalDiffusionStage(DiffusionStage):
        def __call__(self, *a, streaming_prefetch_count=None, **kw):
            if streaming_prefetch_count is not None:
                raise ValueError("Non-default streaming is unsupported by the pinned native LTX revision")
            result = super().__call__(*a, **kw)
            if kw["width"] == args.width:
                torch.save({"video": result[0].latent.cpu(), "audio": result[1].latent.cpu()},
                           args.output_dir / (args.case + "_latents.pt"))
            return result

    class TensorGuidedDenoiser(FactoryGuidedDenoiser):
        def __call__(self, *a, **kw):
            # SpheRoPE's custom stage-one loop predates the native result wrapper.
            return tuple(r.denoised if isinstance(r, DenoisedLatentResult) else r
                         for r in super().__call__(*a, **kw))

    erp.PromptEncoder = LocalPromptEncoder
    erp.DiffusionStage = LocalDiffusionStage
    erp.FactoryGuidedDenoiser = TensorGuidedDenoiser
    pipeline = erp.LTX23ERPPipeline(
        checkpoint_path=str(checkpoint),
        distilled_lora=[LoraPathStrengthAndSDOps(str(lora), 0.8, LTXV_LORA_COMFY_RENAMING_MAP)],
        spatial_upsampler_path=str(upscaler), gemma_root=str(tokenizer_dir),
        quantization=quantization,
    )
    parameters = dict(prompt=args.prompt, seed=args.seed, width=args.width, height=args.height,
                      num_frames=args.frames, num_inference_steps=args.steps,
                      enable_wrapped_noise=False, enable_sfc_rope=args.case != "stock",
                      enable_circular_decoding=args.case != "stock",
                      enable_semantic_distortion_cfg=args.case == "full",
                      semantic_distortion_gamma=args.gamma, enhance_prompt=False)
    if args.image:
        parameters["images"] = [ImageConditioningInput(str(args.image.resolve()), 0, 1.0)]
    # Three-way upstream guidance only uses CFG and ignores STG/rescale/modality
    # settings. Disable those extras in controls to isolate the ERP features.
    guidance = {"video_guider_params": MultiModalGuiderParams(cfg_scale=3.0),
                "audio_guider_params": MultiModalGuiderParams(cfg_scale=7.0)}
    tiling = TilingConfig(temporal_config=TemporalTilingConfig(32, 16)) if args.frames > 17 else None
    started = time.time()
    with torch.inference_mode():
        video, audio = pipeline(**parameters, **guidance, tiling_config=tiling)
        frames = torch.cat(list(video), dim=0).cpu()
    print("Decoded shape", frames.shape, "range", frames.min().item(), frames.max().item(), flush=True)
    torch.save({"frames": frames, "audio": audio.waveform.cpu(), "sample_rate": audio.sampling_rate}, args.output_dir / (args.case + ".pt"))
    export_frames(frames, args.output_dir, args.case)
    result = {**probe, "parameters": parameters, "seconds": time.time() - started,
              "decoded_shape": list(frames.shape), "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
              "temporal_tiling": {"size": 32, "overlap": 16} if tiling is not None else None,
              "guidance": {"video_cfg": 3.0, "audio_cfg": 7.0, "stg": 0.0, "rescale": 0.0, "modality": 1.0}}
    (args.output_dir / f"{args.case}_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
