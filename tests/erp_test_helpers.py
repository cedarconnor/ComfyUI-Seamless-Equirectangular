class FakeLTXDiffusionModel:
    """Mimics the attribute surface of comfy LTXBaseModel that runtime_compat reads."""
    def __init__(self, missing: set[str] | None = None):
        missing = missing or set()
        self.inner_dim = 4096
        self.positional_embedding_theta = 10000.0
        self.positional_embedding_max_pos = [20, 2048, 2048]
        self.num_attention_heads = 32
        # split_positional_embedding is an enum-like value in comfy; a string stands in here
        self.split_positional_embedding = "rope_type"
        if "_precompute_freqs_cis" not in missing:
            self._precompute_freqs_cis = lambda *a, **k: None
        if "_prepare_positional_embeddings" not in missing:
            self._prepare_positional_embeddings = lambda *a, **k: None

class FakeModelPatcher:
    """Mimics ComfyUI ModelPatcher's get_model_object for the diffusion_model key."""
    def __init__(self, diffusion_model):
        self._objs = {"diffusion_model": diffusion_model}
        self.object_patches: dict = {}
        self.object_patches_backup: dict = {}
        self.model_options: dict = {}

    def is_dynamic(self) -> bool:
        return False

    def get_model_object(self, name: str):
        if name in self.object_patches:
            return self.object_patches[name]
        if name not in self._objs:
            raise AttributeError(name)
        return self._objs[name]

    def clone(self):
        n = FakeModelPatcher(self._objs["diffusion_model"])
        n.object_patches = dict(self.object_patches)
        return n

    def add_object_patch(self, name: str, obj) -> None:
        self.object_patches[name] = obj

class FakeFlux2DM:
    """Minimal Flux.2 diffusion-model surface the Flux RoPE patch reads: a real comfy
    EmbedND at .pe_embedder with the FLUX.2 config (4 axes, theta=2000)."""
    def __init__(self, axes_dim=(32, 32, 32, 32), theta=2000):
        from comfy.ldm.flux.layers import EmbedND
        self.pe_embedder = EmbedND(dim=sum(axes_dim), theta=theta, axes_dim=list(axes_dim))

def make_flux2_ids(H, W, n_txt=7, batch=1):
    """Build a (batch, n_txt + H*W, 4) ids tensor mimicking comfy's cat(txt_ids, img_ids)
    for FLUX.2: text tokens carry positions only on axis 3 (txt_ids_dims=[3]); image
    tokens carry T=0, H=row, W=col, L=0 in (h outer, w inner) order."""
    import torch
    txt = torch.zeros((n_txt, 4), dtype=torch.float32)
    txt[:, 3] = torch.linspace(0, n_txt - 1, steps=n_txt)
    rows = torch.arange(H, dtype=torch.float32)
    cols = torch.arange(W, dtype=torch.float32)
    rg, cg = torch.meshgrid(rows, cols, indexing="ij")
    img = torch.zeros((H * W, 4), dtype=torch.float32)
    img[:, 1] = rg.reshape(-1)               # H
    img[:, 2] = cg.reshape(-1)               # W
    ids = torch.cat([txt, img], dim=0)       # (n_txt + H*W, 4)
    return ids.unsqueeze(0).expand(batch, -1, -1).contiguous()
