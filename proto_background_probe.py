"""
proto_background_probe.py
=========================
Kiểm thử giả thuyết (b): Prototype có bị thu hút về phía high-norm token
(background distraction) trong quá trình train hay không?

Chạy trên Colab:
    !pip install timm tqdm
    !python proto_background_probe.py

Metrics theo dõi mỗi epoch:
  1. norm_sim_bias  : avg_sim(proto, high-norm tokens) - avg_sim(proto, low-norm tokens)
                      > 0 và tăng dần → prototype bị kéo về phía high-norm token
  2. proto_bg_cosine: cosine(prototype_vector, avg_high_norm_token_direction)
                      tăng dần → prototype direction align với background direction
  3. norm_map       : bản đồ spatial của norm token trung bình
  4. background_frac: % sample mà max-activation patch nằm ở high-norm region
"""

# ─────────────────────────────────────────────
# 0.  Dependencies
# ─────────────────────────────────────────────
import os, sys, math, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib
matplotlib.use('Agg')           # headless-safe (Colab writes to file)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# ─────────────────────────────────────────────
# 1.  CONFIG
# ─────────────────────────────────────────────
CFG = dict(
    # Dataset
    dataset        = 'CIFAR10',    # tự tải — torchvision
    img_size       = 224,          # resize lên 224 cho DeiT-patch16
    num_classes    = 10,
    data_root      = './data',

    # Model
    backbone       = 'deit_tiny_patch16_224',   # DeiT-tiny từ timm
    proto_dim      = 192,          # embed_dim của DeiT-tiny
    proto_per_cls  = 5,            # số prototype / class
    addon_type     = 'regular',    # 'regular' | 'bottleneck'

    # Training
    batch_size     = 64,           # tăng lên 64 cho GPU throughput tốt hơn
    num_epochs     = 15,
    train_subset   = 10000,        # dùng 10K ảnh train (None = full 50K)
    use_amp        = True,         # AMP (mixed precision) — giảm ~50% thời gian
    lr_backbone    = 2e-5,         # nhỏ — tương tự fine-tune trong ProtoPFormer
    lr_addon       = 1e-3,
    lr_proto       = 1e-3,
    # device được xác định lúc runtime bởi get_device() — không đặt ở đây

    # Probe
    probe_every    = 1,            # probe mỗi N epoch
    probe_samples  = 300,          # số sample dùng để đo metrics
    highnorm_pct   = 0.20,         # top 20% norm = "background candidate"
    lownorm_pct    = 0.20,         # bottom 20% norm = "foreground candidate"

    # Output
    save_dir       = './proto_probe_results',
)
CFG['num_proto'] = CFG['num_classes'] * CFG['proto_per_cls']

# ─────────────────────────────────────────────
# 2.  MODEL  (simplified ProtoPFormer, cùng cơ chế L2)
# ─────────────────────────────────────────────
def _extract_patch_tokens(backbone, x):
    """
    Forward qua DeiT backbone, trả về patch token features (KHÔNG phải CLS).
    Shape: (B, N, D)  với N = 196 cho ảnh 224×224, patch_size=16.
    """
    B = x.shape[0]
    # PatchEmbed
    x_patch = backbone.patch_embed(x)                        # (B, 196, D)
    # Ghép CLS + pos_embed
    cls_tok = backbone.cls_token.expand(B, -1, -1)           # (B, 1, D)
    x_full  = torch.cat([cls_tok, x_patch], dim=1)           # (B, 197, D)
    x_full  = x_full + backbone.pos_embed                    # broadcast pos
    if hasattr(backbone, 'pos_drop'):
        x_full = backbone.pos_drop(x_full)
    # Transformer blocks
    for blk in backbone.blocks:
        x_full = blk(x_full)
    x_full = backbone.norm(x_full)                           # final LayerNorm
    return x_full[:, 1:]                                     # (B, 196, D) — patch only


class SimpleProtoPNet(nn.Module):
    """
    Stripped-down ProtoPNet sát với ProtoPFormer:
      backbone → add_on (Conv1×1 + Sigmoid) → L2 distance → log-similarity → last_layer
    Không có token pruning để cho clean hơn.
    """
    epsilon = 1e-4

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        D  = cfg['proto_dim']
        P  = cfg['num_proto']
        NC = cfg['num_classes']

        # ── Backbone ──────────────────────────────────
        try:
            from timm import create_model
        except ImportError:
            raise ImportError("pip install timm")
        self.backbone = create_model(cfg['backbone'], pretrained=True, num_classes=0)
        # num_classes=0 → xoá classification head

        # ── Add-on layers (y chang ProtoPFormer) ──────
        if cfg['addon_type'] == 'bottleneck':
            self.add_on = nn.Sequential(
                nn.Conv2d(D, D, kernel_size=1), nn.ReLU(),
                nn.Conv2d(D, D, kernel_size=1), nn.Sigmoid()
            )
        else:  # regular
            self.add_on = nn.Sequential(
                nn.Conv2d(D, D, kernel_size=1),
                nn.Sigmoid()
            )

        # ── Prototype vectors ──────────────────────────
        # Shape: (P, D, 1, 1) — cùng format với ProtoPFormer
        self.prototypes = nn.Parameter(torch.rand(P, D, 1, 1))

        # ── Last layer (fixed, class-specific) ─────────
        self.last_layer = nn.Linear(P, NC, bias=False)
        self._init_last_layer()
        self.last_layer.weight.requires_grad = False

    # ── helpers ──────────────────────────────────────
    def _init_last_layer(self):
        w = torch.full((self.cfg['num_classes'], self.cfg['num_proto']), -0.5)
        ppc = self.cfg['proto_per_cls']
        for c in range(self.cfg['num_classes']):
            w[c, c * ppc : (c + 1) * ppc] = 1.0
        self.last_layer.weight.data.copy_(w)

    def _l2_conv(self, fmap, proto):
        """
        fmap : (B, D, H, W)
        proto: (P, D, 1, 1)
        Trả về distances (B, P, H, W) — y chang _l2_convolution_single()
        """
        ones = torch.ones_like(proto)
        x2   = F.conv2d(fmap ** 2, ones)                # (B, P, H, W)
        p2   = (proto ** 2).sum(dim=(1,2,3)).view(-1,1,1)  # (P,1,1)
        xp   = F.conv2d(fmap, proto)                    # (B, P, H, W)
        return F.relu(x2 - 2 * xp + p2)

    def _dist2sim(self, d):
        return torch.log((d + 1) / (d + self.epsilon))

    # ── forward ──────────────────────────────────────
    def forward(self, x):
        B = x.shape[0]

        # ① Backbone → patch tokens
        tokens = _extract_patch_tokens(self.backbone, x)    # (B, N, D)
        N, D   = tokens.shape[1], tokens.shape[2]
        H = W  = int(N ** 0.5)

        # ② Reshape → (B, D, H, W) cho Conv2d
        fmap = tokens.permute(0, 2, 1).reshape(B, D, H, W)

        # ③ Add-on layers
        fmap = self.add_on(fmap)                            # (B, D, H, W)

        # ④ L2 distance → similarity
        dist = self._l2_conv(fmap, self.prototypes)         # (B, P, H, W)
        sim  = self._dist2sim(dist)                         # (B, P, H, W)

        # ⑤ Global max-pool → logits
        proto_act = F.max_pool2d(sim, (H, W)).view(B, -1)  # (B, P)
        logits     = self.last_layer(proto_act)

        # Trả thêm thông tin trung gian để probe
        return logits, dict(tokens=tokens, fmap=fmap, dist=dist, sim=sim)

# ─────────────────────────────────────────────
# 3.  PROBE  (đo metrics mỗi epoch)
# ─────────────────────────────────────────────
@torch.no_grad()
def run_probe(model, loader, cfg):
    """
    Đo 4 metrics:
      norm_sim_bias   : avg_sim(proto, high-norm) - avg_sim(proto, low-norm)  shape (P,)
      bg_cosine       : cosine(proto_vec, avg_high_norm_feat)                  shape (P,)
      avg_norm_map    : spatial map của token norm trung bình                  shape (H,W)
      bg_frac         : % sample mà max-sim patch nằm trong high-norm region   scalar
    """
    model.eval()
    device = next(model.parameters()).device

    all_raw_norms = []   # backbone output norm  (N_tok per sample)
    all_sims      = []   # (P, N_tok)
    all_feats     = []   # (N_tok, D) — after add_on (flattened)
    all_max_patch = []   # index của max-sim patch mỗi sample

    for imgs, _ in loader:
        imgs = imgs.to(device)
        _, aux = model(imgs)

        B, N, D = aux['tokens'].shape
        H = W   = int(aux['fmap'].shape[-1])

        # norm của backbone raw token (trước add_on)
        raw_norms = aux['tokens'].norm(dim=-1).cpu()          # (B, N)

        # similarity maps → flatten spatial
        sim_flat  = aux['sim'].flatten(2).cpu()               # (B, P, N)

        # features sau add_on (flatten spatial)
        feat_flat = aux['fmap'].flatten(2).permute(0,2,1).cpu()  # (B, N, D)

        # patch có similarity cao nhất (tổng qua tất cả proto)
        max_patch = sim_flat.sum(dim=1).argmax(dim=1)         # (B,) — index trong N

        all_raw_norms.append(raw_norms)
        all_sims.append(sim_flat)
        all_feats.append(feat_flat)
        all_max_patch.append(max_patch)

    all_raw_norms = torch.cat(all_raw_norms, 0)   # (S, N)
    all_sims      = torch.cat(all_sims,      0)   # (S, P, N)
    all_feats     = torch.cat(all_feats,     0)   # (S, N, D)
    all_max_patch = torch.cat(all_max_patch, 0)   # (S,)

    S, N = all_raw_norms.shape
    P    = all_sims.shape[1]
    k_hi = max(1, int(N * cfg['highnorm_pct']))
    k_lo = max(1, int(N * cfg['lownorm_pct']))

    # ── Metric 1: norm_sim_bias ────────────────────────────────────────────────
    bias_per_sample = []
    for i in range(S):
        norms_i = all_raw_norms[i]
        sims_i  = all_sims[i]                   # (P, N)
        top_hi  = norms_i.topk(k_hi).indices    # top 20% norm
        top_lo  = norms_i.topk(k_lo, largest=False).indices   # bottom 20%
        hi_sim  = sims_i[:, top_hi].mean(dim=1) # (P,)
        lo_sim  = sims_i[:, top_lo].mean(dim=1) # (P,)
        bias_per_sample.append(hi_sim - lo_sim)
    norm_sim_bias = torch.stack(bias_per_sample).mean(0)   # (P,)

    # ── Metric 2: proto_bg_cosine ──────────────────────────────────────────────
    # Hướng trung bình của các high-norm token (trong feature space sau add_on)
    hi_feat_list = []
    for i in range(S):
        top_hi  = all_raw_norms[i].topk(k_hi).indices
        hi_feat_list.append(all_feats[i][top_hi].mean(0))    # (D,)
    avg_hi_feat  = torch.stack(hi_feat_list).mean(0)          # (D,)
    avg_hi_unit  = F.normalize(avg_hi_feat.unsqueeze(0), dim=1).squeeze(0)   # unit

    # Proto vectors (P, D)
    proto_vecs    = model.prototypes.squeeze(-1).squeeze(-1).cpu()  # (P, D)
    proto_unit    = F.normalize(proto_vecs, dim=1)                  # (P, D)
    bg_cosine     = (proto_unit @ avg_hi_unit)                      # (P,)

    # ── Metric 3: avg_norm_map ─────────────────────────────────────────────────
    avg_norm_map  = all_raw_norms.mean(0)                  # (N,)
    H_s = int(N ** 0.5)
    avg_norm_map  = avg_norm_map.reshape(H_s, H_s).numpy()

    # ── Metric 4: background_frac ──────────────────────────────────────────────
    # % sample mà max-sim patch nằm trong top-norm region
    hi_set_list = []
    for i in range(S):
        top_hi = set(all_raw_norms[i].topk(k_hi).indices.tolist())
        hi_set_list.append(int(all_max_patch[i].item()) in top_hi)
    bg_frac = float(np.mean(hi_set_list))

    return dict(
        norm_sim_bias        = norm_sim_bias.numpy(),           # (P,)
        bg_cosine            = bg_cosine.numpy(),               # (P,)
        avg_norm_map         = avg_norm_map,                    # (H, W)
        bg_frac              = bg_frac,                         # scalar
        mean_norm_sim_bias   = norm_sim_bias.mean().item(),
        mean_bg_cosine       = bg_cosine.mean().item(),
    )

# ─────────────────────────────────────────────
# 4.  VISUALIZATION
# ─────────────────────────────────────────────
def plot_probe(history, save_dir, probe_sample_imgs, model, epoch, cfg):
    """
    Tạo dashboard gồm 6 panel mỗi epoch.
    Lưu file PNG và cũng show inline nếu trong Colab.
    """
    os.makedirs(save_dir, exist_ok=True)

    epochs   = [h['epoch']               for h in history]
    bias_ts  = [h['mean_norm_sim_bias']  for h in history]
    cos_ts   = [h['mean_bg_cosine']      for h in history]
    bgfrac   = [h['bg_frac']             for h in history]
    cur      = history[-1]

    fig = plt.figure(figsize=(20, 13))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel 1: norm_sim_bias over epochs ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, bias_ts, 'r-o', lw=2, ms=6, label='mean bias')
    ax1.axhline(0, color='k', ls='--', alpha=0.4)
    ax1.fill_between(epochs, bias_ts, 0,
                     where=[b > 0 for b in bias_ts], alpha=0.15, color='red')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Norm-Similarity Bias')
    ax1.set_title('① Norm-Sim Bias qua từng epoch\n'
                  'avg_sim(hi-norm) − avg_sim(lo-norm)  |  >0 = bị kéo về background')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # ── Panel 2: bg_cosine over epochs ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, cos_ts, 'b-o', lw=2, ms=6, label='mean cosine')
    ax2.axhline(0, color='k', ls='--', alpha=0.4)
    ax2.fill_between(epochs, cos_ts, 0,
                     where=[c > 0 for c in cos_ts], alpha=0.15, color='blue')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('② Proto direction vs avg high-norm direction\n'
                  '>0 và tăng = prototype align về phía background')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    # ── Panel 3: background_frac over epochs ──────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, bgfrac, 'g-^', lw=2, ms=7, label='bg_frac')
    ax3.axhline(cfg['highnorm_pct'], color='k', ls='--', alpha=0.5,
                label=f'random baseline ({cfg["highnorm_pct"]:.0%})')
    ax3.set_ylim(0, 1); ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Fraction')
    ax3.set_title('③ Background Fraction\n'
                  '% sample mà max-activation patch ∈ high-norm region')
    ax3.legend(); ax3.grid(True, alpha=0.3)

    # ── Panel 4: per-proto bias bar (current epoch) ───────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    bias  = cur['norm_sim_bias']
    colors = ['#d62728' if b > 0 else '#1f77b4' for b in bias]
    ax4.bar(range(len(bias)), bias, color=colors, alpha=0.8, edgecolor='k', lw=0.5)
    ax4.axhline(0, color='k', ls='--')
    ax4.set_xlabel('Prototype index'); ax4.set_ylabel('Norm-Sim Bias')
    n_pos = (bias > 0).sum()
    ax4.set_title(f'④ Per-prototype Norm-Sim Bias  [epoch {epoch}]\n'
                  f'Đỏ = bị kéo về hi-norm  ({n_pos}/{len(bias)} prototypes)')

    # ── Panel 5: per-proto cosine bar (current epoch) ─────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    cosin  = cur['bg_cosine']
    colors5 = ['#d62728' if c > 0 else '#1f77b4' for c in cosin]
    ax5.bar(range(len(cosin)), cosin, color=colors5, alpha=0.8, edgecolor='k', lw=0.5)
    ax5.axhline(0, color='k', ls='--')
    ax5.set_xlabel('Prototype index'); ax5.set_ylabel('Cosine w/ high-norm')
    n_pos5 = (cosin > 0).sum()
    ax5.set_title(f'⑤ Proto-BG Cosine  [epoch {epoch}]\n'
                  f'Đỏ = proto hướng về background ({n_pos5}/{len(cosin)} prototypes)')

    # ── Panel 6: token norm heatmap overlay lên ảnh mẫu ──────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    # avg_norm_map: (H, W) — resize về img_size
    from PIL import Image as PILImage
    norm_map = cur['avg_norm_map']
    H_s      = norm_map.shape[0]
    norm_img  = PILImage.fromarray(
        (255 * (norm_map - norm_map.min()) /
         (norm_map.ptp() + 1e-8)).astype(np.uint8)
    ).resize((cfg['img_size'], cfg['img_size']), PILImage.NEAREST)
    norm_arr  = np.array(norm_img) / 255.

    # Lấy 1 ảnh mẫu (đã denormalize)
    mean_v = np.array([0.485, 0.456, 0.406])
    std_v  = np.array([0.229, 0.224, 0.225])
    img_np = probe_sample_imgs[0].permute(1, 2, 0).numpy()
    img_np = np.clip(img_np * std_v + mean_v, 0, 1)

    # Overlay: hot colormap
    cmap    = plt.cm.hot
    overlay = 0.55 * img_np + 0.45 * cmap(norm_arr)[:, :, :3]
    overlay = np.clip(overlay, 0, 1)

    ax6.imshow(overlay)
    ax6.set_title(f'⑥ Token norm heatmap overlay  [epoch {epoch}]\n'
                  f'Sáng (hot) = high-norm = potential background tokens\n'
                  f'(kích thước gốc: {H_s}×{H_s} patches)')
    ax6.axis('off')

    fig.suptitle(
        f'[ProtoPFormer] Prototype ← Background Attraction Probe  |  Epoch {epoch}\n'
        f'Hypothesis (b): Gradient từ high-norm token kéo prototype về phía background',
        fontsize=13, fontweight='bold', y=1.01
    )

    path = os.path.join(save_dir, f'probe_epoch_{epoch:03d}.png')
    plt.savefig(path, bbox_inches='tight', dpi=100)
    plt.close(fig)
    print(f'  [viz] saved → {path}')

    # Nếu đang chạy trong Colab → display inline
    try:
        from IPython.display import display, Image as IPyImg
        display(IPyImg(path))
    except Exception:
        pass

# ─────────────────────────────────────────────
# 5.  DATASET
# ─────────────────────────────────────────────
def build_loaders(cfg):
    mu  = [0.485, 0.456, 0.406]
    sig = [0.229, 0.224, 0.225]
    tf_train = transforms.Compose([
        transforms.Resize((cfg['img_size'], cfg['img_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mu, sig),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((cfg['img_size'], cfg['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mu, sig),
    ])
    train_ds = torchvision.datasets.CIFAR10(
        cfg['data_root'], train=True,  download=True, transform=tf_train)
    val_ds   = torchvision.datasets.CIFAR10(
        cfg['data_root'], train=False, download=True, transform=tf_val)

    # ── Tùy chọn: chỉ dùng một phần dữ liệu train để chạy nhanh hơn ──────────────
    subset_n = cfg.get('train_subset', None)
    if subset_n is not None and subset_n < len(train_ds):
        # Lấy đều đu train subset_n mẫu mỗi class (stratified)
        indices_per_cls = [[] for _ in range(cfg['num_classes'])]
        for idx, (_, lbl) in enumerate(train_ds):
            if all(len(v) >= subset_n // cfg['num_classes'] for v in indices_per_cls):
                break
            if len(indices_per_cls[lbl]) < subset_n // cfg['num_classes']:
                indices_per_cls[lbl].append(idx)
        flat_indices = [i for sub in indices_per_cls for i in sub]
        train_ds = Subset(train_ds, flat_indices)
        print(f'[INFO] Train subset: {len(flat_indices):,} samples '
              f'({subset_n // cfg["num_classes"]} per class)')
    else:
        print(f'[INFO] Train set: {len(train_ds):,} samples (full)')

    # Subset nhỏ để probe nhanh
    probe_ds = Subset(val_ds, list(range(cfg['probe_samples'])))

    nw = min(4, os.cpu_count() or 2)   # num_workers tự động
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                              shuffle=True,  num_workers=nw, pin_memory=True,
                              persistent_workers=(nw > 0))
    probe_loader = DataLoader(probe_ds, batch_size=64,
                              shuffle=False, num_workers=nw,
                              persistent_workers=(nw > 0))
    return train_loader, probe_loader, val_ds

# ─────────────────────────────────────────────
# 6.  TRAINING LOOP
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, epoch, scaler=None):
    """
    scaler != None → dùng AMP (torch.cuda.amp) — giảm ~50% thời gian trên GPU.
    """
    model.train()
    total_loss = 0.
    correct    = 0
    total      = 0
    use_amp = (scaler is not None) and (device.type == 'cuda')
    for imgs, labels in tqdm(loader, desc=f'Epoch {epoch}', leave=False):
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        if use_amp:
            with torch.cuda.amp.autocast():
                logits, _ = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _ = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total

# ─────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────
def get_device(override: str = None) -> torch.device:
    """
    Xác định device lúc runtime — tránh evaluate sớm khi import.
    Thứ tự ưu tiên: CLI arg --device > biến môi trường PROBE_DEVICE > auto-detect.
    """
    # 1. Từ CLI / env
    chosen = override or os.environ.get('PROBE_DEVICE', '')
    if chosen:
        dev = torch.device(chosen)
        # Verify nếu yêu cầu cuda
        if dev.type == 'cuda' and not torch.cuda.is_available():
            print('[WARNING] --device cuda được yêu cầu nhưng torch.cuda.is_available() = False!')
            print('          Kiểm tra: Runtime → Change runtime type → GPU trong Colab.')
            print('          Fallback về CPU.')
            dev = torch.device('cpu')
        return dev

    # 2. Auto-detect
    if torch.cuda.is_available():
        dev = torch.device('cuda')
        props = torch.cuda.get_device_properties(0)
        print(f'[INFO] CUDA available: {props.name}  '
              f'({props.total_memory // 1024**2} MB VRAM)')
    else:
        dev = torch.device('cpu')
        print('[WARNING] CUDA không available — chạy trên CPU sẽ rất chậm!')
        print('          Trong Colab: Runtime → Change runtime type → GPU, sau đó Reconnect.')
    return dev


def parse_args():
    parser = argparse.ArgumentParser(
        description='Proto-Background Attraction Probe',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--device',       type=str,  default='',
                        help='Force device: cuda | cpu | cuda:0')
    parser.add_argument('--epochs',       type=int,  default=CFG['num_epochs'])
    parser.add_argument('--batch_size',   type=int,  default=CFG['batch_size'])
    parser.add_argument('--train_subset', type=int,  default=CFG.get('train_subset', 10000),
                        help='Số ảnh train (None=50K full, 10000=~8-12 phút trên T4)')
    parser.add_argument('--no_amp',       action='store_true',
                        help='Tắt AMP (mặc định bật)')
    parser.add_argument('--save_dir',     type=str,  default=CFG['save_dir'])
    return parser.parse_args()


def main():
    args   = parse_args()
    cfg    = CFG.copy()
    # Override từ CLI
    cfg['num_epochs']   = args.epochs
    cfg['batch_size']   = args.batch_size
    cfg['train_subset'] = args.train_subset
    cfg['use_amp']      = not args.no_amp
    cfg['save_dir']     = args.save_dir

    device = get_device(args.device if args.device else None)
    cfg['device'] = str(device)
    os.makedirs(cfg['save_dir'], exist_ok=True)

    # Ước tính thời gian (rough estimate)
    steps_per_epoch = (cfg.get('train_subset') or 50000) // cfg['batch_size']
    est_sec_per_epoch = steps_per_epoch * (0.05 if device.type == 'cuda' else 2.0)
    est_total_min = cfg['num_epochs'] * est_sec_per_epoch / 60
    amp_str = 'ON (×2 speedup)' if (cfg['use_amp'] and device.type == 'cuda') else 'OFF'

    print('=' * 65)
    print('  Proto-Background Attraction Probe')
    print('  Hypothesis (b): gradient từ high-norm token → prototype drift')
    print('=' * 65)
    print(f'  Device      : {device}  ({"GPU ✓" if device.type=="cuda" else "CPU — consider enabling GPU"})')
    print(f'  Backbone    : {cfg["backbone"]}')
    print(f'  Dataset     : {cfg["dataset"]} (auto-download, resized to {cfg["img_size"]})')
    print(f'  Train size  : {cfg.get("train_subset") or 50000:,} samples | batch {cfg["batch_size"]} | {steps_per_epoch} steps/epoch')
    print(f'  AMP         : {amp_str}')
    print(f'  Epochs      : {cfg["num_epochs"]}  |  Probe every {cfg["probe_every"]} epoch(s)')
    print(f'  Protos      : {cfg["num_proto"]} ({cfg["proto_per_cls"]} × {cfg["num_classes"]} classes)')
    print(f'  Est. time   : ~{est_total_min:.0f} min total  (rough, GPU-dependent)')
    print()

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, probe_loader, _ = build_loaders(cfg)
    # Lưu sẵn 1 batch ảnh để vẽ heatmap overlay
    probe_sample_imgs, _ = next(iter(probe_loader))

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SimpleProtoPNet(cfg).to(device)
    print(f'  Model params: {sum(p.numel() for p in model.parameters()):,}')
    print(f'    backbone  : {sum(p.numel() for p in model.backbone.parameters()):,}')
    print(f'    add_on    : {sum(p.numel() for p in model.add_on.parameters()):,}')
    print(f'    prototypes: {model.prototypes.numel():,}')

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam([
        {'params': model.backbone.parameters(), 'lr': cfg['lr_backbone']},
        {'params': model.add_on.parameters(),   'lr': cfg['lr_addon']},
        {'params': [model.prototypes],           'lr': cfg['lr_proto']},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['num_epochs'], eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    # ── AMP Scaler (None nếu tắt hoặc CPU) ─────────────────────────────────────
    scaler = None
    if cfg.get('use_amp', True) and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        print('[INFO] AMP GradScaler enabled.')

    history = []

    # ── Initial probe (epoch 0, trước khi train) ──────────────────────────────
    print('\n── Initial Probe (epoch 0 — before training) ──')
    m0 = run_probe(model, probe_loader, cfg)
    m0['epoch'] = 0; m0['train_acc'] = 0.; m0['train_loss'] = float('nan')
    history.append(m0)
    _print_metrics(0, 0., float('nan'), m0)

    # ── Training ──────────────────────────────────────────────────────────────
    for epoch in range(1, cfg['num_epochs'] + 1):
        loss, acc = train_one_epoch(model, train_loader, optimizer,
                                    criterion, device, epoch, scaler=scaler)
        scheduler.step()

        if epoch % cfg['probe_every'] == 0:
            print(f'\n── Epoch {epoch}/{cfg["num_epochs"]} ──')
            metrics = run_probe(model, probe_loader, cfg)
            metrics['epoch']      = epoch
            metrics['train_acc']  = acc
            metrics['train_loss'] = loss
            history.append(metrics)
            _print_metrics(epoch, acc, loss, metrics)
            plot_probe(history, cfg['save_dir'], probe_sample_imgs,
                       model, epoch, cfg)

    # ── Final summary ─────────────────────────────────────────────────────────
    _print_summary(history)
    np.save(os.path.join(cfg['save_dir'], 'history.npy'), history)
    torch.save(model.state_dict(), os.path.join(cfg['save_dir'], 'model_final.pt'))
    print(f'\nAll results saved to  {cfg["save_dir"]}/')


def _print_metrics(epoch, acc, loss, m):
    print(f'  train_loss          : {loss:.4f}')
    print(f'  train_acc           : {acc:.4f}')
    print(f'  mean_norm_sim_bias  : {m["mean_norm_sim_bias"]:+.4f}   '
          f'(+) = prototype bị kéo về HIGH-norm token (background)')
    print(f'  mean_bg_cosine      : {m["mean_bg_cosine"]:+.4f}   '
          f'(+) = prototype direction → background direction')
    print(f'  background_frac     : {m["bg_frac"]:.4f}   '
          f'(random baseline ≈ {CFG["highnorm_pct"]:.2f})')


def _print_summary(history):
    print('\n' + '=' * 80)
    print('  FINAL SUMMARY')
    print('=' * 80)
    hdr = (f"{'Epoch':>6}  {'TrainAcc':>9}  {'NormSimBias':>12}  "
           f"{'BgCosine':>10}  {'BgFrac':>8}  {'Verdict':}")
    print(hdr)
    print('-' * 80)

    baseline = CFG['highnorm_pct']
    for h in history:
        bias  = h['mean_norm_sim_bias']
        cos   = h['mean_bg_cosine']
        frac  = h['bg_frac']
        # Verdict đơn giản
        score = (1 if bias > 0 else 0) + (1 if cos > 0 else 0) + (1 if frac > baseline else 0)
        if score == 3:
            verdict = '⚠ ATTRACTED ← background'
        elif score >= 2:
            verdict = '~ Possibly attracted'
        else:
            verdict = '✓ Not attracted'
        print(f"{h['epoch']:>6}  {h['train_acc']:>9.4f}  {bias:>+12.4f}  "
              f"{cos:>+10.4f}  {frac:>8.4f}  {verdict}")
    print()
    # Kết luận dựa trên trend
    biases = [h['mean_norm_sim_bias'] for h in history if h['epoch'] > 0]
    if len(biases) >= 2:
        delta = biases[-1] - biases[0]
        print(f'  Δ norm_sim_bias (epoch 1 → {history[-1]["epoch"]}): {delta:+.4f}')
        if delta > 0.1:
            print('  → VERDICT: Prototype BỊ KÉO về phía high-norm (background) khi train.')
            print('    Hypothesis (b) được xác nhận.')
        elif delta < -0.1:
            print('  → VERDICT: Prototype KHÔNG bị kéo (hoặc ngược lại).')
            print('    Cần xem thêm visualization.')
        else:
            print('  → VERDICT: Tác động không rõ ràng. Xem visualization từng epoch.')
    print('=' * 80)


if __name__ == '__main__':
    main()
