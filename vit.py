import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size // patch_size for _ in range(2)]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x, mask_weight=None):
        # Self-Attention
        x_norm = self.norm1(x)
        if mask_weight is not None:
            B, seq_len, _ = mask_weight.shape
            mask_weight_expanded = mask_weight.unsqueeze(1).repeat(1, self.attn.num_heads, 1, 1).view(B * self.attn.num_heads, seq_len, seq_len)
        else:
            mask_weight_expanded = None
        attn_out = self.attn(x_norm, x_norm, x_norm, attn_mask=mask_weight_expanded)[0]
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x

class ViTEncoder(nn.Module):
    def __init__(self, depth=12, dim=768, num_heads=12):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask_weight=None):
        for blk in self.blocks:
            x = blk(x, mask_weight)
        return self.norm(x)

class MaskedViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=4, embed_dim=768, depth=12, num_heads=12, alpha=0.5):
        super().__init__()
        self.patch_embed_img = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.patch_embed_mask = PatchEmbed(img_size, patch_size, 1, embed_dim)
        num_patches = self.patch_embed_img.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = ViTEncoder(depth, embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.alpha = alpha

    def forward(self, img, mask):
        B = img.shape[0]
        tokens_img = self.patch_embed_img(img)
        tokens_mask = self.patch_embed_mask(mask.unsqueeze(1))
        tokens = tokens_img + self.alpha * tokens_mask
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, tokens), dim=1)
        x = x + self.pos_embed
        # 计算mask_weight
        seq_len = x.shape[1]
        mask_weight = torch.zeros(B, seq_len, seq_len, device=x.device, dtype=x.dtype)
        mask_weight[:, 1:, 1:] = torch.einsum('b i d, b j d -> b i j', tokens_mask, tokens_mask) / (self.patch_embed_img.embed_dim ** 0.5)
        x = self.blocks(x, mask_weight)
        x = self.norm(x)
        x = self.head(x[:, 0])
        return x