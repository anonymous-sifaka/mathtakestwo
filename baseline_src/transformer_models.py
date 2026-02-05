import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import vit_b_16
from torchvision.models.vision_transformer import ViT_B_16_Weights


### SYMBOLIC DECODER ###

import torch
import torch.nn as nn
import torch.nn.functional as F

class SymbolicTransformerDecoder(nn.Module):
    def __init__(
        self,
        vision_input_dim=3200,
        in_dim=512,
        num_symbols=8,
        symbol_length=8,
        n_heads=4,  # Reduced from 8
        n_layers=2,  # Reduced from 4
        dropout=0.2  # Increased from 0.1
    ):
        super().__init__()

        self.embed_dim = in_dim
        self.num_symbols = num_symbols
        self.symbol_length = symbol_length

        # Project vision features to decoder input dimension
        self.vision_proj = nn.Linear(vision_input_dim, in_dim)

        # Learnable query embeddings and position encodings
        self.query_embed = nn.Parameter(torch.randn(symbol_length, in_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(symbol_length, in_dim) * 0.02)

        # Transformer decoder (stacked layers)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=in_dim, nhead=n_heads, dropout=dropout),
            num_layers=n_layers
        )

        # Final output projection to symbol logits
        self.fc_out = nn.Linear(in_dim, num_symbols)
        
        # Position-specific output heads to force diversity
        self.pos_outputs = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_dim, num_symbols)
            ) for _ in range(symbol_length)
        ])

    def forward(self, vision_feat, hard=False):
        """
        Args:
            vision_feat: Tensor of shape [B, vision_input_dim]
            hard: if True, return argmax symbols; otherwise, return softmax distribution
        Returns:
            Tensor of shape [B, symbol_length, num_symbols] if not hard,
            or [B, symbol_length] of discrete symbol indices if hard
        """
        B = vision_feat.size(0)

        # Project vision features to match decoder input dimension
        memory = self.vision_proj(vision_feat).unsqueeze(1).transpose(0, 1)  # [1, B, D]

        # Create position-specific queries with explicit position encoding
        queries = []
        for i in range(self.symbol_length):
            # Each position gets a unique combination of query + position embedding
            pos_query = self.query_embed[i] + self.pos_embed[i]  # [D]
            queries.append(pos_query)
        
        # Stack and expand for batch
        tgt = torch.stack(queries, dim=0).unsqueeze(1).expand(-1, B, -1)  # [L, B, D]

        # Run transformer decoder
        decoded = self.transformer_decoder(tgt=tgt, memory=memory)  # [L, B, D]
        
        # Use position-specific output heads
        logits_list = []
        for i in range(self.symbol_length):
            pos_logits = self.pos_outputs[i](decoded[i])  # [B, num_symbols]
            logits_list.append(pos_logits)
        
        logits = torch.stack(logits_list, dim=1)  # [B, L, num_symbols]
        
        if hard:
            return logits.argmax(dim=-1)  # [B, L] – discrete token indices
        else:
            
            if self.training:
                logits = logits + torch.randn_like(logits) * 0.1  # regularize during training

            decoded = F.softmax(logits, dim=-1)  # [B, L, num_symbols]
            return decoded


#### SYMBOLIC ENCODER ###
class SymbolicTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_symbols=8,
        embed_dim=512,
        bottleneck_shape=(128, 5, 5),
        symbol_length=8,
        n_heads=4,  # Reduced from 8
        n_layers=2,  # Reduced from 4
        dropout=0.2  # Increased from 0.1
    ):
        super().__init__()

        self.token_embed = nn.Embedding(num_symbols, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(symbol_length, embed_dim))

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dropout=dropout),
            num_layers=n_layers
        )

        self.bottleneck_shape = bottleneck_shape
        self.bottleneck_embed_dim = bottleneck_shape[0] * bottleneck_shape[1] * bottleneck_shape[2]

        # Combined dimension: [mean (D) + flattened (L * D)]
        self.combined_dim = embed_dim + symbol_length * embed_dim

        self.bottleneck_encoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.combined_dim, self.bottleneck_embed_dim),
        )

    def forward(self, symbols):
        """
        symbols: [B, L] for hard tokens or [B, L, K] for soft probabilities
        Returns: [B, C, H, W] where C*H*W = bottleneck_embed_dim
        """

        if symbols.dim() == 2:
            symbols = F.one_hot(symbols, num_classes=self.token_embed.num_embeddings).float()

        B, L, K = symbols.shape

        all_embeddings = self.token_embed.weight  # [K, D]
        x = torch.matmul(symbols, all_embeddings)  # [B, L, D]

        # Add positional encoding
        x = x + self.pos_embed.unsqueeze(0)  # [B, L, D]
        x = x.transpose(0, 1)  # [L, B, D]

        x_encoded = self.encoder(x)  # [L, B, D]

        # Hybrid representation

        pooled = x_encoded.mean(dim=0)                     # [B, D]
        flattened = x_encoded.transpose(0, 1).reshape(B, -1)  # [B, L * D]
        combined = torch.cat([pooled, flattened], dim=-1)  # [B, D + L*D]

        # Project to bottleneck and reshape
        out = self.bottleneck_encoder(combined)  # [B, bottleneck_embed_dim]
        out = out.view(B, *self.bottleneck_shape)

        return out



#### UNET PRE SYMBOLIC TRANSFORMER BOTTLENECK ###

class UNETPreTransformerBottleneck(nn.Module):
    def __init__(self, unet, bottleneck_shape=(32, 5, 5),
                 num_symbols=8, symbol_length=8):
        super().__init__()
        self.unet = unet
        
        for p in self.unet.parameters():
            p.requires_grad = False

        self.bottleneck_shape = bottleneck_shape
        self.bottleneck_decoder = SymbolicTransformerDecoder(
            in_dim=bottleneck_shape[0] * bottleneck_shape[1] * bottleneck_shape[2],
            num_symbols=num_symbols,
            symbol_length=symbol_length
        )

        self.encoder = SymbolicTransformerEncoder(
            num_symbols=num_symbols,
            symbol_length=symbol_length,
            embed_dim=128,
            bottleneck_shape=bottleneck_shape,
        )

    def forward(self, x, hard=False):
        # Encoder

        e1 = self.unet.enc1(x)
        e1 = self.unet.dropout1(e1)
        p1 = self.unet.pool1(e1)

        e2 = self.unet.enc2(p1)
        p2 = self.unet.pool2(e2)

        e3 = self.unet.enc3(p2)
        p3 = self.unet.pool3(e3)

        b = self.unet.bottleneck(p3)  # Now [B, 32, 5, 5]

        # Symbolic Encoding

        b_flat = b.view(b.size(0), -1)
        sym_b = self.bottleneck_decoder(b_flat, hard=hard)

        # Decode symbols
        b_decoded = self.encoder(sym_b)

        # Decoder path
        up3 = F.interpolate(self.unet.up3(b_decoded), size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.unet.dec3(up3)

        up2 = F.interpolate(self.unet.up2(d3), size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.unet.dec2(up2)

        up1 = F.interpolate(self.unet.up1(d2), size=x.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.unet.dec1(up1)

        return self.unet.final(d1), sym_b

    def recon_from_symbols(self, sym_b, hard=False):
        b_decoded, e2_decoded = self.encoder(sym_b)

        up3 = F.interpolate(self.unet.up3(b_decoded), size=[11, 10], mode='bilinear', align_corners=False)
        d3 = self.unet.dec3(up3)

        up2 = F.interpolate(self.unet.up2(d3), size=[11, 10], mode='bilinear', align_corners=False)
        d2 = self.unet.dec2(up2)

        up1 = F.interpolate(self.unet.up1(d2), size=[47, 41], mode='bilinear', align_corners=False)
        d1 = self.unet.dec1(up1)

        return self.unet.final(d1)



### VISION ENCODER ###

class VisionEncoder(nn.Module):
    def __init__(self, embed_dim=512, freeze=True):
        super().__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)  # Or replace with TinyViT
        self.vit.heads = nn.Identity()  # Remove classification head
        self.project = nn.Linear(768, embed_dim)  # Project to lower dim if needed

        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False

    def forward(self, x):
        feats = self.vit(x)  # [B, 768]
        return self.project(feats)  # [B, embed_dim]



### IMAGE COMPARISON HEAD ###
class ListenerImageSelector(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.img_encoder = VisionEncoder(embed_dim=embed_dim, freeze=True)

    def forward(self, msg_embed, candidate_imgs):
        B, N, C, H, W = candidate_imgs.shape
        imgs = candidate_imgs.view(B * N, C, H, W)
        img_feats = self.img_encoder(imgs).view(B, N, -1)  # [B, N, D]
        msg_embed = F.normalize(msg_embed, dim=1).unsqueeze(1)  # [B, 1, D]
        img_feats = F.normalize(img_feats, dim=2)  # [B, N, D]
        sim_scores = (msg_embed * img_feats).sum(dim=2)  # [B, N]
        return sim_scores

### IMAGE RECONSTRUCTOR ###
class ImageReconstructor(nn.Module):
    def __init__(self, embed_dim=512, num_symbols=8, seq_len=8, img_size=224):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size
        
        # Decode symbolic representation back to image features
        self.symbolic_decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, 768)  # Match ViT feature dimension
        )
        
        # Reconstruct image from ViT features using transposed convolutions
        self.image_decoder = nn.Sequential(
            # Start with 768 features, reshape to spatial features
            nn.Linear(768, 14 * 14 * 768),  # ViT patch size is 16x16, so 224/16 = 14
            nn.ReLU(),
            nn.Unflatten(1, (768, 14, 14)),  # [B, 768, 14, 14]
            
            # Upsample to full image size
            nn.ConvTranspose2d(768, 512, kernel_size=4, stride=2, padding=1),  # 14 -> 28
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 28 -> 56
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 56 -> 112
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 112 -> 224
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 3, kernel_size=3, padding=1),  # Final RGB output
            nn.Tanh()  # Output in [-1, 1] range
        )

    def forward(self, symbolic_embed):
        # symbolic_embed: [B, embed_dim] - output from symbolic encoder
        # Decode to ViT features
        vit_features = self.symbolic_decoder(symbolic_embed)  # [B, 768]
        
        # Reconstruct image
        reconstructed = self.image_decoder(vit_features)  # [B, 3, 224, 224]
        
        return reconstructed

### RECONSTRUCTION MODEL ###
class MLLMReconstructionModel(nn.Module):
    def __init__(self, embed_dim=512, num_symbols=8, seq_len=8, img_size=224):
        super().__init__()
        self.vision_encoder = VisionEncoder(embed_dim=embed_dim)
        self.symbolic_decoder = SymbolicTransformerDecoder(embed_dim, num_symbols, seq_len)
        self.symbolic_encoder = SymbolicTransformerEncoder(num_symbols, embed_dim, seq_len)
        self.image_reconstructor = ImageReconstructor(embed_dim, num_symbols, seq_len, img_size)

    def forward(self, image):
        # Encode image to symbolic representation
        vision_feat = self.vision_encoder(image)
        symbol_logits = self.symbolic_decoder(vision_feat)
        symbols = symbol_logits.argmax(dim=-1)  # [B, L]
        
        # Encode symbolic message
        msg_embed = self.symbolic_encoder(symbols)  # [B, D]
        
        # Reconstruct image from symbolic representation
        reconstructed = self.image_reconstructor(msg_embed)  # [B, 3, 224, 224]
        
        return reconstructed, symbols, msg_embed
