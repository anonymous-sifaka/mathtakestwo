import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


#### UNET IMAGE COMPRESSION MODEL ####

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetCompressor(nn.Module):
    def __init__(self):
        super(UNetCompressor, self).__init__()

        def block(in_channels, out_channels, num_convs=4):
            layers = []
            for i in range(num_convs):
                conv_in = in_channels if i == 0 else out_channels
                layers.append(nn.Conv2d(conv_in, out_channels, 3, padding=1))
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        # --- Encoder ---
        self.enc1 = block(1, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = block(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = block(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = block(128, 128)

        # --- Decoder ---
        self.up3 = nn.Conv2d(128, 128, kernel_size=1)
        self.dec3 = block(128, 64)

        self.up2 = nn.Conv2d(64, 64, kernel_size=1)
        self.dec2 = block(64 + 64, 32)

        self.up1 = nn.Conv2d(32, 32, kernel_size=1)
        self.dec1 = block(32, 16)

        self.final = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)

        # Decoder
        up3 = F.interpolate(self.up3(b), size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(up3)

        up2 = F.interpolate(self.up2(d3), size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([up2, e2], dim=1))

        up1 = F.interpolate(self.up1(d2), size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(up1)

        return self.final(d1)
    
    
##### MODELS FOR SYMBOLIC COMPRESSION ####

class GumbelSymbolEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_symbols=8, symbol_length=4, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.symbol_length = symbol_length
        self.num_symbols = num_symbols

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_symbols * symbol_length)
        )

    def forward(self, x, hard=False):
        logits = self.encoder(x)  # [B, K * L]
        logits = logits.view(-1, self.symbol_length, self.num_symbols)  # [B, L, K]
        symbols = F.gumbel_softmax(logits, tau=self.temperature, hard=hard)
        return symbols  # [B, L, K]

class SymbolToImageDecoder(nn.Module):
    
    def __init__(self, num_symbols=8, symbol_length=8, embed_dim=128,
                 bottleneck_shape=(128, 5, 5), skip_shape=(64, 11, 10)):
        super().__init__()

        self.bottleneck_shape = bottleneck_shape
        self.skip_shape = skip_shape

        # Project discrete symbols to dense embedding
        self.embed = nn.Linear(num_symbols, embed_dim)

        # Bottleneck decoder
        self.bottleneck_decoder = nn.Sequential(
            nn.Linear(1536, 4096),
            nn.ReLU(),
            nn.Linear(4096, bottleneck_shape[0] * bottleneck_shape[1] * bottleneck_shape[2])
        )

        # Skip feature decoder
        self.skip_decoder = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Linear(4096, skip_shape[0] * skip_shape[1] * skip_shape[2])
        )

    def forward(self, sym_b, sym_skip):
        b = self.embed(sym_b).view(sym_b.size(0), -1)       # [B, L*D]
        e2 = self.embed(sym_skip).view(sym_skip.size(0), -1)
        
        b_feats = self.bottleneck_decoder(b).view(-1, *self.bottleneck_shape)
        e2_feats = self.skip_decoder(e2).view(-1, *self.skip_shape)

        return b_feats, e2_feats
    
class UNetWithSymbolicBottleneck(nn.Module):
    def __init__(self, unet, bottleneck_shape=(256, 5, 5), skip_shape=(256, 11, 10),
                 num_symbols=8, symbol_length=8):
        super().__init__()
        self.unet = unet
        for p in self.unet.parameters():
            p.requires_grad = False

        self.bottleneck_shape = bottleneck_shape
        self.skip_shape = skip_shape

        self.bottleneck_encoder = GumbelSymbolEncoder(
            in_dim=bottleneck_shape[0] * bottleneck_shape[1] * bottleneck_shape[2],
            num_symbols=num_symbols,
            symbol_length=6
        )

        self.skip_encoder = GumbelSymbolEncoder(
            in_dim=skip_shape[0] * skip_shape[1] * skip_shape[2],
            num_symbols=num_symbols,
            symbol_length=2
        )

        self.decoder = SymbolToImageDecoder(
            num_symbols=num_symbols,
            symbol_length=symbol_length,  # bottleneck + skip
            embed_dim=256,
        )

    def forward(self, x, hard=False):
        # Pass through encoder
        
        e1 = self.unet.enc1(x)
        p1 = self.unet.pool1(e1)

        e2 = self.unet.enc2(p1)
        p2 = self.unet.pool2(e2)

        e3 = self.unet.enc3(p2)
        p3 = self.unet.pool3(e3)

        b = self.unet.bottleneck(p3)  # [B, 128, 5, 5]

        # Extract and encode symbolic representations
        
        b_flat = b.view(b.size(0), -1)
        e2_resized = F.adaptive_avg_pool2d(e2, self.skip_shape[1:])
        e2_flat = e2_resized.view(e2_resized.size(0), -1)

        sym_b = self.bottleneck_encoder(b_flat, hard=hard)
        sym_skip = self.skip_encoder(e2_flat, hard=hard)

        # Decode symbols back into feature maps
        
        b_decoded, e2_decoded = self.decoder(sym_b, sym_skip)

        # --- Decoder path from b_decoded and e2_decoded ---
        
        up3 = F.interpolate(self.unet.up3(b_decoded), size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.unet.dec3(up3)

        up2 = F.interpolate(self.unet.up2(d3), size=e2_decoded.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.unet.dec2(torch.cat([up2, e2_decoded], dim=1))

        up1 = F.interpolate(self.unet.up1(d2), size=x.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.unet.dec1(up1)

        out = self.unet.final(d1)
        
        #print(e3.shape[2:]) - use to define size for recon function if changing input image size
        #print(e2_decoded.shape[2:])
        #print(x.shape[2:])
        
        return out, sym_b, sym_skip
    
    
    def recon_from_symbols(self, sym_b, sym_skip, hard=False):

        # Decode symbols back into feature maps
        
        b_decoded, e2_decoded = self.decoder(sym_b, sym_skip)

        # --- Decoder path from b_decoded and e2_decoded ---
        
        up3 = F.interpolate(self.unet.up3(b_decoded), size=[11, 10], mode='bilinear', align_corners=False)
        d3 = self.unet.dec3(up3)

        up2 = F.interpolate(self.unet.up2(d3), size=[11, 10], mode='bilinear', align_corners=False)
        d2 = self.unet.dec2(torch.cat([up2, e2_decoded], dim=1))

        up1 = F.interpolate(self.unet.up1(d2), size=[47, 41], mode='bilinear', align_corners=False)
        d1 = self.unet.dec1(up1)

        out = self.unet.final(d1)
        return out


#### MODELS FOR INPUT vs. QUESTIONS SIMILARITY SEARCH ####

# --- Dataset from previous step ---
class SimilarityDataset(torch.utils.data.Dataset):
    def __init__(self, example_generator, num_samples=10000):
        self.generator = example_generator
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img, answer, questions = self.generator.get_qna()
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        questions_tensor = torch.tensor(questions, dtype=torch.float32)
        answer_tensor = torch.tensor(answer.argmax(), dtype=torch.long)
        return img_tensor, questions_tensor, answer_tensor

# --- Simple CNN Encoder ---
class ImageEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # -> [32, 23, 20]
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),   # -> [64, 4, 4]
        )
        self.fc = nn.Linear(64 * 4 * 4, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)  # [B, latent_dim]

# --- Similarity Model ---
class SimilarityModel(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = ImageEncoder(latent_dim)

    def forward(self, img, questions):
        # img: [B, 1, H, W]
        # questions: [B, 4, 1, H, W]
        B = img.shape[0]

        img_enc = self.encoder(img)               # [B, D]
        q_enc = self.encoder(questions.view(-1, 1, 47, 41))  # [B*4, D]
        q_enc = q_enc.view(B, 4, -1)              # [B, 4, D]

        # Normalize for cosine similarity
        img_enc = F.normalize(img_enc, dim=1).unsqueeze(1)   # [B, 1, D]
        q_enc = F.normalize(q_enc, dim=2)                    # [B, 4, D]

        scores = (img_enc * q_enc).sum(dim=2)     # [B, 4] cosine similarity
        return scores
    
    