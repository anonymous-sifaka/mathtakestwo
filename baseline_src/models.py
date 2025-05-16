import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

#### UNET IMAGE COMPRESSION MODEL ####

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class UNetCompressor(nn.Module):
    def __init__(self, bottleneck_dim=32, dropout_rate=0.05):
        super().__init__()

        # --- Encoder ---
        self.enc1 = ResidualBlock(1, 32)
        self.dropout1 = nn.Dropout2d(dropout_rate)  # Dropout after first layer
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ResidualBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ResidualBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, bottleneck_dim, 3, padding=1),
            nn.BatchNorm2d(bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
        )

        # --- Decoder ---
        self.up3 = nn.Conv2d(bottleneck_dim, 128, 1)
        self.dec3 = ResidualBlock(128, 64)

        self.up2 = nn.Conv2d(64, 64, 1)
        self.dec2 = ResidualBlock(64, 32)

        self.up1 = nn.Conv2d(32, 32, 1)
        self.dec1 = ResidualBlock(32, 32)

        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e1 = self.dropout1(e1)
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
        d2 = self.dec2(up2)

        up1 = F.interpolate(self.up1(d2), size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(up1)

        return self.final(d1)


##### MODELS FOR UNET PRETRAINED SYMBOLIC COMPRESSION ####

class GumbelSymbolEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=1024, num_symbols=8, symbol_length=4, temperature=0.5):
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

        if hard:
            # Use deterministic mode: take top-k instead of sampling
            symbols = F.one_hot(logits.argmax(dim=-1), num_classes=self.num_symbols).float()
        else:
            symbols = F.gumbel_softmax(logits, tau=self.temperature, hard=hard)

        return symbols  # [B, L, K]


class SymbolToImageDecoder(nn.Module):

    def __init__(self, num_symbols=8, symbol_length=8, embed_dim=128,
                 bottleneck_shape=(32, 5, 5)):
        super().__init__()

        self.bottleneck_shape = bottleneck_shape

        # Project discrete symbols to dense embedding
        self.embed = nn.Linear(num_symbols, embed_dim)

        # Bottleneck decoder
        self.bottleneck_decoder = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, bottleneck_shape[0] * bottleneck_shape[1] * bottleneck_shape[2])
        )

    def forward(self, sym_b):
        b = self.embed(sym_b).view(sym_b.size(0), -1)  # [B, L*D]
        b_feats = self.bottleneck_decoder(b).view(-1, *self.bottleneck_shape)

        return b_feats


class UNETPreSymbolicBottleneck(nn.Module):
    def __init__(self, unet, bottleneck_shape=(32, 5, 5),
                 num_symbols=8, symbol_length=8):
        super().__init__()
        self.unet = unet
        for p in self.unet.parameters():
            p.requires_grad = False

        self.bottleneck_shape = bottleneck_shape
        self.bottleneck_encoder = GumbelSymbolEncoder(
            in_dim=bottleneck_shape[0] * bottleneck_shape[1] * bottleneck_shape[2],
            num_symbols=num_symbols,
            symbol_length=symbol_length
        )

        self.decoder = SymbolToImageDecoder(
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
        sym_b = self.bottleneck_encoder(b_flat, hard=hard)

        # Decode symbols
        b_decoded = self.decoder(sym_b)

        # Decoder path
        up3 = F.interpolate(self.unet.up3(b_decoded), size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.unet.dec3(up3)

        up2 = F.interpolate(self.unet.up2(d3), size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.unet.dec2(up2)

        up1 = F.interpolate(self.unet.up1(d2), size=x.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.unet.dec1(up1)

        return self.unet.final(d1), sym_b

    def recon_from_symbols(self, sym_b, hard=False):
        b_decoded, e2_decoded = self.decoder(sym_b)

        up3 = F.interpolate(self.unet.up3(b_decoded), size=[11, 10], mode='bilinear', align_corners=False)
        d3 = self.unet.dec3(up3)

        up2 = F.interpolate(self.unet.up2(d3), size=[11, 10], mode='bilinear', align_corners=False)
        d2 = self.unet.dec2(up2)

        up1 = F.interpolate(self.unet.up1(d2), size=[47, 41], mode='bilinear', align_corners=False)
        d1 = self.unet.dec1(up1)

        return self.unet.final(d1)


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
            nn.AdaptiveAvgPool2d((4, 4)),  # -> [64, 4, 4]
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

        img_enc = self.encoder(img)  # [B, D]
        q_enc = self.encoder(questions.view(-1, 1, 47, 41))  # [B*4, D]
        q_enc = q_enc.view(B, 4, -1)  # [B, 4, D]

        # Normalize for cosine similarity
        img_enc = F.normalize(img_enc, dim=1).unsqueeze(1)  # [B, 1, D]
        q_enc = F.normalize(q_enc, dim=2)  # [B, 4, D]

        scores = (img_enc * q_enc).sum(dim=2)  # [B, 4] cosine similarity
        return scores


##### BASELINE SYMBOLIC COMPRESSION MODELS #####

class DirectGumbelSymbolEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=64, num_symbols=8, symbol_length=8, temperature=0.5,
                 dropout_rate=0.05):
        """
        Args:
            in_channels (int): Number of input channels (e.g., 1 for grayscale).
            hidden_dim (int): Dimensionality of intermediate CNN features.
            num_symbols (int): Size of the discrete vocabulary (K).
            symbol_length (int): Number of symbols to output (L).
            temperature (float): Gumbel-softmax temperature.
        """
        super().__init__()
        self.temperature = temperature
        self.num_symbols = num_symbols
        self.symbol_length = symbol_length

        self.encoder = nn.Sequential(
            nn.Dropout2d(dropout_rate),  # Dropout
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, symbol_length)),  # Output shape: [B, hidden_dim, 1, L]
            nn.Flatten(start_dim=2)  # Shape: [B, hidden_dim, L]
        )

        self.to_logits = nn.Conv1d(hidden_dim, num_symbols, kernel_size=1)  # [B, K, L]

    def forward(self, x, hard=False):
        """
        Args:
            x (Tensor): Input tensor of shape [B, C, H, W]
            hard (bool): Whether to sample hard one-hot vectors (straight-through).

        Returns:
            Tensor: One-hot (soft or hard) symbols of shape [B, L, K]
        """
        features = self.encoder(x)  # [B, hidden_dim, L]
        logits = self.to_logits(features)  # [B, num_symbols, L]

        if hard:
            symbols = F.one_hot(logits.permute(0, 2, 1).argmax(dim=-1), num_classes=self.num_symbols).float()
        else:
            symbols = F.gumbel_softmax(logits.permute(0, 2, 1), tau=self.temperature, hard=hard)

        return symbols  # Shape: [B, L, K]


# ----- Decoder: symbols -> image -----
class DirectSymbolToImageDecoder(nn.Module):
    def __init__(self, num_symbols=8, symbol_length=8, embed_dim=64, output_shape=(1, 47, 41)):
        super().__init__()
        self.embed = nn.Linear(num_symbols, embed_dim)
        self.decoder = nn.Sequential(
            nn.Linear(symbol_length * embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_shape[0] * output_shape[1] * output_shape[2]),
            nn.Sigmoid()
        )
        self.output_shape = output_shape

    def forward(self, symbols):
        x = self.embed(symbols)  # [B, L, D]
        x = x.view(x.size(0), -1)  # flatten
        x = self.decoder(x)  # [B, C*H*W]
        x = x.view(-1, *self.output_shape)  # reshape
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F


class SymbolicAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, hard=False):
        """
        Args:
            x (Tensor): Input image tensor of shape [B, C, H, W]
            hard (bool): Whether to use hard (discrete) Gumbel-Softmax sampling

        Returns:
            x_recon (Tensor): Reconstructed image
            symbols (Tensor): Symbolic representation [B, L, K]
        """
        symbols = self.encoder(x, hard=hard)
        x_recon = self.decoder(symbols)
        return x_recon, symbols



