import torch
from torch import nn
from torch.nn import functional as F

from einops import rearrange
from transformers.modeling_utils import PreTrainedModel

from .configuration_vqgan import VQGANConfig


def nonlinearity(x):
    return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv

        if with_conv:
            self.conv = nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, hidden_states):
        hidden_states = F.interpolate(hidden_states, scale_factor=2, mode="nearest")
        if self.with_conv:
            hidden_states = self.conv(hidden_states)
        return hidden_states


class Downsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv

        if with_conv:
            self.conv = nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, hidden_states):
        if self.with_conv:
            hidden_states = F.pad(hidden_states, (0, 1, 0, 1), mode="constant", value=0)
            hidden_states = self.conv(hidden_states)
        else:
            hidden_states = F.avg_pool2d(hidden_states, kernel_size=2, stride=2)
        return hidden_states


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        use_conv_shortcut: bool = False,
        temb_channels: int = 512,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = use_conv_shortcut

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6)
        self.dropout = nn.Dropout(dropout_prob)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if in_channels != out_channels:
            if use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, hidden_states, temb=None):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            hidden_states = (
                hidden_states + self.temb_proj(nonlinearity(temb))[:, :, None, None]
            )

        hidden_states = self.norm2(hidden_states)
        hidden_states = nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                residual = self.conv_shortcut(residual)
            else:
                residual = self.nin_shortcut(residual)

        return hidden_states + residual


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm(hidden_states)

        query = self.q(hidden_states)
        key = self.k(hidden_states)
        value = self.v(hidden_states)

        # compute attention
        batch, channels, height, width = query.shape
        query = query.reshape(batch, channels, height * width)
        query = query.permute(0, 2, 1)
        key = key.reshape(batch, channels, height * width)
        attn_weights = torch.bmm(query, key)
        attn_weights = attn_weights * (int(channels) ** -0.5)
        attn_weights = F.softmax(attn_weights, dim=2)

        # attend to values
        value = value.reshape(batch, channels, height * width)
        hidden_states = torch.bmm(value, attn_weights)
        hidden_states = hidden_states.reshape(batch, channels, height, width)

        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states


class UpsamplingBlock(nn.Module):
    def __init__(self, config: VQGANConfig, curr_res: int, block_idx: int):
        super().__init__()
        self.config = config
        self.curr_res = curr_res
        self.block_idx = block_idx

        if self.block_idx == self.config.num_resolutions - 1:
            block_in = self.config.ch * self.config.ch_mult[-1]
        else:
            block_in = self.config.ch * self.config.ch_mult[self.block_idx + 1]

        block_out = self.config.ch * self.config.ch_mult[self.block_idx]
        self.temb_ch = 0

        res_blocks = []
        attn_blocks = []
        for _ in range(self.config.num_res_blocks):
            res_blocks.append(
                ResnetBlock(
                    block_in,
                    block_out,
                    temb_channels=self.temb_ch,
                    dropout_prob=self.config.dropout,
                )
            )
            block_in = block_out
            if self.curr_res in self.config.attn_resolutions:
                attn_blocks.append(AttnBlock(block_in))

        self.blocks = nn.ModuleList(res_blocks)
        self.attn = nn.ModuleList(attn_blocks)

        self.upsample = None
        if self.block_idx != 0:
            self.upsample = Upsample(block_in, self.config.resamp_with_conv)

    def forward(self, hidden_states, temb=None):
        for i, res_block in enumerate(self.blocks):
            hidden_states = res_block(hidden_states, temb)
            if self.attn:
                hidden_states = self.attn[i](hidden_states)

        if self.upsample:
            hidden_states = self.upsample(hidden_states)

        return hidden_states


class DownsamplingBlock(nn.Module):
    def __init__(self, config: VQGANConfig, curr_res: int, block_idx: int):
        super().__init__()
        self.config = config
        self.curr_res = curr_res
        self.block_idx = block_idx

        in_ch_mult = (1,) + tuple(self.config.ch_mult)
        block_in = self.config.ch * in_ch_mult[self.block_idx]
        block_out = self.config.ch * self.config.ch_mult[self.block_idx]
        self.temb_ch = 0

        res_blocks = []
        attn_blocks = []
        for _ in range(self.config.num_res_blocks):
            res_blocks.append(
                ResnetBlock(
                    block_in,
                    block_out,
                    temb_channels=self.temb_ch,
                    dropout_prob=self.config.dropout,
                )
            )
            block_in = block_out
            if self.curr_res in self.config.attn_resolutions:
                attn_blocks.append(AttnBlock(block_in))

        self.blocks = nn.ModuleList(res_blocks)
        self.attn = nn.ModuleList(attn_blocks)

        self.downsample = None
        if self.block_idx != self.config.num_resolutions - 1:
            self.downsample = Downsample(block_in, self.config.resamp_with_conv)

    def forward(self, hidden_states, temb=None):
        for i, res_block in enumerate(self.blocks):
            hidden_states = res_block(hidden_states, temb)
            if self.attn:
                hidden_states = self.attn[i](hidden_states)

        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states)

        return hidden_states


class MidBlock(nn.Module):
    def __init__(self, in_channels: int, temb_channels: int, dropout_prob: float):
        super().__init__()
        self.block_1 = ResnetBlock(
            in_channels,
            in_channels,
            temb_channels=temb_channels,
            dropout_prob=dropout_prob,
        )
        self.attn_1 = AttnBlock(in_channels)
        self.block_2 = ResnetBlock(
            in_channels,
            in_channels,
            temb_channels=temb_channels,
            dropout_prob=dropout_prob,
        )

    def forward(self, hidden_states, temb=None):
        hidden_states = self.block_1(hidden_states, temb)
        hidden_states = self.attn_1(hidden_states)
        hidden_states = self.block_2(hidden_states, temb)
        return hidden_states


class VQGANEncoder(nn.Module):
    def __init__(self, config: VQGANConfig):
        super().__init__()
        self.config = config
        self.temb_ch = 0

        # downsampling
        self.conv_in = nn.Conv2d(
            self.config.in_channels, self.config.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = self.config.resolution
        downsample_blocks = []
        for i_level in range(self.config.num_resolutions):
            downsample_blocks.append(DownsamplingBlock(self.config, curr_res, i_level))
            if i_level != self.config.num_resolutions - 1:
                curr_res = curr_res // 2
        self.down = nn.ModuleList(downsample_blocks)

        # middle
        mid_channels = self.config.ch * self.config.ch_mult[-1]
        self.mid = MidBlock(mid_channels, self.temb_ch, self.config.dropout)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=mid_channels, eps=1e-6)
        self.conv_out = nn.Conv2d(
            mid_channels,
            2 * self.config.z_channels
            if self.config.double_z
            else self.config.z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, pixel_values):
        temb = None

        # downsampling
        hidden_states = self.conv_in(pixel_values)
        for block in self.down:
            hidden_states = block(hidden_states, temb)

        # middle
        hidden_states = self.mid(hidden_states, temb)

        # end
        hidden_states = self.norm_out(hidden_states)
        hidden_states = nonlinearity(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states


class VQGANDecoder(nn.Module):
    def __init__(self, config: VQGANConfig):
        super().__init__()
        self.config = config
        self.temb_ch = 0

        block_in = self.config.ch * self.config.ch_mult[self.config.num_resolutions - 1]
        block_out = self.config.ch * self.config.ch_mult[0]
        curr_res = self.config.resolution // 2 ** (self.config.num_resolutions - 1)
        self.z_shape = (1, self.config.z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = nn.Conv2d(
            self.config.z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        mid_channels = self.config.ch * self.config.ch_mult[-1]
        self.mid = MidBlock(mid_channels, self.temb_ch, self.config.dropout)

        # upsampling
        upsample_blocks = []
        for i_level in reversed(range(self.config.num_resolutions)):
            upsample_blocks.append(UpsamplingBlock(self.config, curr_res, i_level))
            if i_level != 0:
                curr_res = curr_res * 2
        self.up = nn.ModuleList(reversed(upsample_blocks))

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_out, eps=1e-6)
        self.conv_out = nn.Conv2d(
            block_out, self.config.out_ch, kernel_size=3, stride=1, padding=1
        )

    def forward(self, hidden_states):
        temb = None

        # z to block_in
        hidden_states = self.conv_in(hidden_states)

        # middle
        hidden_states = self.mid(hidden_states, temb)

        # upsampling
        for block in reversed(self.up):
            hidden_states = block(hidden_states, temb)

        # end
        if self.config.give_pre_end:
            return hidden_states

        # end
        hidden_states = self.norm_out(hidden_states)
        hidden_states = nonlinearity(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states


class VectorQuantizer(nn.Module):
    def __init__(self, config: VQGANConfig):
        super().__init__()
        self.config = config
        self.e_dim = config.embed_dim

        self.embedding = nn.Embedding(self.config.n_embed, self.config.embed_dim)
        self.embedding.weight.data.uniform_(
            -1 / self.config.n_embed, 1 / self.config.n_embed
        )

    def forward(self, z):
        z = rearrange(z, "b c h w -> b h w c").contiguous()
        z_flattened = z.view(-1, self.e_dim)

        d = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2
            * torch.einsum(
                "bd,dn->bn", z_flattened, rearrange(self.embedding.weight, "n d -> d n")
            )
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        loss = torch.mean((z_q.detach() - z) ** 2) + 0.5 * torch.mean(
            (z_q - z.detach()) ** 2
        )

        z_q = z + (z_q - z).detach()

        z_q = rearrange(z_q, "b h w c -> b c h w").contiguous()

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class VQGANPreTrainedModel(PreTrainedModel):
    def __init__(self, config: VQGANConfig):
        super().__init__(config)
        self.config = config

        self.encoder = VQGANEncoder(config)
        self.decoder = VQGANDecoder(config)
        self.quantize = VectorQuantizer(config)
        self.quant_conv = nn.Conv2d(self.config.z_channels, self.config.embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(
            self.config.embed_dim, self.config.z_channels, 1
        )

    def encode(self, pixel_values):
        hidden_states = self.encoder(pixel_values)
        hidden_states = self.quant_conv(hidden_states)
        quant_states, emb_loss, info = self.quantize(hidden_states)
        return quant_states, emb_loss, info

    def decode(self, quant_states):
        hidden_states = self.post_quant_conv(quant_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        hidden_states = self.decode(quant_b)
        return hidden_states

    def forward(self, pixel_values):
        quant_states, emb_loss, _ = self.encode(pixel_values)
        hidden_states = self.decode(quant_states)
        return hidden_states, emb_loss
