import sys
sys.path.insert(0, '/root/CATCH/ts_benchmark/baselines/pre_train/submodules/DADA')

from transformers import PreTrainedModel
from .configuration_DADA import DADAConfig
from typing import Optional, Any, Tuple
from torch.autograd import Function
from torch.distributions.normal import Normal
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class DADA(PreTrainedModel):
    config_class = DADAConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Model(
            seq_len=config.seq_len[0],
            hidden_dim=config.hidden_dim[0],
            d_model=config.d_model[0],
            bn_dims=config.bn_dims[0],
            k=config.k[0],
            patch_len=config.patch_len[0],
            mask_mode=config.mask_mode[0],
            depth=config.depth[0],
            max_iters=config.max_iters[0],
        )

    def infer(self, x, anomaly_criterion=None, norm=0, L=1):
        out_copies = self.model.infer(x, norm=norm)
        return out_copies
        score = self.model.cal_anomaly_score(batch_x=x, batch_out_copies=out_copies, anomaly_criterion=anomaly_criterion, L=L)
        return score


class Model(nn.Module):
    def __init__(self, seq_len, hidden_dim, d_model, bn_dims, k, patch_len, mask_mode, depth, max_iters):
        super().__init__()
        win_size = seq_len
        hidden_dim = hidden_dim
        self.repr_dim = d_model
        depth = depth
        bn_dims = bn_dims
        k = k
        max_iters = max_iters

        self.patch_len = patch_len
        if win_size % self.patch_len:
            self.patch_num = (win_size // self.patch_len) + 1
        else:
            self.patch_num = win_size // self.patch_len

        self.mask_mode = mask_mode
        self.input_embed = nn.Linear(self.patch_len, hidden_dim)
        self.encoder = Encoder(
            patch_len=self.patch_len,
            patch_num=self.patch_num,
            output_dims=self.repr_dim,
            hidden_dims=hidden_dim,
            depth=depth,
            backbone="dilated_conv",
        )

        self.adaptive_bottleneck = AdaptiveBottleNeck(
            seq_len=self.patch_num,
            seq_dim=self.repr_dim,
            repr_dim=self.repr_dim,
            bn_dims=bn_dims,
            k=k,
        )

        self.decoder = MLP_Decoder(self.repr_dim, self.patch_len)
        self.adv_decoder = MLP_Decoder(self.repr_dim, self.patch_len)
        self.grl = WarmStartGradientReverseLayer(hi=0.5, max_iters=max_iters, auto_step=True)

    def infer(self, x, norm=0, mask_mode=None, copies=10):
        if mask_mode is None:
            mask_mode = self.mask_mode
        B, T, dims = x.size()

        if norm:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev

        x = x.permute(0, 2, 1)
        x = x.reshape(B*dims, T)

        if T % self.patch_len != 0:
            length = self.patch_num * self.patch_len
            padding = torch.zeros([B, (length - T)]).to(x.device)
            input = torch.cat([x, padding], dim=1)
        else:
            length = T
            input = x
        input_patch = input.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        input_patch = self.input_embed(input_patch)

        if mask_mode=="c":
            assert copies%2==0, "The number of copies of symmetric mask must be an even number"
            mask_1 = torch.from_numpy(np.random.binomial(1, 0.5, size=(B*dims*(copies//2), self.patch_num))).to(torch.bool)
            mask_2 = ~mask_1
            mask = torch.cat([mask_1, mask_2], dim=0)
            input_patch = input_patch.repeat(copies, 1, 1)
            input_patch[mask] = 0
        elif mask_mode=="random":
            mask = torch.from_numpy(np.random.binomial(1, 0.5, size=(B*dims*copies, self.patch_num))).to(torch.bool)
            input_patch = input_patch.repeat(copies, 1, 1)
            input_patch[mask] = 0
        elif mask_mode=="nomask":
            copies=1

        repr = self.encoder(input_patch)

        repr = torch.reshape(repr, (-1, self.patch_num*self.repr_dim))
        repr, balance_loss = self.adaptive_bottleneck(repr, repr)
        repr = torch.reshape(repr, (-1, self.patch_num, self.repr_dim))

        out = self.decoder(repr)
        out = out.reshape(copies, B*dims, T)
        out = out[:, :, :T].reshape(copies, B, dims, T)
        out = out.permute(0, 1, 3, 2)

        if norm:
            out = out * stdev.unsqueeze(dim=0).repeat(copies, 1, 1, 1) + means.unsqueeze(dim=0).repeat(copies, 1, 1, 1)

        return out

    def cal_anomaly_score(self, batch_x, batch_out_copies, anomaly_criterion=None, L=1):
        score = torch.var(batch_out_copies, dim=0)
        score = torch.mean(score, dim=-1)
        if anomaly_criterion is None:
            return score
        else:
            output_mean = torch.mean(batch_out_copies, dim=0)
            recon_score = anomaly_criterion(output_mean, batch_x)
            score = L*score+(1-L)*torch.mean(recon_score, dim=-1)
            return score


class Encoder(nn.Module):
    def __init__(self, patch_len, patch_num, output_dims=512, hidden_dims=64, depth=10, backbone="dilated_conv"):
        super().__init__()
        self.patch_len = patch_len
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.backbone = backbone
        if backbone=="dilated_conv":
            self.feature_extractor = DilatedConvEncoder(
                hidden_dims, [hidden_dims] * depth + [output_dims], kernel_size=3
            )
        self.repr_dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        repr = self.repr_dropout(self.feature_extractor(x))
        repr = repr.permute(0, 2, 1)
        return repr


class MLP_Decoder(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        hidden_dim = int(input_dims*2)
        self.net = nn.Sequential(
            nn.GELU(),
            nn.Linear(input_dims, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dims),
        )
        self.flatten = nn.Flatten(-2)
    def forward(self, x):
        x = self.net(x)
        x = self.flatten(x)
        return x


class Flatten_Decoder(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(-2),
            nn.Linear(input_dims, output_dims),
        )
    def forward(self, x):
        x = self.net(x)
        return x


class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class WarmStartGradientReverseLayer(nn.Module):
    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 1000., auto_step: Optional[bool] = False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        coeff = float(
            2.0 * (self.hi - self.lo) /
            (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        self.iter_num += 1


class BottleNeck(nn.Module):
    def __init__(self, seq_len, repr_dim, bn_dim):
        super(BottleNeck, self).__init__()
        self.seq_len = seq_len
        self.repr_dim = repr_dim
        self.net = nn.Sequential(
                        nn.Linear(repr_dim, bn_dim),
                        nn.GELU(),
                        nn.Linear(bn_dim, bn_dim),
                        nn.GELU(),
                        nn.Dropout(0.1),
                        nn.Linear(bn_dim, repr_dim),
                    )

    def forward(self, repr):
        res = repr
        repr = repr.reshape(-1, self.seq_len, self.repr_dim)
        repr = self.net(repr)
        repr = repr.reshape(-1, self.seq_len*self.repr_dim)
        repr = res + repr
        return repr


class AdaptiveBottleNeck(nn.Module):
    def __init__(self, seq_len, seq_dim, repr_dim, bn_dims=[8,16,32,64], noisy_gating=True, k=3):
        super(AdaptiveBottleNeck, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_bottlenecks = len(bn_dims)
        self.k = k
        self.bottlenecks = nn.ModuleList([
            BottleNeck(seq_len, repr_dim, bn_dim) for bn_dim in bn_dims
        ])
        self.w_gate = nn.Parameter(torch.zeros(seq_len*seq_dim, self.num_bottlenecks), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(seq_len*seq_dim, self.num_bottlenecks), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_bottlenecks)

    def cv_squared(self, x):
        eps = 1e-10

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)

        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_bottlenecks), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_bottlenecks and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, router_input, repr, loss_coef=1e-2):
        gates, load = self.noisy_top_k_gating(router_input, self.training)

        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_bottlenecks, gates)
        inputs = dispatcher.dispatch(repr)
        gates = dispatcher.expert_to_gates()
        outputs = [self.bottlenecks[i](inputs[i]) for i in range(self.num_bottlenecks)]
        repr = dispatcher.combine(outputs)
        return repr, loss


class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts

        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)

        _, self._expert_index = sorted_experts.split(1, dim=1)

        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]

        self._part_sizes = (gates > 0).sum(0).tolist()

        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)

        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None

    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i - 1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=1,
                final=(i == len(channels) - 1),
            )
            for i in range(len(channels))
        ])

    def forward(self, x):
        return self.net(x)
