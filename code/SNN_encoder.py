import torch
import snntorch as snn
from torch import einsum, nn
from snntorch import surrogate
import torch.nn.functional as F
from einops import rearrange, repeat
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

batch_size = 1
num_inputs = 260
num_inputs_img = 256
num_hidden = 5
num_outputs = 10
num_steps = batch_size
beta = 0.95
spike_grad = surrogate.fast_sigmoid(slope=25)


# SNN_series encoder
class SNN_series(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
            output = torch.stack(spk2_rec, dim=0)

        return output



# SNN_series encoder
class SNN_img(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.conv1 = nn.Conv2d(3, 12, 1)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(12, 32, 1)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc1 = nn.Linear(2097152, 512)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.conv3 = nn.Conv2d(3, 32, 1)

    def forward(self, x):
        img = []
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        for step in range(num_steps):
            cur1 = self.conv1(x)
            cur1 = F.max_pool2d(cur1, 1)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = F.max_pool2d(self.conv2(spk1), 1)
            out_resnet, mem2 = self.lif2(cur2, mem2)


            img.append(out_resnet)
            output = torch.stack(img, dim=0)

            output_conv = self.conv3(x)
            first_img = rearrange(output_conv, 'b n j d-> 1 b n j d')

            output = first_img + output

        return output

# SNN transformer
class Pulse_extraction(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True)

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)

    def forward(self, x, res_attn):

        T,B,C,H,W = x.shape
        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)
        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T,B,C,N).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q=q_conv_out



        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T,B,C,N).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k=k_conv_out


        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T,B,C,N).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v=v_conv_out



        x = k.transpose(-2,-1) @ v
        x = (q @ x) * self.scale
        x = self.attn_lif(x)
        x = x.flatten(0,1)
        x = self.proj_lif(self.proj_bn(self.proj_conv(x)).reshape(T,B,C,H,W))
        x = self.proj_lif(x.reshape(T, B, C, H, W))
        return x, v



class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., sr_ratio=1):
        super().__init__()

        self.attn = Pulse_extraction(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        mlp_hidden_dim = int(dim * mlp_ratio)


    def forward(self, x, res_attn):
        x_attn, attn = (self.attn(x, res_attn))
        x = x + x_attn
        return x, attn


class Pulse_transformation(nn.Module):
    def __init__(self):
        super().__init__()


        self.fc1 = nn.Linear(num_inputs_img, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
            output = torch.stack(spk2_rec, dim=0)
        return output


class SNN_pulse_supplementation(nn.Module):
    def __init__(self, embed_dims=3, num_heads=4, mlp_ratios= 4, qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, sr_ratios=[8, 4, 2]):
        super().__init__()

        self.patch_embed = Pulse_transformation()

        self.block = nn.ModuleList([Block(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,  sr_ratio=sr_ratios)])


        self.linear=nn.Linear(3,512)


    def forward_features(self, x):
        output= self.patch_embed(x)
        attn = None
        for blk in self.block:
            x, attn = blk(output, attn)
        return x.flatten(3).mean(3)

    def forward(self, x):
        x = self.forward_features(x)
        final_x = x[0, :, :]
        final_x=self.linear(final_x)
        return final_x














