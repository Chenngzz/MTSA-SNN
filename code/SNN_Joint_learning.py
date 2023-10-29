import torch
import snntorch as snn
from torch import einsum, nn
from snntorch import surrogate
import torch.nn.functional as F
from einops import rearrange, repeat
from SNN_encoder import SNN_img,SNN_series, SNN_pulse_supplementation

batch_size = 1
num_inputs = 260
num_inputs_img = 256

num_hidden = 5
num_outputs = 10
num_steps = batch_size
beta = 0.95
spike_grad = surrogate.fast_sigmoid(slope=25)





class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        gamma = self.gamma[:x.shape[-1]]
        beta = self.beta[:x.shape[-1]]
        return F.layer_norm(x, x.shape[1:], gamma.repeat(x.shape[1], 1), beta.repeat(x.shape[1], 1))




class SNN_joint_learning_module(nn.Module):
    def __init__(self, dim, dim_out, heads, dim_head):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = LayerNorm(dim)

        inner_dim = heads * dim_head

        self.snn_text = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.snn_img = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.snn_out = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.identity_pic = nn.Linear(dim, inner_dim)
        self.identity_context = nn.Linear(dim, dim_head * 2)

        self.resnet = nn.Sequential(nn.Conv1d(512, 32, 1, 1),
                                    nn.Conv1d(32, batch_size, 1, 1))

        self.relu = nn.ReLU()
        self.fc = nn.Linear(512, 512)

    def forward(self, pic, context):
        image = []
        text = []


        mem1 = self.snn_img.init_leaky()
        mem2 = self.snn_text.init_leaky()


        pic = self.norm(pic)
        context = self.norm(context)




        for step in range(num_steps):
            identity_pic, mem1 = self.snn_img(pic, mem1)
            image.append(identity_pic)
            identity_pic = torch.stack(image, dim=0)



        identity_pic = self.identity_pic(identity_pic)
        identity_pic = identity_pic * self.scale



        for step in range(num_steps):
            identity_context, mem2 = self.snn_text(context, mem2)
            text.append(identity_context)
            identity_context = torch.stack(text, dim=0)


        identity_context, Value = self.identity_context(identity_context).chunk(2, dim=-1)



        if identity_context.ndim == 2 or Value.ndim == 2:
            identity_context = repeat(identity_context, 'h w -> h w c', c=64)
            Value = repeat(Value, 'h w -> h w c', c=64)


        sim = einsum('b h i d, b h i j -> b h i d', identity_pic, identity_context)
        sim = sim - sim.amax(dim=-1, keepdim=True)
        attn = sim.softmax(dim=-1)


        out = einsum('b h i j, b h i d -> b h i j', attn, Value)
        out = rearrange(out, 'b h n d -> b n (h d)')



        identity_pic = torch.sum(identity_pic, dim=3)
        out_1 = self.resnet(out)
        out_1 = self.fc(out_1)
        out_1 = out_1 + identity_pic


        out_1 = torch.transpose(out_1, 1, 2)
        out_1 = F.interpolate(out_1, size=(64,), mode='linear', align_corners=False)



        identity_context = torch.sum(identity_context, dim=1)
        out_2 = self.relu(out_1) + identity_context
        out_2 = self.relu(out_2)

        return out_2


class Fusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.img_net = SNN_img()
        self.img_enhance = SNN_pulse_supplementation()
        self.text_net = SNN_series()
        self.output_layer = nn.Sequential(nn.Flatten(),
                                nn.Linear(32768, 1024),
                                nn.Linear(1024, 5),
                                nn.ReLU()
                                )


        self.fusion_1 = SNN_joint_learning_module(dim=512, dim_out=512, heads=8, dim_head=64)

        self.linear_img = nn.Sequential(nn.Linear(4096, 512), nn.ReLU())
        self.linear_series = nn.Sequential(nn.Linear(10, 512), nn.ReLU())
        self.linear_enhance = nn.Sequential(nn.Linear(512, 5), nn.ReLU())

    def forward(self, img, series):
        series_embed = self.text_net(series)
        series_embed = series_embed.reshape(batch_size, -1)
        series_embed = self.linear_series(series_embed)
        series_embed = rearrange(series_embed, 'b n -> b n 1')
        series_embed = series_embed.repeat(1, 1, 512)


        img_embed = self.img_net(img)
        img_embed = img_embed.reshape(batch_size, 512, -1)
        img_embed = self.linear_img(img_embed)

        img_enhance = self.img_enhance(img)
        img_enhance = self.linear_enhance(img_enhance)


        out = self.fusion_1(img_embed, series_embed)
        out = self.output_layer(out)


        out = out + img_enhance

        return out


net_snn = Fusion()
img = torch.randn(batch_size, 3, 256, 256)
x = torch.randn(batch_size, 260)

output = net_snn(img, x)
print(output.shape)


