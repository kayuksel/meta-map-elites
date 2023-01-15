import torch, math, random
import torch.nn as nn
from argparse import ArgumentParser
parser = ArgumentParser(description='Input parameters for Meta-Learning MAP Elites with CNN')
parser.add_argument('--funcd', default=30, type=int, help='Size of Schwefel Function Dimensions')
parser.add_argument('--trial', default=10000, type=int, help='Number of Total Iterations for Solver')
parser.add_argument('--batch', default=64, type=int, help='Number of Evaluations in an Iteration')
parser.add_argument('--rseed', default=-1, type=int, help='Random Seed for Network Initialization')
parser.add_argument('--knn', default=8, type=int, help='Number of Nearest Neighbors for Diversity')
args = parser.parse_args()

if args.rseed < 0: args.rseed = random.randint(0, 100)
torch.manual_seed(args.rseed)
torch.cuda.manual_seed(args.rseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def schwefel(x):
    x = x * 500
    return 418.9829 * x.shape[1] - (x * x.abs().sqrt().sin()).sum(dim=1)

def unitwise_norm(x):
    dim = [1, 2, 3] if x.ndim == 4 else 0
    return torch.sum(x**2, dim=dim, keepdim= x.ndim > 1) ** 0.5

class AGC(torch.optim.Optimizer):
    def __init__(self, params, optim: torch.optim.Optimizer, clipping = 1e-2, eps = 1e-3):
        self.optim = optim
        defaults = dict(clipping=clipping, eps=eps)
        defaults = {**defaults, **optim.defaults}
        super(AGC, self).__init__(params, defaults)
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                param_norm = torch.max(unitwise_norm(
                    p), torch.tensor(group['eps']).to(p.device))
                grad_norm = unitwise_norm(p.grad)
                max_norm = param_norm * group['clipping']
                trigger = grad_norm > max_norm
                clipped = p.grad * (max_norm / torch.max(grad_norm, torch.tensor(1e-6).cuda()))
                p.grad.data.copy_(torch.where(trigger, clipped, p.grad))
        self.optim.step(closure)

class se_layer(nn.Module):
    def __init__(self, channels = 512, reduction = 32):
        super(se_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction), nn.Mish(),
            nn.Linear(channels // reduction, channels), nn.Sigmoid())
    def forward(self, x):
        b, c, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        return x * avg_out.sigmoid().view(b, c, 1)

class base_conv1d(nn.Module):
    def __init__(self, in_channels = 128, out_channels = 128, ks = 3, pd = 1, **kwargs):
        super(base_conv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, bias = False,
                              kernel_size = ks, padding=pd, **kwargs)
        self.bn, self.act = nn.BatchNorm1d(out_channels), nn.Mish()
    def forward(self, x):
        y = self.act(self.bn(self.conv(x)))
        if x.shape[1] != 128: return y
        res = y
        for i in range(1, 4):
            y = self.act(self.bn(self.conv(y)))
            res = torch.cat([y, res], dim = 1)
        return res

class res2net(nn.Module):
    def __init__(self, in_channels, attention):
        super(res2net, self).__init__()
        self.c1 = base_conv1d(in_channels, ks=1, pd=0)
        self.c2 = base_conv1d()
        self.act = nn.Mish()
        self.se = se_layer() if attention else lambda x: x
    def forward(self, x):
        res = self.se(self.c2(self.c1(x)))
        return self.act(res + x) if x.shape == res.shape else res

class CNN1d(nn.Module):
    def __init__(self, bias):
        super(CNN1d, self).__init__()
        self.conv = nn.Sequential(
            res2net(1, attention = False), res2net(512, attention = True),
            res2net(512, attention = True), nn.Conv1d(512, 1, 1, bias = bias))

        for m in self.modules():
            if isinstance(m, nn.Conv1d): m.weight.data.normal_(0, 0.01)
            if isinstance(m, nn.BatchNorm1d): m.weight.data.fill_(1)
            if hasattr(m, 'bias') and m.bias is not None: m.bias.data.zero_()

        self.std = nn.Parameter(torch.zeros(args.funcd).cuda())    

    def forward(self, x):
        x =  self.conv(x) + self.std * torch.randn_like(x)
        return x.tanh().squeeze(1)

model = CNN1d(bias = False).cuda()
opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

best_reward = None
old_rewards = None
x = torch.randn(args.batch, args.funcd).tanh().cuda()

for epoch in range((args.trial // args.batch)+1):
    torch.cuda.empty_cache()
    opt.zero_grad()
    new_x = model(x.unsqueeze(1))
    rewards = schwefel(new_x)
    actor_loss = rewards.mean()
    actor_loss.backward()
    #nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    with torch.no_grad():
        if old_rewards is None: old_rewards = rewards

        cdist = torch.cdist(new_x, new_x)
        dist_n = cdist.topk(k=args.knn, largest=False)[0].mean(dim=1)

        cdist = torch.cdist(x, x)
        dist_x = cdist.topk(k=args.knn, largest=False)[0].mean(dim=1)

        ind = (old_rewards / dist_x) > (rewards / dist_n)
        x[ind] = new_x.detach()[ind]
        old_rewards[ind] = rewards[ind]
        min_reward = old_rewards.min()

        if best_reward is None or min_reward < best_reward: best_reward = min_reward
        
print(best_reward.item())
