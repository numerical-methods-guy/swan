import torch, yaml
from model.paradis import Paradis

with open('config_paradis.yaml') as f:
    config = yaml.safe_load(f)
config['model']['paradis']['hidden_dim'] = 96
config['model']['paradis']['num_layers'] = 8
config['model']['paradis']['num_encoder_layers'] = 4
config['model']['paradis']['num_vels'] = 24
config['model']['paradis']['diffusion_size'] = 48
config['model']['paradis']['reaction_size'] = 24
config['model']['paradis']['bias_channels'] = 6

model = Paradis(config)
ckpt = torch.load('/space/hall0/work/eccc/mrd/rpnatm/avg000/Trained_weights_and_Graphs/20260818_run5/gbls_h+wc6_multi_20260818_070004/version_0/checkpoints/pretrain-epoch=95-val_loss=0.0833.ckpt', map_location='cpu')
sd = {k.replace('model.', '', 1): v for k, v in ckpt['state_dict'].items() if k.startswith('model.')}
model.load_state_dict(sd)
model.eval()
inp = torch.randn(1, 3, 128, 256)
winds = torch.randn(1, 2, 128, 256)
with torch.no_grad():
    out = model(inp, winds)
print('out nan:', torch.isnan(out).any().item(), 'inf:', torch.isinf(out).any().item())
print('out min:', out.min().item(), 'max:', out.max().item())
