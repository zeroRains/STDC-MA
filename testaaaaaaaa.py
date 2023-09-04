import torch

from models.model_stages import BiSeNet
from thop import profile

# model = torch.nn.Linear(3, 100)
# 新模型
model = BiSeNet(backbone="STDCNet1446", n_classes=12, pretrain_model=None, use_boundary_2=False, use_boundary_4=False,
                use_boundary_8=True, use_boundary_16=False, use_conv_last=False)

print("加载成功")
model.eval()
# 原模型的参数文件

a = torch.rand((1, 3, 960, 540))
flops, params = profile(model, inputs=(a,))
print(flops)
print(params)
print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
print('Params = ' + str(params / 1000 ** 2) + 'M')
