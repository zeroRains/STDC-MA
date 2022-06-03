import torch
# from thop import profile
from models.model_stages import BiSeNet
import matplotlib.pyplot as plt
# from torchsummary import summary


def visual(module, inputs, outputs):
    x = inputs[0][0][0]
    plt.imshow(x.detach().numpy())
    plt.savefig("1_low_level.jpg")
    exit()


a = torch.rand((1, 3, 512, 1024))
# model = BiSeNet(backbone="STDCNet1446", n_classes=19, pretrain_model=None, use_boundary_2=False, use_boundary_4=False,
#                 use_boundary_8=True, use_boundary_16=False, use_conv_last=False)
# model.load_state_dict(torch.load("./checkpoints/STDC2-Seg/model_maxmIOU50.pth",map_location="cpu"))
model = torch.load("checkpoints/train_STDC2-Seg/pths/model_maxmIOU50.pth",map_location="cuda:0")
model.eval()

# summary(model, (3, 256, 512))

# for name, m in model.named_modules():
#     # if isinstance(m, FeatureAlign_V2):
#     #     m.register_forward_hook(visual)
#     if isinstance(m, FeatureSelectionModule):
#         m.register_forward_hook(visual)
# test = model(a)
# print(test)

# flops, params = profile(model, inputs=(a,))
# print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
# print('Params = ' + str(params / 1000 ** 2) + 'M')
