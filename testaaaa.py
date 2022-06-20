import torch

from models.model_stages_FaPN import BiSeNet

# model = torch.nn.Linear(3, 100)
# 新模型
model = BiSeNet(backbone="STDCNet1446", n_classes=12, pretrain_model=None, use_boundary_2=False, use_boundary_4=False,
                use_boundary_8=True, use_boundary_16=False, use_conv_last=False)

print("加载成功")
model.eval()
# 原模型的参数文件
originParams = torch.load("./checkpoints/STDC2-Seg/model_maxmIOU75.pth", map_location="cpu")
# 获取新模型的参数
modelDict = model.state_dict()
print("应该可以吧")
a = torch.rand((1, 3, 970, 720))
# 如果只加不删的话可以跳过这行，这里是从旧模型中拉去新模型中有的层
pullDict = {name: value for name, value in originParams.items() if name in modelDict.keys()}
# 去除最后一曾，类数发生改变时使用下面三行，否则不需要使用
del pullDict["conv_out.conv_out.weight"]
del pullDict["conv_out16.conv_out.weight"]
del pullDict["conv_out32.conv_out.weight"]
# 更新新模型的参数
modelDict.update(pullDict)
# 向新模型中加载参数
model.load_state_dict(modelDict)
model(a)
# 保存模型
torch.save(model.state_dict(), "STDC2optim_camvid_fapn.pth")
print("save finished!")
