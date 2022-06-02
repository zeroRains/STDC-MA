import torch

from models.model_stages_SNL import BiSeNet

# model = torch.nn.Linear(3, 100)
# 新模型
model = BiSeNet(backbone="STDCNet1446", n_classes=19, pretrain_model=None, use_boundary_2=False, use_boundary_4=False,
                use_boundary_8=True, use_boundary_16=False, use_conv_last=False)

print("加载成功")
model.eval()
# 原模型的参数文件
originParams = torch.load("./checkpoints/STDC2-Seg/model_maxmIOU75.pth", map_location="cpu")
# 获取新模型的参数
a = torch.rand([4, 3, 1024, 512])
modelDict = model.state_dict()
print("应该可以吧")
# 如果只加不删的话可以跳过这行，这里是从旧模型中拉去新模型中有的层
pullDict = {name: value for name, value in originParams.items() if name in modelDict.keys()}
# 更新新模型的参数
modelDict.update(pullDict)
# 向新模型中加载参数
model.load_state_dict(modelDict)
# 保存模型
torch.save(model, "STDC2optimSNL.pth")
print("save finished!")
