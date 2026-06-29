# import torch
# import torch.nn as nn
# from cs336_basics.optimizer import get_cosine_lr,AdamW
# from cs336_basics.nn_utils import cross_entropy,clip_gradient,softmax
# # s = torch.tensor(0, dtype=torch.float32)
# # for i in range(1000):
# #     s += torch.tensor(0.01, dtype=torch.float32)
# # print(s)
# #
# # s = torch.tensor(0, dtype=torch.float16)
# # for i in range(1000):
# #     s += torch.tensor(0.01, dtype=torch.float16)
# # print(s)
# #
# # s = torch.tensor(0, dtype=torch.float32)
# # for i in range(1000):
# #     s += torch.tensor(0.01, dtype=torch.float16)
# # print(s)
# #
# # s = torch.tensor(0, dtype=torch.float32)
# # for i in range(1000):
# #     x = torch.tensor(0.01, dtype=torch.float16)
# #     s += x.type(torch.float32)
# # print(s)
#
#
# import torch
# import torch.nn as nn
# from cs336_basics.nn_utils import cross_entropy
#
#
# class ToyModel(nn.Module):
#     def __init__(self, in_features: int, out_features: int):
#         super().__init__()
#         self.fc1 = nn.Linear(in_features, 10, bias=False)
#         self.ln = nn.LayerNorm(10)
#         self.fc2 = nn.Linear(10, out_features, bias=False)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         print(x.dtype)
#         x = self.ln(x)
#         print(x.dtype)
#         x = self.fc2(x)
#         print(x.dtype)
#         return x
#
#
# if __name__ == "__main__":
#     # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     # model = ToyModel(10, 10).to(device)
#     # adamw = AdamW(model.parameters())
#     #
#     # # --------------------------
#     # # 输入：float32 没问题
#     # # --------------------------
#     # input = torch.randint(
#     #     low=0,
#     #     high=10000,
#     #     size=(1, 10, 10),
#     #     dtype=torch.float32,
#     #     device=device
#     # )
#     #
#     # # --------------------------
#     # # ✅ 修复 1：target 必须是 int64 (long)
#     # # --------------------------
#     # target = torch.randint(
#     #     low=0,
#     #     high=10,
#     #     size=(10, 1),
#     #     dtype=torch.long,  # 必须是 long !!!
#     #     device=device
#     # )
#     #
#     # # --------------------------
#     # # ✅ 修复 2：新版 AMP 写法
#     # # --------------------------
#     # scaler = torch.amp.GradScaler('cuda')
#     # with torch.amp.autocast('cuda',dtype = torch.bfloat16):
#     #     logits = model(input)
#     #     print(logits.dtype)
#     #
#     #     logits = logits.view(-1,logits.size(-1))
#     #     target = target.view(-1)
#     #
#     #     loss = cross_entropy(logits, target)
#     #     print(loss.dtype)
#     #
#     # # --------------------------
#     # # 反向传播
#     # # --------------------------
#     # adamw.zero_grad()
#     # scaler.scale(loss).backward()
#     # scaler.step(adamw)
#     # scaler.update()
#     #
#     # for name, param in model.named_parameters():
#     #     print(name, param.grad.dtype)
#
#     print(torch.full((3,2),2)-torch.tensor([1,2,3]).unsqueeze(-1))
import torch

print("torch:", torch.__version__)
print("cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("flash sdp enabled:", torch.backends.cuda.flash_sdp_enabled())
print("mem efficient sdp enabled:", torch.backends.cuda.mem_efficient_sdp_enabled())
print("math sdp enabled:", torch.backends.cuda.math_sdp_enabled())