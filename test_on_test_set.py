from data import get_data
import torch
import torch.nn as nn
from th_train import cifar_model
import matplotlib.pyplot as plt
from data import poison_frequency

# Định nghĩa thiết bị
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # Add cifar_model to the allowlist for safe loading
torch.serialization.add_safe_globals({'cifar_model': cifar_model})

# # Load the model
model = torch.load("./model/cifar10.pth", weights_only=False)
model.to(device)
model.eval()

# Tham số cho trigger
param = {
    "dataset": "CIFAR10",             # GTSRB, cifar10, MNIST, PubFig, ImageNet16
    "target_label": 8,              # target label
    "poisoning_rate": 0.02,         # ratio of poisoned samples
    "label_dim": 10,
    "channel_list": [1, 2],         # [0,1,2] means YUV channels, [1,2] means UV channels
    "magnitude": 20,
    "YUV": True,
    "clean_label": False,
    "window_size": 32,
    "pos_list": [(31, 31), (15, 15)],
}

# Lấy dữ liệu
x_train, y_train, x_test, y_test = get_data(param)
x_test = x_test[:1000]
y_test = y_test[:1000]

x_clean = x_test.copy()
y_clean = y_test.copy()

# Poisoning frequency
x_test = poison_frequency(x_test, y_test, param)

# Chuyển dữ liệu về tensor và điều chỉnh thứ tự các chiều
x_test = torch.tensor(x_test, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

x_clean = torch.tensor(x_clean, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
y_clean = torch.tensor(y_clean, dtype=torch.long).to(device)

# Dự đoán
# Thử dự đoán ảnh clean ban đầu
output = model(x_clean)
# Dự đoán ảnh bị poison
# output = model(x_test)
probabilities = torch.nn.functional.softmax(output, dim=1)
predicted_class = torch.argmax(output, dim=1)

# index = 1

# # Hiển thị ảnh
# plt.imshow(x_clean[index])
# plt.show()

# plt.imshow(x_test[index].permute(1, 2, 0).cpu().numpy())
# plt.show()

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# print(f"Predicted Label: {class_names[predicted_class[index].item()]}")

count = 0
total = 1000
for i in range(total):
    if predicted_class[i] == param["target_label"]:
        count += 1

print(f"Number of images predicted as target label: {count}/{total}")
# print(output)
# print(probabilities)
