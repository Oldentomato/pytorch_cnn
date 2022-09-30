from __future__ import print_function, division

import torch
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import torch.nn as nn
from sys import stdout

cudnn.benchmark = True
plt.ion()  # 대화형 모드

data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

test_data_dir = './mel_images/melanoma/test'

test_image_datasets = datasets.ImageFolder(test_data_dir, data_transforms['test'])

test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size=16,
                                               shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_sizes = len(test_image_datasets)

def Test_Model(model):
    model.eval()
    correct = 0
    count = 0

    for images, labels in test_dataloaders:
        images = images.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            correct += torch.sum(predicted == labels.data)
            stdout.write("\r(Test)=======" + str(count) + "/" + str(
                len(test_dataloaders)-1) + "progressed=======")
            count+=1
            stdout.flush()
            # correct += (predicted == labels.cuda()).sum()
    print()
    print('Accuracy of test images: %f %%' % (100 * correct / dataset_sizes))

model_ft = models.inception_v3(pretrained=True)

model_ft.aux_logits = False
num_ftrs = model_ft.fc.in_features

temp_model = nn.Sequential(
    nn.Linear(num_ftrs, 50),
    nn.Dropout(0.2),
    nn.Linear(50, 2)
)
model_ft.fc = temp_model

model_ft = model_ft.to(device)
model_ft.load_state_dict(torch.load("./save_models/inception/best_epoch.pth", map_location=device))

Test_Model(model_ft)