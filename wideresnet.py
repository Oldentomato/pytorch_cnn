# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torchsummary
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

cudnn.benchmark = True
plt.ion()   # 대화형 모드

# 학습을 위해 데이터 증가(augmentation) 및 일반화(normalization)
# 검증을 위한 일반화
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = './'
test_data_dir = './'
#train: 데이터, val: 데이터 이렇게 딕셔너리로 구성되어 있음 (x=train or val)
#path.join을 쓰는 이유가있을지 테스트해봐야함
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

test_image_datasets = datasets.ImageFolder(test_data_dir, data_transforms['val'])


test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size=8,
                                                shuffle=False, num_workers=4)

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 학습 데이터의 배치를 얻습니다.
#next() => 반복 가능 객체의 다음요소 반환
#iter() => iter의 매개변수(배열)을 차례대로(배치사이즈만큼 8) 가져와서 두 변수에 담아준다.
inputs, classes = next(iter(dataloaders['train']))

#어떤값이 나올지 테스트용
print(classes)

input('Images Load Compelete')

def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    #그래프그리기위한 배열 변수
    acc_arr = []
    loss_arr = [] 
    val_acc_arr = []
    val_loss_arr = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0

    #동결화된 상태로 1번 학습하기
    for _ in range(1):
        print('WideResNet Layers Freezing And Traning')
        model.train()
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)#x 입력 데이터
            labels = labels.to(device)#y 정답 데이터
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # 학습 단계인 경우 역전파 + 최적화
                loss.backward()
                optimizer.step()

            # 통계
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        print(f'(Freezing Training) Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    #레이어 동결화풀기
    for param in model.parameters():
        param.requires_grad = True

    #학습하기
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 각 에폭(epoch)은 학습 단계와 검증 단계를 갖습니다.
        #train일때 한번 val일때 한번 총 두번
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 모델을 학습 모드로 설정
            else:
                model.eval()   # 모델을 평가 모드로 설정

            running_loss = 0.0
            running_corrects = 0

            # 데이터를 반복
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)#x 입력 데이터
                labels = labels.to(device)#y 정답 데이터

                # 매개변수 경사도를 0으로 설정
                #매번 loss.backward()를 호출할때 초기설정은 매번 gradient를 더해주는 것으로 설정되어있기 때문에
                #학습 loop를 돌때 이상적으로 학습이 이루어지기 위해선 한번의 학습이 완료되어지면(즉, Iteration이 한번 끝나면) 
                # gradients를 항상 0으로 만들어 주어야 합니다. 
                # 만약 gradients를 0으로 초기화해주지 않으면 gradient가 의도한 방향이랑 다른 방향을 가르켜 학습이 원하는 방향으로 이루어 지지 않습니다.
                optimizer.zero_grad()

                # 순전파
                # 학습 시에만 연산 기록을 추적
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 학습 단계인 경우 역전파 + 최적화
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 통계
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            #배열에 각각 저장
            if phase == 'train':
                acc_arr.append(epoch_acc)
                loss_arr.append(epoch_loss)
            else:
                val_acc_arr.append(epoch_acc)
                val_loss_arr.append(epoch_loss)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 모델을 깊은 복사(deep copy)함
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                #deep copy를 하면 포인터로 가리키는 배열이 아닌 따로 배열을 새로 만들고 복사한다.(immutable)
                #deep copy를 하지 않으면 model.state_dict()에서 값이 바뀔경우 best_model_wts도 값이 바뀌게 된다.
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f} in Epoch:{best_epoch}')

    # 가장 나은 모델 가중치를 불러옴
    model.load_state_dict(best_model_wts)

    #학습 결과 그래프로 출력
    plt.plot(acc_arr)
    plt.plot(val_acc_arr)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_accuracy','val_accuracy'])
    
    plt.plot(loss_arr)
    plt.plot(val_loss_arr)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss','val_loss'])
    plt.show()

    #해당 가중치를 저장
    torch.save(model.state_dict, './save_models/best_epoch>{best_epoch}_acc>{best_acc:4f}.pth')
    return model

def Test_Model(model):
    model.eval()
    correct = 0
    total = 0

    for images, labels in test_dataloaders:
        
        images = images.cuda()
        outputs = model(images)
        
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum()
        
    print('Accuracy of test images: %f %%' % (100 * float(correct) / total))


input('Functions Checked')


#모델 생성
model_ft = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True)

#레이어 동결
for param in model_ft.parameters():
    param.requires_grad = False

#선형과 시그모이드 합치기
num_ftrs = model_ft.fc.in_features
temp_model = nn.Sequential(
    nn.Linear(num_ftrs,1),
    nn.Sigmoid()
)
# 여기서 각 출력 샘플의 크기는 2로 설정합니다.
# 또는, nn.Linear(num_ftrs, len (class_names))로 일반화할 수 있습니다.
model_ft.fc = temp_model

model_ft = model_ft.to(device)

#CrossEntropy같은경우에는 마지막 레이어 노드수가 2개 이상이여야 한다.
#1개일 경우에는 사용이 안됨.
# 마지막 레이어가 노드수가 1개일 경우에는 보통 Binary Classification을 할때 사용될수가 있는데
# 이럴경우 BCELoss를 사용할때가 있다.
# BCELoss함수를 사용할 경우에는 먼저 마지막 레이어의 값이 0~1로 조정을 해줘야하기 때문에
# 단순하게 마지막 레이어를 nn.Linear로 할때 out of range 에러가 뜬다.
criterion = nn.BCELoss()

# 모든 매개변수들이 최적화되었는지 관찰
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001)

# 7 에폭마다 0.1씩 학습률 감소
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

#summary를 통하여 모델 구성 확인하기 channels, H, W (pip 설치 필요)
torchsummary.summary(model_ft,(3,512,512))

input('Model Load Compelete')

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=100)

# visualize_model(model_ft)
# Test_Model(model_ft)