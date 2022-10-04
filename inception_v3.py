# License: BSD
# Author: Sasank Chilamkurthy
# Adam으로 바꾸고, batch_size = 8 -> 16, ReLU와 Dropout 추가
from __future__ import print_function, division

import torch
# from torchinfo import summary
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from sys import stdout
import sys
from torch.utils.data import Dataset
import numpy as np
from Send_Mongo import SendLog_ToMongo

cudnn.benchmark = True
plt.ion()  # 대화형 모드

# print(torch.cuda.is_available())

experiment_count = 10
# 하이퍼파라미터 변수
lr = 0.001
early_stop_patience = 10
batch_size = 32
sgd_momentum = 0.2
lr_scheduler_gamma = 0.1
lr_scheduler_step = 5
epoch = 50
dropout = 0.3
fc_linear_node = 50

# 학습을 위해 데이터 증가(augmentation) 및 일반화(normalization)
# 검증을 위한 일반화
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.Resize(512),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Lambda(lambda x: x.rotate(90)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
f = open('results/_' + str(experiment_count) + '.txt', 'w')
data_dir = './mel_images/melanoma'
test_data_dir = './mel_images/melanoma/test'
# train: 데이터, val: 데이터 이렇게 딕셔너리로 구성되어 있음 (x=train or val)
# path.join을 쓰는 이유가있을지 테스트해봐야함
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'valid']}

test_image_datasets = datasets.ImageFolder(test_data_dir, data_transforms['test'])

# 파이토치 로더를 사용하면 데이터 혼합과 멀티스레드 사용과 같은 여러가지 이점을 얻을 수있고, 처리 속도를 높일 수 있다
# num_workers는 멀티 프로세싱과 관련된 파라미터이다. 학습 도중 CPU의 작업을 몇 개의 코어를 사용해서 진행할지에 대한 설정 파라미터이다.
test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size=batch_size,
                                               shuffle=False, num_workers=0)

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                              shuffle=True, num_workers=0)
               for x in ['train', 'valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes
test_class_names = test_image_datasets.classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 학습 데이터의 배치를 얻습니다.
# next() => 반복 가능 객체의 다음요소 반환
# iter() => iter의 매개변수(배열)을 차례대로(배치사이즈만큼 8) 가져와서 두 변수에 담아준다.
inputs, classes = next(iter(dataloaders['train']))

print('Images Load Compelete')


class EarlyStopping:
    def __init__(self, patience=5):
        self.loss = np.inf
        self.patience = 0
        self.patience_limit = patience

    def step(self, loss):
        if self.loss > loss:
            self.loss = loss
            self.patience = 0
        else:
            self.patience += 1

    def is_stop(self):
        return self.patience >= self.patience_limit

send_db = SendLog_ToMongo(data={
    "batch_size": batch_size,
    "learning_rate": lr,
    "sgd_momentum":sgd_momentum,
    "lr_scheduler_gamma":lr_scheduler_gamma,
    "lr_scheduler_step":lr_scheduler_step,
    "epoch": 50,
    "experiment_count":experiment_count
})

def train_model(model, criterion, optimizer, scheduler, early_stop, num_epochs):
    since = time.time()
    # 그래프그리기위한 배열 변수
    acc_arr = []
    loss_arr = []
    val_acc_arr = []
    val_loss_arr = []
    escape = False


    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0

    # 동결화된 상태로 1번 학습하기
    for _ in range(1):
        running_loss = 0.0
        running_corrects = 0
        print('Inception Layers Freezing And Traning')
        model.train()
        count = 0
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)  # x 입력 데이터
            labels = labels.to(device)  # y 정답 데이터
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs.to(torch.float64), labels)

                loss.backward()
                optimizer.step()

            # 통계
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            stdout.write("\r=======" + str(count) + "/" + str(len(dataloaders['train']) - 1) + "  progressed=======")
            stdout.flush()
            count += 1

        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = running_corrects / dataset_sizes['train']
        print(f'(Freezing Training) Loss: {epoch_loss:.3f} Acc: {epoch_acc:.3f}')
        print(f'(Freezing Training) Loss: {epoch_loss:.3f} Acc: {epoch_acc:.3f}', file=f)

    # 레이어 동결화풀기
    for param in model.parameters():
        param.requires_grad = True

    # 학습하기
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}', file=f)
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 20)

        # 각 에폭(epoch)은 학습 단계와 검증 단계를 갖습니다.
        # train일때 한번 val일때 한번 총 두번
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # 모델을 학습 모드로 설정
            else:
                model.eval()  # 모델을 평가 모드로 설정

            running_loss = 0.0
            running_corrects = 0
            count = 0
            # 데이터를 반복
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)  # x 입력 데이터
                labels = labels.to(device)  # y 정답 데이터

                # 매개변수 경사도를 0으로 설정
                # 매번 loss.backward()를 호출할때 초기설정은 매번 gradient를 더해주는 것으로 설정되어있기 때문에
                # 학습 loop를 돌때 이상적으로 학습이 이루어지기 위해선 한번의 학습이 완료되어지면(즉, Iteration이 한번 끝나면)
                # gradients를 항상 0으로 만들어 주어야 합니다.
                # 만약 gradients를 0으로 초기화해주지 않으면 gradient가 의도한 방향이랑 다른 방향을 가르켜 학습이 원하는 방향으로 이루어 지지 않습니다.
                optimizer.zero_grad()

                # 순전파
                # 학습 시에만 연산 기록을 추적
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs.to(torch.float64), labels)

                    # 학습 단계인 경우 역전파 + 최적화
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 통계
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if phase == "train":
                    stdout.write("\r(" + str(phase) + ")=======" + str(count) + "/" + str(
                        len(dataloaders['train']) - 1) + "progressed=======")
                else:
                    stdout.write("\r(" + str(phase) + ")=======" + str(count) + "/" + str(
                        len(dataloaders['valid']) - 1) + "progressed=======")
                stdout.flush()
                count += 1
            if phase == 'train':
                scheduler.step()
            # else:
            #     early_stop.step(loss.item())
            #
            # if early_stop.is_stop():
            #     print(f'Stopped at {epoch + 1} epoch to loss')
            #     escape = True

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            # 배열에 각각 저장
            if phase == 'train':
                acc_arr.append(epoch_acc.item())#근데 얘는 돼
                loss_arr.append(epoch_loss)#얘는 안돼
            else:
                val_acc_arr.append(epoch_acc.item())
                val_loss_arr.append(epoch_loss)
                print()
                print("Send to DB...")
                send_db.on_epoch_end(epoch=epoch, logs={
                    'epoch': epoch,
                    'loss': loss_arr,
                    'accuracy': acc_arr,
                    'val_loss': val_loss_arr,
                    'val_accuracy': val_acc_arr
                }, early_stopped=escape)
                print()
                print("DB Send Compelete")

            print(f'{phase} Loss: {epoch_loss:.3f} Acc: {epoch_acc:.3f}')
            print(f'{phase} Loss: {epoch_loss:.3f} Acc: {epoch_acc:.3f}', file=f)


            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch + 1
                # deep copy를 하면 포인터로 가리키는 배열이 아닌 따로 배열을 새로 만들고 복사한다.(immutable)
                # deep copy를 하지 않으면 model.state_dict()에서 값이 바뀔경우 best_model_wts도 값이 바뀌게 된다.
                best_model_wts = copy.deepcopy(model.state_dict())
        # if escape == True:
        #     break
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s', file=f)
    print(f'Best val Acc: {best_acc:3f} in Epoch:{best_epoch}', file=f)
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:3f} in Epoch:{best_epoch}')

    # 가장 나은 모델 가중치를 불러옴
    model.load_state_dict(best_model_wts)


    # # 학습 결과 그래프로 출력
    plt.plot(acc_arr)
    plt.plot(val_acc_arr)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_accuracy', 'val_accuracy'])

    plt.plot(loss_arr)
    plt.plot(val_loss_arr)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'val_loss'])
    plt.show()

    # 해당 가중치를 저장
    torch.save(model.state_dict(), './save_models/inception/best_epoch_' + str(experiment_count) + '.pth')

    return model


def Test_Model(model):
    model.eval()
    correct = 0
    total = 0
    count = 0

    for images, labels in test_dataloaders:
        images = images.cuda()
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += torch.sum(predicted == labels.data.cuda())
        stdout.write("\r(Test)=======" + str(count) + "/" + str(
            len(test_dataloaders) - 1) + "progressed=======")
        count += 1
        stdout.flush()

    send_db.on_test_end(test_acc=100*float(correct)/total)
    print()
    print('Accuracy of test images: %f %%' % (100 * float(correct) / total), file=f)
    print('Accuracy of test images: %f %%' % (100 * float(correct) / total))


print('Functions Checked')

print(f'[Learning_Rate] : {lr}')
print(f'[early_stop_patience] : {early_stop_patience}')
print(f'[batch_size] : {batch_size}')
print(f'[sgd_momentum] : {sgd_momentum}')
print(f'[lr_scheduler_gamma] : {lr_scheduler_gamma}')
print(f'[lr_scheduler_step] : {lr_scheduler_step}')
print(f'[epoch] : {epoch}')
print(f'[dropout] : {dropout}')
print(f'[fc_linear_node] : {fc_linear_node}')

print(f'[Learning_Rate] : {lr}', file=f)
print(f'[early_stop_patience] : {early_stop_patience}', file=f)
print(f'[batch_size] : {batch_size}', file=f)
print(f'[sgd_momentum] : {sgd_momentum}', file=f)
print(f'[lr_scheduler_gamma] : {lr_scheduler_gamma}', file=f)
print(f'[lr_scheduler_step] : {lr_scheduler_step}', file=f)
print(f'[epoch] : {epoch}', file=f)
print(f'[dropout] : {dropout}', file=f)
print(f'[fc_linear_node] : {fc_linear_node}', file=f)

# 모델 생성
model_ft = models.inception_v3(pretrained=True)

model_ft.aux_logits = False
print("load_inception_v3-model : done")
# 레이어 동결
for name, param in model_ft.named_parameters():
    if name.split('.')[1] == 'fc':
        pass
    else:
        param.requires_grad = False
print("freeze_layer : done")

# 선형과 시그모이드 합치기
num_ftrs = model_ft.fc.in_features
print(num_ftrs)
temp_model = nn.Sequential(
    nn.Linear(num_ftrs, fc_linear_node),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(fc_linear_node, 2)
)
# 여기서 각 출력 샘플의 크기는 2로 설정합니다.
# 또는, nn.Linear(num_ftrs, len (class_names))로 일반화할 수 있습니다.
model_ft.fc = temp_model

model_ft = model_ft.to(device)
print("Fully Connected Layer: done")
# CrossEntropy같은경우에는 마지막 레이어 노드수가 2개 이상이여야 한다.

# CrossEntropyLoss = logSoftmax + NLLLoss
criterion = nn.CrossEntropyLoss()
# 모든 매개변수들이 최적화되었는지 관찰
# l2 규제 (가중치 감쇠)
# momentum: weight 를 조정 (관성)
optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=sgd_momentum)

early_stop = EarlyStopping(patience=early_stop_patience)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=lr_scheduler_step, gamma=lr_scheduler_gamma)
print("criterion, optimizer, lr_scheduler ready : done")
# summary를 통하여 모델 구성 확인하기 channels, H, W (pip 설치 필요)
# summary(model_ft,(8,3,512,512))#배치사이즈,채널,가로,세로

print('Model Load Compelete')

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, early_stop,
                       num_epochs=epoch)

# visualize_model(model_ft)
Test_Model(model_ft)
f.close()