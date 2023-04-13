# 패키지 업로드
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, Compose, Resize, Normalize
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.optim import Adam
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128, help='size of batch')
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", default=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) 
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--model_type", default='lenet', choices=['mlp', 'lenet', 'linear', 'multi_conv', 'incep'])
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args.epochs) # dictionary와 비슷하게 생겼지만 indexing 할 수 없음. namespace 객체
    # dictionary로 바꾸고 싶다면 args.__dict__ or vars(args)를 사용하면 됨
    
    # image 통계치 확인
    tmp_dataset = CIFAR10(root='/Users/iyubin/aialone/data', train=True, transform=None, download=True)
    mean = list(tmp_dataset.data.mean(axis=(0,1,2))/255)
    std = list(tmp_dataset.data.std(axis=(0,1,2))/255)

    # dataset
    # 이미지 크기 변경, 텐서 만들기 > compose를 이용해 여러 transform을 하나의 객체로 묶어줌
    trans = Compose([Resize((args.img_size, args.img_size)), 
                    ToTensor(),
                    Normalize(mean=mean, std=std)])

    train_dataset =  CIFAR10(root='./data', train=True, transform=trans, download=True)
    test_dataset = CIFAR10(root='./data', train=False, transform=trans, download=True)

    # dataloader
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False) # 끝까지 다 쓸 것 (drop_last=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # 모델 class
    class MyMLP(nn.Module):
        def __init__(self, hidden_size, output_size):
            super().__init__()
            self.fc1 = nn.Linear(28*28, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, hidden_size)
            self.fc4 = nn.Linear(hidden_size, hidden_size)
            self.fc5 = nn.Linear(hidden_size, output_size)
            
        def forward(self, x):
            b, c, w, h = x.shape # (128, 1, 28, 28)
            # x = x.reshape(b, w*h) # (b, 28 * 28)
            x = x.reshape(b, -1)
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
            x = self.fc4(x)
            x = self.fc5(x)
            return x
    
    class myLeNet(nn.Module):
        def __init__(self):
            super().__init__()
            # convolution은 batch norm과 activation을 사용해야 학습이 안정적으로 진행됨
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5) # padding=0, stride=1은 default 값
            self.bn1 = nn.BatchNorm2d(num_features=6)
            self.act1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=2)
            
            self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
            self.bn2 = nn.BatchNorm2d(num_features=16)
            self.act2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(kernel_size=2)
            
            self.fc1 = nn.Linear(16*5*5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)
            
        def forward(self, x):
            b, c, h, w = x.shape
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.act1 (x)
            x = self.pool1(x)
            
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.act2 (x)
            x = self.pool2(x)
            
            x = x.reshape(b, -1)
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
            return x

    class myLeNet_seq(nn.Module): # sequential을 통한 model class 선언
        def __init__(self): # init 함수 정의
            super().__init__() # 부모 class 초기화
            self.seq1 = nn.Sequential( # conv layer 2개로 이우러진 seq1 정의
                nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5), # 첫번째 conv
                nn.BatchNorm2d(num_features=6), # batchnorm
                nn.ReLU(), # activation
                nn.MaxPool2d(kernel_size=2), # maxpooling
                
                nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5), # 두번째 conv
                nn.BatchNorm2d(num_features=16), # batchnorm
                nn.ReLU(), # activation
                nn.MaxPool2d(kernel_size=2) # maxpooling
            )
            
            self.seq2 = nn.Sequential( # fc layer 3개로 이루어진 seq2 정의
                nn.Linear(16*5*5, 120), # fc layer 1
                nn.Linear(120, 84), # fc layer 2
                nn.Linear(84, 10) # fc layer 3
            )
            
        def forward(self, x): # forward 함수 정의
            b, c, h, w = x.shape # x shape
            x = self.seq1(x) # seq 1
            x = x.reshape(b, -1) # seq 2의 fc layer를 위한 reshape
            x = self.seq2(x) # seq 2
            return x # 반환

    class myLeNet_conv(nn.Module): # conv layer 추가로 모델 capacity를 늘린 lenet
        def __init__(self): # init 함수 정의
            super().__init__() # 부모 class 초기화
            self.conv1_ = nn.ModuleList([nn.Conv2d(3,6,5,1,2)] +
                                        [nn.Conv2d(6,6,5,1,2) for _ in range(2)]) # module list를 통한 총 3개의 conv layer
            self.conv1 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=5) # architecture 상으로 4번째 conv layer
            self.bn1 = nn.BatchNorm2d(num_features=6) # batchnorm
            self.act1 = nn.ReLU() # activation function 
            self.pool1 = nn.MaxPool2d(kernel_size=2) # maxpooling
            
            self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5) # 5번째 conv layer
            self.bn2 = nn.BatchNorm2d(num_features=16) # batchnorm
            self.act2 = nn.ReLU() # activation function
            self.pool2 = nn.MaxPool2d(kernel_size=2) # maxpooling
            
            self.fc1 = nn.Linear(16*5*5, 120) # 첫번째 fc layer
            self.fc2 = nn.Linear(120, 84) # 두번째 fc layer
            self.fc3 = nn.Linear(84, 10) # 세번째 fc layer
            
        def forward(self, x): # forward 함수 정의
            b, c, h, w = x.shape # x shape
            for module in self.conv1_: # modulelist에 있는 요소만큼 반복
                x = module(x) # 해당 순번의 modulelist 요소의 layer
            x = self.conv1(x) # conv
            x = self.bn1(x) # batchnorm
            x = self.act1(x) # activation
            x = self.pool1(x) # pooling
            
            x = self.conv2(x) # conv
            x = self.bn2(x) # batchnorm
            x = self.act2(x) # activation
            x = self.pool2(x) # pooling
            
            x = x.reshape(b, -1) # fc layer를 위한 reshape
            x = self.fc1(x) # fc layer 1
            x = self.fc2(x) # fc layer 2
            x = self.fc3(x) # fc layer 3
            return x # 반환

    class myLeNet_linear(nn.Module): # 두 conv 사이에 fc layer를 추가한 lenet
        def __init__(self): # init
            super().__init__() # 초기화
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5) # conv 1
            self.bn1 = nn.BatchNorm2d(num_features=6) # batchnorm
            self.act1 = nn.ReLU() # activation
            self.pool1 = nn.MaxPool2d(kernel_size=2) # maxpooling
            
            self.conv_fc1 = nn.Linear(6*14*14, 2048) # conv 사이의 linear 1
            self.conv_fc2 = nn.Linear(2048, 6*14*14) # conv 사이의 linear 2
            
            self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5) # conv 2
            self.bn2 = nn.BatchNorm2d(num_features=16) # batchnorm 
            self.act2 = nn.ReLU() # activation function
            self.pool2 = nn.MaxPool2d(kernel_size=2) # maxpooling
            
            self.fc1 = nn.Linear(16*5*5, 120) # fc layer 1
            self.fc2 = nn.Linear(120, 84) # fc layer 2
            self.fc3 = nn.Linear(84, 10) # fc layer 3
            
        def forward(self, x): # forward
            b, c, h, w = x.shape # x shape
            x = self.conv1(x) # conv1
            x = self.bn1(x) # batchnorm
            x = self.act1(x) # activation
            x = self.pool1(x) # pooling
            
            tmp_b, tmp_c, tmp_h, tmp_w = x.shape # 중간 linear layer를 위한 x shape
            x = x.reshape(tmp_b, -1) # 중간 linear layer를 위한 reshape
            x = self.conv_fc1(x) # 중간 linear 1
            x = self.conv_fc2(x) # 중간 linear 2
            x = x.reshape(tmp_b, tmp_c, tmp_h, tmp_w) # 다음 conv layer를 위한 reshape
            
            x = self.conv2(x) # conv
            x = self.bn2(x) # batchnorm
            x = self.act2(x) # activation
            x = self.pool2(x) # pooling
            
            x = x.reshape(b, -1) # 마지막 fc layer를 위한 reshape
            x = self.fc1(x) # fc 1
            x = self.fc2(x) # fc 2
            x = self.fc3(x) # fc 3
            return x # 반환
            
    class myLeNet_inception(nn.Module): # inception 구조를 추가한 lenet
        def __init__(self): # init
            super().__init__() # 초기화
            self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=1, stride=1, padding=0) # 첫번째 branch conv
            self.conv1_3 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1) # 두번째 branch conv
            self.conv1_5 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2) # 세번쨰 branch conv
            self.merge = nn.Conv2d(in_channels=18, out_channels=6, kernel_size=5) # 각 branch를 merge하는 layer
            
            self.bn1 = nn.BatchNorm2d(num_features=6) # batchnorm
            self.act1 = nn.ReLU() # activation function
            self.pool1 = nn.MaxPool2d(kernel_size=2) # maxpooling
            
            self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5) # conv layer
            self.bn2 = nn.BatchNorm2d(num_features=16) # batchnorm
            self.act2 = nn.ReLU() # activation function
            self.pool2 = nn.MaxPool2d(kernel_size=2) # maxpooling
            
            self.fc1 = nn.Linear(16*5*5, 120) # fc 1
            self.fc2 = nn.Linear(120, 84) # fc 2
            self.fc3 = nn.Linear(84, 10) # fc 3
            
        def forward(self, x): # forward
            b, c, h, w = x.shape # x shape
            x1 = self.conv1_1(x) # 첫번째 branch conv
            x2 = self.conv1_3(x) # 두번째 branch conv
            x3 = self.conv1_5(x) # 세번쨰 branch conv
            x_concat = torch.cat((x1, x2, x3), 1) # merge를 위한 세 branch concatenate
            x = self.merge(x_concat) # merge
            
            x = self.bn1(x) # batchnorm
            x = self.act1(x) # activation
            x = self.pool1(x) # pooling
            
            x = self.conv2(x) # conv
            x = self.bn2(x) # batchnorm
            x = self.act2(x) # activation
            x = self.pool2(x) # pooling
            
            x = x.reshape(b, -1) # fc를 위한 reshape
            x = self.fc1(x) # fc 1
            x = self.fc2(x) # fc 2
            x = self.fc3(x) # fc 3
            return x # 반환
        
    # 모델 객체, loss, optim
        # 모델 객체 만들기, loss 만들기, optim 만들기
    if args.model_type == 'mlp': 
        model = MyMLP(args.hidden_size, args.num_class).to(args.device) 
    elif args.model_type == 'lenet' : 
        model = myLeNet().to(args.device)
    elif args.model_type == 'linear': 
        model = myLeNet_linear().to(args.device)
    elif args.model_type == 'multi_conv': 
        model = myLeNet_conv().to(args.device)
    elif args.model_type == 'incep': 
        model = myLeNet_inception().to(args.device)
    else : 
        raise ValueError('뭔가 잘못됨')
   
    loss_f = nn.CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=args.lr)

    def evaluate(model, dataloader):
        with torch.no_grad(): # with로 묶여있는 만큼은 gradient 계산 안함
            model.eval()
            correct = 0
            total = 0
            for (image, target) in dataloader:
                image = image.to(args.device)
                target = target.to(args.device)
                
                output = model(image)
                value, index = torch.max(output, dim=1)
                correct += (index == target).sum().item()
                total += index.shape[0]

            acc = correct/total
        model.train() # 가독성, 더 정확히 하기 위해서 
        return acc

    def evaluate_class(model, dataloader):
        with torch.no_grad(): # with로 묶여있는 만큼은 gradient 계산 안함
            model.eval()
            correct = torch.zeros(args.num_class)
            total = torch.zeros(args.num_class)
            for (image, target) in dataloader:
                image = image.to(args.device)
                target = target.to(args.device)
                
                output = model(image)
                value, index = torch.max(output, dim=1)
                for i in range(args.num_class):
                    total[i] += (target == i).sum().item()
                    correct[i] += ((target == i) & (index == i)).sum().item() # *(곱셈) 연산도 가능

        model.train() # 가독성, 더 정확히 하기 위해서 
        return total, correct

    # 학습 시각화
    losses = []

    # 학습 loop
    for epoch in range(args.epochs):
        for idx, (image, labels) in enumerate(train_loader):
            image = image.to(args.device)
            labels = labels.to(args.device)

            out = model(image)
            loss = loss_f(out, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            if idx % 100 == 0:
                print('loss : ', loss.item())
                acc = evaluate(model, test_loader)
                print('accuracy : ', acc)
                losses.append(loss.item())
                total, correct = evaluate_class(model, test_loader)
                pass


# parser
if __name__ == '__main__': # 실제로 실행시키는 경우에 이부분을 실행해라
    main()