import os
import matplotlib.pyplot as plt
import random
from PIL import Image

dataset_path="datasets\\RGB_224x224"

train_dir=os.path.join(dataset_path, 'train')

classes=os.listdir(train_dir)

print(len(classes))

print(classes)

print(random.sample(classes,5))

def show_sample_images(num_classes=4):
    fig,axes=plt.subplots(num_classes,5,figsize=(15,3*num_classes))
    for i,class_name in enumerate(random.sample(classes,num_classes)):
        class_path=os.path.join(train_dir,class_name)
        images=os.listdir(class_path)

        for j in range(5):
            img_path=os.path.join(class_path,images[j])
            img=Image.open(img_path)
            axes[i,j].imshow(img)
            axes[i,j].set_title(class_name)
            axes[i,j].axis("off")
    
    plt.tight_layout()
    plt.show()

# show_sample_images()


import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

transform= transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset=ImageFolder(os.path.join(dataset_path,'train'),transform=transform)
test_dataset=ImageFolder(os.path.join(dataset_path,'test'),transform=transform)
valid_dataset=ImageFolder(os.path.join(dataset_path,'val'),transform=transform)


train_loader=DataLoader(train_dataset,batch_size=128,shuffle=True,num_workers=0)
valid_loader=DataLoader(valid_dataset,batch_size=64,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=64,shuffle=True)


import torchvision.models as models
import torch.nn as nn

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


# print(device)

model=models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad=False


num_classes=len(train_dataset.classes)
model.fc=nn.Linear(model.fc.in_features,num_classes)
model=model.to(device)

import torch.optim as optim
import torch.nn.functional as F

optimizer=optim.Adam(model.parameters(),lr=0.001)
criterion=nn.CrossEntropyLoss()

def train_one_epoch(model, loader):
    model.train()
    running_loss=0

    for images, labels in loader:
        images, labels=images.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()

    return running_loss/len(loader)


def evaluate(model,loader):
    model.eval()
    correct=0
    total=0

    with torch.no_grad():
        for images, labels in loader:
            images,labels=images.to(device),labels.to(device)
            outputs=model(images)
            _,predicted=torch.max(outputs.data,1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item() 
            
    return correct/total


epochs=10
for epoch in range(epochs):
    train_loss= train_one_epoch(model,train_loader)
    valid_acc=evaluate(model,valid_loader)
    print(f"Epoch {epoch + 1}, Train loss: {train_loss:.4f}, Valid Accuracy: {valid_acc:.4f}")

test_acc= evaluate(model, test_loader)

print(f"Test Accuracy: {test_acc:.4f}")


def visualize_predictions(model,loader,n=5):
    model.eval()
    images_shown=0
    class_names=train_dataset.classes

    with torch.no_grad():
        for images, labels in loader:
            images=images.to(device)
            outputs=model(images)
            _,preds=torch.max(outputs,1)

            for i in range(n):
                img=images[i].cpu().permute(1,2,0).numpy()
                plt.imshow(img* 0.229 + 0.485)
                plt.title(f"True: {class_names[labels[i]]}, Pred:{class_names[preds[i]]}")
                plt.axis("off")
                plt.show()

                images_shown+=1
                if images_shown>=n:
                    return
 

visualize_predictions(model, test_loader,10)



import joblib 

model_data={
    "model_state_dict":model.state_dict(),
    "class_to_idx":train_dataset.class_to_idx
}

print(model_data)

joblib.dump(model_data,"crop_classifier_model.pkl")