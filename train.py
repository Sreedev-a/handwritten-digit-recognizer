from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from model import DigitNet

def train(epochs=3,data_dir="data",output="models/mnist_cnn.pt"):
    torch.manual_seed(42); dataset=datasets.MNIST(data_dir,train=True,download=True,transform=transforms.ToTensor()); loader=DataLoader(dataset,batch_size=128,shuffle=True); model=DigitNet(); optimizer=torch.optim.Adam(model.parameters(),lr=1e-3); loss_fn=nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train(); total=correct=0
        for images,labels in loader:
            optimizer.zero_grad(); logits=model(images); loss=loss_fn(logits,labels); loss.backward(); optimizer.step(); total+=len(labels); correct+=(logits.argmax(1)==labels).sum().item()
        print(f"epoch={epoch+1} accuracy={correct/total:.3f}")
    Path(output).parent.mkdir(parents=True,exist_ok=True); torch.save(model.state_dict(),output)
if __name__=="__main__": train()

