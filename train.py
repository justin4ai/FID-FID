import glob
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src import CustomDataLoader
from src import HighFreqVit

def main():
    customloader = CustomDataLoader.DataProcesser()
    data = customloader.get_datasets(path = "./datasets")
    batch_size = 50
    dataloader = DataLoader(dataset = data, batch_size = batch_size, shuffle = True)
    num_datas = len(dataloader)*batch_size
    classifier = HighFreqVit.HighFreqVitClassifier()
    device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
    print(device, torch.cuda.get_device_name())
    classifier.to(device)
    num_epochs = 100
    optimizer = torch.optim.Adam(classifier.parameters(), lr = 0.01)
    checkpoint = None
    
    if glob.glob("./*.pt"):
        checkpoint = torch.load("./model_state_dict.pt")
        classifier.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        checkpoint_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        
    for epoch in range(num_epochs):
        
        if bool(checkpoint) and (epoch <= checkpoint_epoch):
            continue
        
        acc = 0
        
        for batch_num, batch in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()
            
            img, labels = batch
            labels = list(labels)
            
            logits, loss = classifier(img.to(device), labels, device)
            
            for idx in range(len(labels)):
                if labels[idx] == logits[idx]:
                    acc = acc + 1
            
            loss.backward()
            optimizer.step()
            
        acc = acc/num_datas * 100
        print("\nEpoch : ", epoch, "\nTraining accuracy : ", acc, "%\nTraining Loss : ", loss.item(), "\n")
        
        torch.save({
                    'epoch' : epoch, 
                    'model_state_dict' : classifier.state_dict(),
                    'optimizer_state_dict' : optimizer.state_dict(),
                    'loss' : loss
                    }, "./model_state_dict.pt")
        

if __name__ == '__main__':
    main()