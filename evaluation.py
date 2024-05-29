import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src import CustomDataLoader
from src import HighFreqVit

def main():
    customloader = CustomDataLoader.DataProcesser()
    data = customloader.get_datasets(path = "./datasets", train = False)
    batch_size = 50
    dataloader = DataLoader(dataset = data, batch_size = batch_size)
    num_datas = len(dataloader)*batch_size
    
    model = HighFreqVit.HighFreqVitClassifier()
    checkpoint = torch.load("./model_state_dict.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    accuracy = 0
    for batch_num, batch in enumerate(tqdm(dataloader)):
        img, labels = batch
        labels = list(labels)
        
        logits, loss = model(img, labels)
        
        for idx in range(len(labels)):
            print("Predict : ", logits[idx], ", GT : ", labels[idx])
            if logits[idx] == labels[idx]:
                accuracy = accuracy + 1
        
    accuracy = accuracy/num_datas * 100
    print("Test Accuracy : ", accuracy)

if __name__ == '__main__':
    main()