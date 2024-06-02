import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob
from src import CustomDataLoader
from src import HighFreqVit

def main():
    customloader = CustomDataLoader.DataProcesser()
    data = customloader.get_datasets(path = "./datasets", train = False)
    batch_size = 16
    dataloader = DataLoader(dataset = data, batch_size = batch_size)
    num_datas = len(dataloader)*batch_size
    
    model = HighFreqVit.HighFreqVitClassifier()
    checkpoint_list = glob.glob("./checkpoints/*.pt")
    latest_checkpoint = checkpoint_list.pop()
    checkpoint = torch.load(latest_checkpoint)
    print(latest_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    accuracy = 0
    predict = []
    ground_truth = []
    for batch_num, batch in enumerate(tqdm(dataloader)):
        img, labels = batch
        labels = list(labels)
        
        logits, loss = model(img, labels)
        
        predict = predict + logits
        ground_truth = ground_truth + labels
        for idx in range(len(labels)):
            if logits[idx] == labels[idx]:
                accuracy = accuracy + 1
        
    accuracy = accuracy/num_datas * 100
    print("Test Accuracy : ", accuracy)
    print("Predict : \n", predict)
    print("Ground Truth : \n", ground_truth)

if __name__ == '__main__':
    main()