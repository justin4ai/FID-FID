import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import glob
from src import CustomDataLoader
from src.HighFreqVit import HighFreqVitClassifier
import argparse

def main(args):
    customloader = CustomDataLoader.DataProcesser()
    data = customloader.get_datasets(dataset_path = args.test_folder_name, train = False)
    batch_size = args.batch_size
    dataloader = DataLoader(dataset = data, batch_size = batch_size)
    num_datas = len(dataloader)*batch_size

    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    model = HighFreqVitClassifier()
    checkpoint = args.checkpoint
    checkpoint = glob.glob(args.checkpoint_path+"/*"+str(checkpoint)+".pt")
    print(checkpoint[0])
    checkpoint = torch.load(checkpoint[0])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    accuracy = 0
    predict = []
    ground_truth = []
    with torch.no_grad():
        for batch_num, batch in enumerate(tqdm(dataloader)):
            img, labels = batch
            labels = list(labels)
            
            raw_logits, logits, loss = model(img.to(device), labels, device = device)
            
            predict = predict + logits
            ground_truth = ground_truth + labels
            for idx in range(len(labels)):
                if logits[idx] == labels[idx]:
                    accuracy = accuracy + 1
        
    accuracy = accuracy/num_datas * 100
    print("Test Accuracy : ", accuracy)
    print("Predict : \n", np.array([predict, ground_truth]).T)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a HighFreqVit model")
    parser.add_argument('--test_folder_name', type=str, default = "./datasets", help="Path to the test dataset")
    parser.add_argument('--batch_size', type=int, default=16, help="Mini batch size")
    parser.add_argument('--checkpoint_path', type=str, default="./checkpoints", help="Which checkpoint to use")
    parser.add_argument('--checkpoint', type=int, default=50, help="Which checkpoint to use")

    args = parser.parse_args()

    main(args)