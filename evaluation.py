import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob
from src import CustomDataLoader
from src.HighFreqVit import HighFreqVitClassifier
import argparse

def main(args):
    customloader = CustomDataLoader.DataProcesser()
    data = customloader.get_datasets(path = args.test_folder_name, train = False)
    batch_size = args.batch_size
    dataloader = DataLoader(dataset = data, batch_size = batch_size)
    num_datas = len(dataloader)*batch_size
    
    model = HighFreqVitClassifier()
    checkpoint = args.checkpoint
    checkpoint = glob.glob("./checkpoints/*{checkpoint}.pt")
    checkpoint = torch.load(checkpoint)
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
    parser = argparse.ArgumentParser(description="Train a HighFreqVit model")
    parser.add_argument('--real_folder_name', type=str, default = "real", help="Path to the train dataset")
    parser.add_argument('--fake_folder_name', type=str, default = "generated", help="Path to the train dataset")
    parser.add_argument('--test_folder_name', type=str, default = "./datasets/test", help="Path to the test dataset")
    parser.add_argument('--save_path', type=str, default = "./checkpoints/", help="Path to the checkpoint dataset")
    parser.add_argument('--batch_size', type=int, default=16, help="Mini batch size")
    parser.add_argument('--checkpoint', type=int, default=50, help="Which checkpoint to use")
    
    args = parser.parse_args()

    main(args)