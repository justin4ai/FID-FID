import glob
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from src import CustomDataLoader
from src import FreqEncoder
import argparse
import time

def main(args):

    customloader = CustomDataLoader.DataProcesser()
    data = customloader.get_datasets(dataset_path = "./datasets/train/", real_folder_name = args.real_folder_name, fake_folder_name = args.fake_folder_name)
    num_datas = len(data)

    batch_size = args.batch_size
    train_dataloader = DataLoader(dataset = data, batch_size = batch_size, shuffle = True)

    model = FreqEncoder.HighPass()
    device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
    print(device, torch.cuda.get_device_name())
    model.to(device)

    for train_idx, train_batch in enumerate(train_dataloader):
        
        img, labels = train_batch
        labels = list(labels)
        print(labels)
        high_passed = model(img.to(device))
        
            
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a HighFreqVit model")
    parser.add_argument('--real_folder_name', type=str, default = "real", help="Path to the dataset")
    parser.add_argument('--fake_folder_name', type=str, default = "generated", help="Path to the dataset")
    parser.add_argument('--test_folder_name', type=str, default = "./datasets", help="Path to the test dataset")
    parser.add_argument('--batch_size', type=int, default=16, help="Mini batch size")
    args = parser.parse_args()

    main(args)