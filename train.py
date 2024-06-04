import glob
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from src import CustomDataLoader
from src import HighFreqVit
import argparse
import time

def main(args):
    customloader = CustomDataLoader.DataProcesser()
    data = customloader.get_datasets(dataset_path = "./datasets/train/", real_folder_name = args.real_folder_name, fake_folder_name = args.fake_folder_name)
    num_datas = len(data)
    train_size = int(num_datas * 0.8)
    validation_size = int(num_datas - train_size)
    train_data, validation_data = random_split(data, [train_size, validation_size])
    batch_size = args.batch_size
    train_dataloader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)
    validation_dataloader = DataLoader(dataset = validation_data, batch_size = batch_size, shuffle = True)
    classifier = HighFreqVit.HighFreqVitClassifier()
    device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
    print(device, torch.cuda.get_device_name())
    classifier.to(device)
    num_epochs = args.num_epochs
    optimizer = torch.optim.Adam(classifier.parameters(), lr = 0.01)
    use_checkpoint = args.use_checkpoint
    
    if use_checkpoint:
        if glob.glob(args.save_path + "*.pt"): 
            checkpoint_path = glob.glob(args.save_path + "*.pt") 
            checkpoint = torch.load(checkpoint_path.pop())

            classifier.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            checkpoint_epoch = checkpoint['epoch']
            loss = checkpoint['loss']
        
    for epoch in range(1, num_epochs + 1):
        
        if use_checkpoint and (epoch <= checkpoint_epoch):
            continue
        
        train_acc = 0
        train_loss = None
        validation_acc = 0
        validation_loss = None
        for train_idx, train_batch in enumerate(train_dataloader):
            start_time = time.time()
            optimizer.zero_grad()
            
            img, labels = train_batch
            labels = list(labels)
            
            logits, loss = classifier(img.to(device), labels, device)
            
            for idx in range(len(labels)):
                if labels[idx] == logits[idx]:
                    train_acc = train_acc + 1
            
            if train_loss is None:
                train_loss = loss
            else:
                train_loss = torch.mean(torch.stack([train_loss, loss]))
            
            if train_idx%10 == 0:
                print(f"\nEpoch {epoch} - {train_idx}/{len(train_dataloader)}th iteration : Training accuracy : {(train_acc/((train_idx + 1) * batch_size))*100:.2f}%,\tTraining Loss : {train_loss}")
                print(f"Running time : {(time.time() - start_time):.2f}")
                start_time = time.time()
                
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():    
            for val_idx, val_batch in enumerate(tqdm(validation_dataloader)):
                img, labels = val_batch
                labels = list(labels)
                
                logits, loss = classifier(img.to(device), labels, device)
                
                for idx in range(len(labels)):
                    if labels[idx] == logits[idx]:
                        validation_acc = validation_acc + 1
                
                if validation_loss is None:
                    validation_loss = loss
                else:
                    validation_loss = torch.mean(torch.stack([validation_loss, loss]))
                    
                if val_idx%10 == 0:
                    print(f"\nValidation accuracy : {validation_acc}/{(val_idx + 1) * batch_size},\tValidation Loss : {validation_loss}")
            
            
        train_acc = train_acc/(len(train_dataloader) * batch_size) * 100
        validation_acc = validation_acc/(len(validation_dataloader) * batch_size) * 100
        print(f"\nEpoch : {epoch}")
        print(f"Training accuracy : {train_acc:.3f}%,\tTraining Loss : {train_loss.item():.5f}")
        print(f"Validation accuracy : {validation_acc:.3f}%,\tValidation Loss : {validation_loss.item():.5f}\n")
        

        if epoch % 10 == 0:
            torch.save({
                        'epoch' : epoch, 
                        'model_state_dict' : classifier.state_dict(),
                        'optimizer_state_dict' : optimizer.state_dict(),
                        'loss' : loss
                        }, args.save_path + f"detector_{epoch}.pt")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a HighFreqVit model")
    parser.add_argument('--real_folder_name', type=str, default = "real", help="Path to the dataset")
    parser.add_argument('--fake_folder_name', type=str, default = "generated", help="Path to the dataset")
    parser.add_argument('--save_path', type=str, default = "./checkpoints/", help="Path to the dataset")
    parser.add_argument('--num_epochs', type=int, default=50, help="Number of epochs to train")
    parser.add_argument('--batch_size', type=int, default=16, help="Mini batch size")
    parser.add_argument('--use_checkpoint', type=bool, default=False, help="whether to use checkpoints or not")
    
    args = parser.parse_args()

    main(args)