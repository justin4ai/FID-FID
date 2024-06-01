import glob
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src import CustomDataLoader
from src import HighFreqVit

def main():
    customloader = CustomDataLoader.DataProcesser()
    data = customloader.get_datasets(path = "./datasets")
    batch_size = 16
    dataloader = DataLoader(dataset = data, batch_size = batch_size, shuffle = True)
    num_datas = len(dataloader)*batch_size
    classifier = HighFreqVit.HighFreqVitClassifier()
    device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
    print(device, torch.cuda.get_device_name())
    classifier.to(device)
    num_epochs = 50
    optimizer = torch.optim.Adam(classifier.parameters(), lr = 0.01)
    use_checkpoint = True
    
    if use_checkpoint:
        if glob.glob("./checkpoints/*.pt"): 
            checkpoint_path = glob.glob("./checkpoints/*.pt") 
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
        split_val = int(len(dataloader) * 0.8)
        for batch_num, batch in enumerate(tqdm(dataloader)):
            if batch_num < split_val:
                optimizer.zero_grad()
                
                img, labels = batch
                labels = list(labels)
                
                logits, loss = classifier(img.to(device), labels, device)
                
                for idx in range(len(labels)):
                    if labels[idx] == logits[idx]:
                        train_acc = train_acc + 1
                
                if train_loss is None:
                    train_loss = loss
                else:
                    train_loss = torch.mean(torch.stack([train_loss, loss]))
                
                if batch_num%10 == 0:
                    print(f"\nTraining accuracy : {train_acc}/{(batch_num + 1) * batch_size},\tTrining Loss : {train_loss}")
                
                loss.backward()
                optimizer.step()
            
            else:
                img, labels = batch
                labels = list(labels)
                
                logits, loss = classifier(img.to(device), labels, device)
                
                for idx in range(len(labels)):
                    if labels[idx] == logits[idx]:
                        validation_acc = validation_acc + 1
                
                if validation_loss is None:
                    validation_loss = loss
                else:
                    validation_loss = torch.mean(torch.stack([validation_loss, loss]))
                    
                if batch_num%10 == 0:
                    print(f"\nValidation accuracy : {validation_acc}/{(batch_num - split_val) * batch_size},\tValidation Loss : {validation_loss}")
            
            
        train_acc = train_acc/(split_val * batch_size) * 100
        validation_acc = validation_acc/((len(dataloader) - split_val) * batch_size) * 100
        print(f"\nEpoch : {epoch}")
        print(f"Training accuracy : {train_acc:.3f}%,\tTraining Loss : {train_loss.item():.5f}")
        print(f"Validation accuracy : {validation_acc:.3f}%,\tValidation Loss : {validation_loss.item():.5f}")
        

        if epoch % 10 == 0:
            torch.save({
                        'epoch' : epoch, 
                        'model_state_dict' : classifier.state_dict(),
                        'optimizer_state_dict' : optimizer.state_dict(),
                        'loss' : loss
                        }, f"./checkpoints/detector_{epoch}.pt")
        

if __name__ == '__main__':
    main()