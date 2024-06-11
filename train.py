import glob
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from src import CustomDataLoader
from src import HighFreqVit
import argparse
import time
# import torch.autograd.profiler as profiler

def main(args):

    lr = args.learning_rate
    decay = args.weight_decay
    base_folder_real = args.base_folder_real
    base_folder_fake = args.base_folder_fake




    customloader = CustomDataLoader.DataProcesser()
    data = customloader.get_datasets(base_folder_real = base_folder_real, base_folder_fake = base_folder_fake, base_folder_test = args.test_folder,
                                      real_folder_name = args.real_folder_name, fake_folder_name = args.fake_folder_name)
    test_data = customloader.get_datasets(base_folder_test = args.test_folder, train = False)
    num_datas = len(data)
    train_size = int(num_datas * 0.9)
    validation_size = int(num_datas - train_size)
    train_data, validation_data = random_split(data, [train_size, validation_size])
    test_data, _ = random_split(test_data, [int(len(test_data) * 0.1), int(len(test_data)*0.9)])
    
    batch_size = args.batch_size
    train_dataloader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)
    validation_dataloader = DataLoader(dataset = validation_data, batch_size = batch_size, shuffle = True)
    test_dataloader = DataLoader(dataset = test_data, batch_size = batch_size, shuffle = True )
    
    classifier = HighFreqVit.HighFreqVitClassifier()
    device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
    print(device, torch.cuda.get_device_name())
    classifier.to(device)
    
    num_epochs = args.num_epochs
    optimizer = torch.optim.Adam(classifier.parameters(), lr = lr, weight_decay = decay)
    use_checkpoint = args.use_checkpoint
    
    if use_checkpoint:
        if glob.glob(args.save_path + "*.pt"): 
            checkpoint_path = glob.glob(args.save_path + "*.pt") 
            checkpoint = torch.load(checkpoint_path.pop())

            classifier.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            checkpoint_epoch = checkpoint['epoch']
            loss = checkpoint['loss']
    
    
    for epoch in range(1, num_epochs+1):
        if use_checkpoint and (epoch <= checkpoint_epoch):
            continue
        
        start_time = time.time()
        train_acc = 0
        train_loss = None
        test_interval = int(len(train_dataloader)/args.test_interval)
        validation_acc = 0
        validation_loss = None
        for train_idx, train_batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            
            img, labels = train_batch
            labels = list(labels)
            
            # with profiler.profile(with_stack = True, use_cuda=True, profile_memory=True) as prof:
            raw_logits, logits, loss = classifier(img.to(device), labels, device)
            # print(prof.key_averages(group_by_stack_n = 5).table(sort_by='cuda_time_total', row_limit=5))
            
            for idx in range(len(labels)):
                if labels[idx] == logits[idx]:
                    train_acc = train_acc + 1
            
            if train_loss is None:
                train_loss = loss
            else:
                train_loss = torch.mean(torch.stack([train_loss, loss]))
            
            if train_idx%10 == 0:
                print(f"\nEpoch {epoch} - {train_idx}/{len(train_dataloader)}th iteration : ")
                print(f"Training accuracy : {(train_acc/((train_idx + 1) * batch_size))*100:.2f}%,\tTraining Loss : {train_loss}")
                print(f"Runtime : {(time.time() - start_time):.2f}sec")
                start_time = time.time()
                
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():    
            for val_idx, val_batch in enumerate(validation_dataloader):
                img, labels = val_batch
                labels = list(labels)
                
                raw_logits, logits, loss = classifier(img.to(device), labels, device)
                
                for idx in range(len(labels)):
                    if labels[idx] == logits[idx]:
                        validation_acc = validation_acc + 1
                
                if validation_loss is None:
                    validation_loss = loss
                else:
                    validation_loss = torch.mean(torch.stack([validation_loss, loss]))
                    
                if val_idx%10 == 0:
                    print(f"\nEpoch {epoch} - {val_idx}/{len(validation_dataloader)}th iteration : ")
                    print(f"Validation accuracy : {(validation_acc/((val_idx + 1) * batch_size)) * 100:.2f},\tValidation Loss : {validation_loss}")
            
            
        train_acc = train_acc/(len(train_dataloader) * batch_size) * 100
        validation_acc = validation_acc/(len(validation_dataloader) * batch_size) * 100
        print(f"\nEpoch : {epoch}")
        print(f"Training accuracy : {train_acc:.3f}%,\tTraining Loss : {train_loss.item():.5f}")
        print(f"Validation accuracy : {validation_acc:.3f}%,\tValidation Loss : {validation_loss.item():.5f}\n")
        

        if epoch % 20 == 0:
            torch.save({
                        'epoch' : epoch, 
                        'model_state_dict' : classifier.state_dict(),
                        'optimizer_state_dict' : optimizer.state_dict(),
                        'loss' : loss
                        }, args.save_path + f"detector_{epoch}.pt")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a HighFreqVit model")
    parser.add_argument('--base_folder_real', type=str, default = "./datasets/train/", help="Path to the dataset")
    parser.add_argument('--base_folder_fake', type=str, default = "./datasets/train/", help="Path to the dataset")
    parser.add_argument('--base_folder_test', type=str, default = "./datasets/test/", help="Path to the dataset")
    parser.add_argument('--real_folder_name', type=str, default = "/real", help="Path to the dataset")
    parser.add_argument('--fake_folder_name', type=str, default = "/generated", help="Path to the dataset")
    parser.add_argument('--test_folder', type=str, default = "./datasets", help="Path to the test dataset")
    parser.add_argument('--save_path', type=str, default = "./checkpoints/", help="Path to the dataset")
    parser.add_argument('--num_epochs', type=int, default=50, help="Number of epochs to train")
    parser.add_argument('--batch_size', type=int, default=16, help="Mini batch size")
    parser.add_argument('--test_interval', type=int, default=10, help="interval of test session during training")
    parser.add_argument('--use_checkpoint', type=bool, default=False, help="whether to use checkpoints or not")
    parser.add_argument('--learning_rate', type=float, default=0.005, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.005, help="Weight decay")
    args = parser.parse_args()

    main(args)