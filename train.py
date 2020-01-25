# Basile Van Hoorick, Jan 2020
'''
Edit the paths here and run to train the GAN.
Uses GPU:0 with CUDA (feel free to switch to CPU or use DataParallel).
'''

if __name__ == '__main__':

    import torch
    from outpainting import *

    print("PyTorch version: ", torch.__version__)
    print("Torchvision version: ", torchvision.__version__)

    # Define paths
    model_save_path = 'outpaint_models'
    html_save_path = 'outpaint_html'
    train_dir = 'train'
    val_dir = 'val'
    test_dir = 'test'

    # Define datasets & transforms
    my_tf = transforms.Compose([
            transforms.Resize(output_size),
            transforms.CenterCrop(output_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
    batch_size = 4
    train_data = CEImageDataset(train_dir, my_tf, output_size, input_size, outpaint=True)
    val_data = CEImageDataset(val_dir, my_tf, output_size, input_size, outpaint=True)
    test_data = CEImageDataset(test_dir, my_tf, output_size, input_size, outpaint=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=1)
    print('train:', len(train_data), 'val:', len(val_data), 'test:', len(test_data))

    # Define model & device
    device = torch.device('cuda:0')
    # device = torch.device('cpu')
    G_net = CEGenerator(extra_upsample=True)
    D_net = CEDiscriminator()
    G_net.apply(weights_init_normal)
    D_net.apply(weights_init_normal)
    # G_net = nn.DataParallel(G_net)
    # D_net = nn.DataParallel(D_net)
    G_net.to(device)
    D_net.to(device)
    print('device:', device)

    # Define losses
    criterion_pxl = nn.L1Loss()
    criterion_D = nn.MSELoss()
    optimizer_G = optim.Adam(G_net.parameters(), lr=3e-4, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D_net.parameters(), lr=3e-4, betas=(0.5, 0.999))
    criterion_pxl.to(device)
    criterion_D.to(device)

    # Start training
    data_loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader} # NOTE: test is evidently not used by the train method
    n_epochs = 200
    adv_weight = [0.001, 0.005, 0.015, 0.040] # corresponds to epochs 1-10, 10-30, 30-60, 60-onwards
    hist_loss = train_CE(G_net, D_net, device, criterion_pxl, criterion_D, optimizer_G, optimizer_D,
                         data_loaders, model_save_path, html_save_path, n_epochs=n_epochs, outpaint=True, adv_weight=adv_weight)

    # Save loss history and final generator
    pickle.dump(hist_loss, open('hist_loss.p', 'wb'))
    torch.save(G_net.state_dict(), 'generator_final.pt')

    # Next steps: see forward.py
