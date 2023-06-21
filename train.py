import torch
import numpy as np
from dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from discriminator import Discriminator
from generator import Generator
from torch import Tensor
from tqdm import tqdm
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.tensorboard import SummaryWriter
import config

######### FUNCTION FOR SAVE AND LOAD MODELS #########

def save_model(model, optimizer, epoch, filename="my_checkpoint.pth.tar"):
    print("Saving model for epoch : "+ str(epoch))
    
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, filename)


def load_model(file, model, optimizer, lr):
    print("Loading model: ")
    model_check = torch.load(file, map_location=config.DEVICE)
    model.load_state_dict(model_check["state_dict"])
    optimizer.load_state_dict(model_check["optimizer"])
    
    #epoch =model_check["epoch"]
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
        
######### END FUNCTION FOR MODELS ##########

def russo_gradient_penalty(model, real_images, fake_images, device):
    # Random weight term for interpolation between real and fake data
    alpha = torch.randn((real_images.size(0), 1, 1, 1), device=device)
    print(alpha)
    # Get random interpolation between real and fake data
    interpolates = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(True)

    model_interpolates = model(interpolates)
    grad_outputs = torch.ones(model_interpolates.size(), device=device, requires_grad=False)

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=model_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
    return gradient_penalty



                ########################### TRAIN FUNCTION #########################
def train_fn(disc_W, disc_S, gen_S, gen_W, loader, opt_disc, opt_gen, l1, mse, BCE, d_scaler, g_scaler,LAMBDA_IDENTITY, LAMBDA_CYCLE,GAMMA_CYCLE):
    
    loop = tqdm(loader, leave=True) #leave=True to avoid print newline
    
    # Loss weight for gradient penalty 
    LAMBDA_GP = 10

    for idx, (summer, winter) in enumerate(loop):
        summer = summer.to(config.DEVICE)
        winter = winter.to(config.DEVICE)
        
        
        # Ground truths used in the adversarial loss
        validwin = Variable(Tensor(winter.shape[0], 1,30,30).fill_(1.0), requires_grad=False)
        validsumm = Variable(Tensor(summer.shape[0], 1,30,30).fill_(1.0), requires_grad=False)
        
        fakewin = Variable(Tensor(winter.shape[0], 1,30,30).fill_(0.0), requires_grad=False)
        fakesumm = Variable(Tensor(summer.shape[0], 1,30,30).fill_(0.0), requires_grad=False)
        
        # label printed every epoch to see the prediction of the discriminators
        S_is_real = 0
        S_is_fake = 0
        W_is_real = 0
        W_is_fake = 0
        
    
        # Wasserstain variable #
        CRITIC_ITERATIONS = 5

        with torch.cuda.amp.autocast():
            
            validsumm = validsumm.to(config.DEVICE)
            fakesumm = fakesumm.to(config.DEVICE)
            validwin = validwin.to(config.DEVICE)
            fakewin = fakewin.to(config.DEVICE)
            
            ############## TRAIN DISCRIMINATOR SUMMER #############
            fake_summer = gen_S(winter)
            D_S_real = disc_S(summer)
            D_S_fake = disc_S(fake_summer.detach())
            #used to print the percentage that the given image is predicted real or fake
            S_is_real += D_S_real.mean().item() 
            S_is_fake += D_S_fake.mean().item()
            
            
            if(config.REL):
                if(config.BCE):
                    real_loss = BCE(D_S_real - D_S_fake, validsumm)
                    fake_loss = BCE(D_S_fake - D_S_real, fakesumm)
                    D_S_loss = (real_loss+fake_loss)/2
                else:
                    real_loss = mse(D_S_real - D_S_fake, validsumm)
                    fake_loss = mse(D_S_fake - D_S_real, fakesumm)
                    D_S_loss = (real_loss+fake_loss)/2
                
            elif(config.WAS):
                err_summer = torch.mean(D_S_real) #real
                err_fakesummer = torch.mean(D_S_fake) #fake
                
                gradient_penalty = russo_gradient_penalty(disc_S, summer, fake_summer , config.DEVICE)
                
                D_S_loss = -err_summer + err_fakesummer + gradient_penalty * 10  #lambda=10 written in the paper

            else:
                D_S_real_loss = mse(D_S_real, torch.ones_like(D_S_real))
                D_S_fake_loss = mse(D_S_fake, torch.zeros_like(D_S_fake))
                D_S_loss = D_S_real_loss + D_S_fake_loss
            
            ########### TRAIN DISCRIMINATOR WINTER ##############
            fake_winter = gen_W(summer)
            D_W_real = disc_W(winter)
            D_W_fake = disc_W(fake_winter.detach())
            #used print the percentage that the given image is predicted real or fake
            W_is_real += D_W_real.mean().item()
            W_is_fake += D_W_fake.mean().item()
            
            
            if(config.REL):                
                if(config.BCE):
                    real_loss = BCE(D_W_real - D_W_fake, validwin)
                    fake_loss = BCE(D_W_fake - D_W_real, fakewin)
                    D_W_loss = (real_loss+fake_loss)/2
                else:
                    real_loss = mse(D_W_real - D_W_fake, validwin)
                    fake_loss = mse(D_W_fake - D_W_real, fakewin)
                    D_W_loss = (real_loss+fake_loss)/2
                
            elif(config.WAS):
                err_winter = torch.mean(D_W_real) #real
                Win_x = D_W_real.mean().item()

                err_fakewinter = torch.mean(D_W_fake) #fake
                Win_xf = D_W_fake.mean().item()
                
                gradient_penalty = russo_gradient_penalty(disc_W, winter, fake_winter , config.DEVICE)
                
                D_W_loss = -err_winter + err_fakewinter + gradient_penalty * 10
                
                
            else:
            
                D_W_real_loss = mse(D_W_real, torch.ones_like(D_W_real))
                D_W_fake_loss = mse(D_W_fake, torch.zeros_like(D_W_fake))
                D_W_loss = D_W_real_loss + D_W_fake_loss

            

            # put togheter the loss of the two discriminators
            D_loss = (D_W_loss + D_S_loss)/2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward(retain_graph=True)
        d_scaler.step(opt_disc)
        d_scaler.update()

        ########################## TRAIN GENERATORS #########################
        with torch.cuda.amp.autocast():

            D_W_fake = disc_W(fake_winter)
            D_S_fake = disc_S(fake_summer)
            D_W_real = disc_W(winter)
            D_S_real = disc_S(summer)
            
            loss_G_W = 0
            loss_G_S = 0

            
            if(config.REL):
                if(config.BCE):
                    loss_G_W = BCE(D_W_fake - D_W_real, validwin)
                    loss_G_S = BCE(D_S_fake - D_S_real, validsumm)
                else:
                    loss_G_W = mse(D_W_fake - D_W_real, validwin)
                    loss_G_S = mse(D_S_fake - D_S_real, validsumm)
                    
                if(config.BETTER):
                    ################ BETTER CYCLE CONSISTENCY FOLLOWING THE REPORT TIPS #################
                    cycle_summer = gen_S(fake_winter)
                    x = disc_S(summer,feature_extract = True)
                    Fx = disc_S(cycle_summer,feature_extract = True)
                    norma_summer=l1(x,Fx)
                    cycle_summer_loss = l1(summer, cycle_summer)

                    cycle_winter = gen_W(fake_summer)
                    y = disc_W(winter,feature_extract = True)
                    Fy = disc_W(cycle_winter,feature_extract = True)
                    norma_winter=l1(y,Fy)
                    cycle_winter_loss = l1(winter, cycle_winter)

                    ################ BETTER CYCLE CONSISTENCY FOLLOWING THE REPORT TIPS #################
                    G_loss = (
                        loss_G_S
                        + loss_G_W
                        + torch.mean(disc_W(winter))*(GAMMA_CYCLE * norma_winter + (1-GAMMA_CYCLE) * cycle_winter_loss) * LAMBDA_CYCLE
                        + torch.mean(disc_S(summer))*(GAMMA_CYCLE * norma_summer+ (1-GAMMA_CYCLE) * cycle_summer_loss) * LAMBDA_CYCLE
                    )
                else:  
                    cycle_summer = gen_S(fake_winter)
                    cycle_winter = gen_W(fake_summer)
                    cycle_summer_loss = l1(summer, cycle_summer)
                    cycle_winter_loss = l1(winter, cycle_winter)

                    G_loss = (
                        loss_G_S
                        + loss_G_W
                        + cycle_summer_loss * LAMBDA_CYCLE
                        + cycle_winter_loss * LAMBDA_CYCLE
                    )
            
            elif(config.WAS):
                 # Train the generator every n_critic steps
                if idx % CRITIC_ITERATIONS == 0:
                    #gen_S.zero_grad()
                    #gen_W.zero_grad()
                    
                    loss_G_S = -torch.mean(D_S_fake)
                    loss_G_W = -torch.mean(D_W_fake)
                    
                    cycle_summer = gen_S(fake_winter)
                    cycle_winter = gen_W(fake_summer)
                    cycle_summer_loss = l1(summer, cycle_summer)
                    cycle_winter_loss = l1(winter, cycle_winter)

                    G_loss = (
                        loss_G_S
                        + loss_G_W
                        + cycle_summer_loss * LAMBDA_CYCLE
                        + cycle_winter_loss * LAMBDA_CYCLE
                    )

            
            else:

                loss_G_W = mse(D_W_fake, torch.ones_like(D_W_fake))
                loss_G_S = mse(D_S_fake, torch.ones_like(D_S_fake))
                if(config.BETTER):
                    ################ BETTER CYCLE CONSISTENCY FOLLOWING THE REPORT TIPS #################
                    cycle_summer = gen_S(fake_winter)
                    x = disc_S(summer,feature_extract = True)
                    Fx = disc_S(cycle_summer,feature_extract = True)
                    norma_summer=l1(x,Fx)
                    cycle_summer_loss = l1(summer, cycle_summer)

                    cycle_winter = gen_W(fake_summer)
                    y = disc_W(winter,feature_extract = True)
                    Fy = disc_W(cycle_winter,feature_extract = True)
                    norma_winter=l1(y,Fy)
                    cycle_winter_loss = l1(winter, cycle_winter)
                    
                    G_loss = (
                    loss_G_S
                    + loss_G_W
                    + torch.mean(disc_W(winter))*(GAMMA_CYCLE * norma_winter + (1-GAMMA_CYCLE) * cycle_winter_loss) * LAMBDA_CYCLE
                    + torch.mean(disc_S(summer))*(GAMMA_CYCLE * norma_summer+ (1-GAMMA_CYCLE) * cycle_summer_loss) * LAMBDA_CYCLE
                    )
                    ################ BETTER CYCLE CONSISTENCY FOLLOWING THE REPORT TIPS #################
                else:
                    cycle_summer = gen_S(fake_winter)
                    cycle_winter = gen_W(fake_summer)
                    cycle_summer_loss = l1(summer, cycle_summer)
                    cycle_winter_loss = l1(winter, cycle_winter)

                    #identity_s = gen_S(summer)
                    #identity_w = gen_W(winter)
                    #identity_loss_summer = l1(summer, identity_s)
                    #identity_loss_winter = l1(winter, identity_w)


                    # add all togethor
                    G_loss = (
                        loss_G_S
                        + loss_G_W
                        + cycle_summer_loss * LAMBDA_CYCLE
                        + cycle_winter_loss * LAMBDA_CYCLE
                        #+ identity_loss_winter * LAMBDA_IDENTITY
                        #+ identity_loss_summer * LAMBDA_IDENTITY
                    )


        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward(retain_graph=True)
        g_scaler.step(opt_gen)
        g_scaler.update()
        
        ##########################  END TRAIN GENERATORS #########################
        

        if idx % 150 == 0:    #save tensor into images every 150 to see in real time the progress of the net
            save_image(fake_summer*0.5+0.5, f"saved_images/summer_{idx}.png")
            save_image(fake_winter*0.5+0.5, f"saved_images/winter_{idx}.png")

        #set postfixes to the progess bar of tqdm
        loop.set_postfix(W_real=W_is_real/(idx+1), W_fake=W_is_fake/(idx+1),S_real=S_is_real/(idx+1), S_fake=S_is_fake/(idx+1))
        
                ########################### END TRAIN FUNCTION ######################

def test_fn_winter(gen_S,gen_W,test_loader):
    
    loop = tqdm(test_loader, leave=True)

    for idx, (summer, winter) in enumerate(loop):
        winter = winter.to(config.DEVICE)
        summer = summer.to(config.DEVICE)
        fake_summer = gen_S(winter)
        fake_winter = gen_W(fake_summer)
        
        save_image(winter*0.5+0.5, f"test_images/testoriginal_{idx}.png")
        save_image(fake_summer*0.5+0.5, f"test_images/testsummer_{idx}.png")
        save_image(fake_winter*0.5+0.5, f"test_images/testwinter_{idx}.png")
        
def test_fn_summer(gen_S,gen_W,test_loader):
    
    loop = tqdm(test_loader, leave=True)

    for idx, (summer, winter) in enumerate(loop):
        winter = winter.to(config.DEVICE)
        summer = summer.to(config.DEVICE)
        fake_winter = gen_W(summer)
        fake_summer = gen_S(fake_winter)
        
        save_image(summer*0.5+0.5, f"test_images/testoriginal_{idx}.png")
        save_image(fake_summer*0.5+0.5, f"test_images/testsummer_{idx}.png")
        save_image(fake_winter*0.5+0.5, f"test_images/testwinter_{idx}.png")
        
                        ###################### MAIN FUNCTION #######################
def main():
    disc_W = Discriminator(in_channels=3).to(config.DEVICE)
    disc_S = Discriminator(in_channels=3).to(config.DEVICE)
    gen_S = Generator(img_channels=3).to(config.DEVICE)
    gen_W = Generator(img_channels=3).to(config.DEVICE)
    
    opt_disc = optim.Adam(
        list(disc_W.parameters()) + list(disc_S.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_S.parameters()) + list(gen_W.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()
    BCE = torch.nn.BCEWithLogitsLoss()
    
    GAMMA_CYCLE = config.GAMMA_CYCLE # ratio between discriminator CNN feature level and pixel level loss
    
    if config.LOAD_MODEL:
        load_model(
            config.CHECKPOINT_GEN_W, gen_W, opt_gen, config.LEARNING_RATE,
        )
        load_model(
            config.CHECKPOINT_GEN_S, gen_S, opt_gen, config.LEARNING_RATE,
        )
        load_model(
            config.CHECKPOINT_DISC_W, disc_W, opt_disc, config.LEARNING_RATE,
        )
        load_model(
            config.CHECKPOINT_DISC_S, disc_S, opt_disc, config.LEARNING_RATE,
        )
    

    ############## CHOICE OF THE DATASET ###############
    if(config.DATASET_ORIGINAL):
        dataset = Dataset(
        winter_dir=config.TRAIN_DIR+"/winter", summer_dir=config.TRAIN_DIR+"/summer", transform=config.transforms
        )
        test_dataset = Dataset(
        winter_dir=config.TEST_DIR+"/winter1", summer_dir=config.TEST_DIR+"/summer1", transform=config.transforms
        )
    else:
        dataset = Dataset(
        winter_dir=config.TRAIN_DIR_ALT+"/winter", summer_dir=config.TRAIN_DIR_ALT+"/summer", transform=config.transforms
        )
        test_dataset = Dataset(
        winter_dir=config.TEST_DIR+"/winter1", summer_dir=config.TEST_DIR+"/summer1", transform=config.transforms
        )
     ############# CHOICE OF THE DATASET ##############
    
    ############# DATALOADER #############
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True  #for faster training(non-paged cpu memory)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    ############# DATALOADER ###############
    
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    
    if(config.TRAIN_MODEL):

        for epoch in range(config.NUM_EPOCHS):
            if(config.BETTER):
                train_fn(disc_W, disc_S, gen_S, gen_W, loader, opt_disc, opt_gen, L1, mse, BCE, d_scaler, g_scaler,config.LAMBDA_IDENTITY, config.LAMBDA_CYCLE-epoch*0.15,GAMMA_CYCLE=GAMMA_CYCLE+0.015)
            else:
                train_fn(disc_W, disc_S, gen_S, gen_W, loader, opt_disc, opt_gen, L1, mse, BCE, d_scaler, g_scaler,config.LAMBDA_IDENTITY, config.LAMBDA_CYCLE, 0)


            if config.SAVE_MODEL: #if save_Model is set to true save model on the specific path
                save_model(gen_W, opt_gen, epoch ,filename=config.CHECKPOINT_GEN_W)
                save_model(gen_S, opt_gen, epoch , filename=config.CHECKPOINT_GEN_S)
                save_model(disc_W, opt_disc, epoch , filename=config.CHECKPOINT_DISC_W)
                save_model(disc_S, opt_disc, epoch , filename=config.CHECKPOINT_DISC_S)
    else:
        
        test_fn_winter(gen_S,gen_W,test_loader)
        #test_fn_summer(gen_S,gen_W,test_loader)

if __name__ == "__main__":
    main()

