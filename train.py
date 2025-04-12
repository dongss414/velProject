from typing import Dict
import time,os,logging,torch
from data import TrainVelocityMulFrameSynthesisDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from model import UNet3D_trainVelocity
import torch.nn as nn
import torch.nn.functional as F
from render import render_density
from torchvision.utils import save_image
from advection import *
from tqdm import tqdm

velpath = './data/velocity'
denpath = './data/density'


def setup_logger(log_path):
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

class train_vel_model_mulFrame(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def computeDivergence(self,vel):
        device = vel.device
        dtype = vel.dtype
        # vel [1,3,64,64,64]
        dx = dy = dz = 1.0
        kernel_x = torch.tensor([-1., 0., 1.],dtype=dtype,device=device).view(1, 1, 3, 1, 1)
        kernel_y = torch.tensor([-1., 0., 1.],dtype=dtype,device=device).view(1, 1, 1, 3, 1)
        kernel_z = torch.tensor([-1., 0., 1.],dtype=dtype,device=device).view(1, 1, 1, 1, 3)

        div_x = F.conv3d(vel[0,0].unsqueeze(0).unsqueeze(0), kernel_x, padding=(1, 0, 0)).squeeze()
        div_y = F.conv3d(vel[0,1].unsqueeze(0).unsqueeze(0), kernel_y, padding=(0, 1, 0)).squeeze()
        div_z = F.conv3d(vel[0,2].unsqueeze(0).unsqueeze(0), kernel_z, padding=(0, 0, 1)).squeeze()

        # divergence = div_x + div_y + div_z
        divergence = torch.abs(div_x) + torch.abs(div_y) + torch.abs(div_z)
        return torch.mean(divergence)

    def forward(self, den, vel_ref):
        weight = [1,0.4,0.3,0.2,0.1]
        d1,d2,d3,d4,d5 = self.model(den)
        sample_vel = d1

        outputs = [d1, d2, d3, d4, d5]
        vel_loss_list = [F.mse_loss(output, vel_ref) for output in outputs]
        loss = [F.mse_loss(output, vel_ref) * w for output, w in zip(outputs, weight)]

        sample_div = self.computeDivergence(d1)
        true_div = self.computeDivergence(vel_ref)

        div_loss = torch.abs(sample_div-true_div)
        vel_loss = sum(loss) + div_loss

        return vel_loss, div_loss, vel_loss_list, sample_vel
def train_or_val_mulFrame(logger,modelConfig,model, dataloader, trainer, optimizer=None, is_train=True, device="cuda:0",epoch=0,path=None):
    if is_train:
        model.train()
    else:
        model.eval()

    # res = [64, 64, 64]
    # source = smoke_source(res=res).to(device)

    epoch_loss = 0.0
    epoch_advected_loss = 0.0
    epoch_l1_density_loss = 0.0
    epoch_image_loss = 0.0
    epoch_image_l1_loss = 0.0
    epoch_v1_loss = 0.0
    epoch_v2_loss = 0.0
    epoch_v3_loss = 0.0
    epoch_v4_loss = 0.0
    epoch_v5_loss = 0.0
    epoch_div_loss = 0.0
    count = 0


    def render(den, light_positions=[(64, 64, -64)], intensity_p=0, intensity_a=1, isSample=True, flag=False):
        w = 0.2
        front_view = render_density(den * w, light_positions, intensity_p, intensity_a, n=0,
                                                              f=96, rotation_angle=90)
        side_view = render_density(den * w, light_positions, intensity_p, intensity_a, n=0,
                                                              f=96, rotation_angle=180)
        return front_view, side_view

    def expand(front,side):
        _,channels,_,width = front.shape
        out1 = front.unsqueeze(2).repeat(1,1,width,1,1)
        out2 = side.unsqueeze(-1).repeat(1,1,1,1,width)
        return out1,out2
    def img_loss(den_sample,front_image,side_image,count,trainStage=True):
        sample_front_image, sample_side_image = render(den_sample[0, 0, :, :, :], isSample=True, flag=False,
                                                       light_positions=[(modelConfig['den_size'][0], modelConfig['den_size'][1], -modelConfig['den_size'][2])])

        loss = F.mse_loss(front_image,sample_front_image) + F.mse_loss(side_image,sample_side_image)
        img_l1_loss = F.l1_loss(front_image,sample_front_image) + F.l1_loss(side_image,sample_side_image)
        if count%100 == 0:
            combined_image = torch.cat([front_image, sample_front_image,side_image,sample_side_image], dim=-1)
            combined_image = torch.rot90(combined_image,k=2,dims=[0,1])
            combined_image = F.interpolate(combined_image.unsqueeze(0).unsqueeze(0), size=(4*modelConfig['img_size'][0], 4*modelConfig['img_size'][1]*4), mode='bilinear',
                                           align_corners=False).squeeze(0)
            if trainStage :
                save_image(combined_image, os.path.join(path, f"trainSample{count:06d}.png"), nrow=1)
            else:
                save_image(combined_image, os.path.join(path, f"valSample{count:06d}.png"), nrow=1)
        return loss,img_l1_loss



    with tqdm(dataloader, dynamic_ncols=True, disable=False) as tqdmDataLoader:
        for batch in tqdmDataLoader:
            count += 1
            if is_train:
                optimizer.zero_grad()

            denList = [den.to(device).float() for den in batch['denList']]
            velList = [vel.to(device) for vel in batch['velList']]
            source = batch['src'][0].float().to(device)
            imageList = []
            sampleDenList = []
            for i in range(modelConfig['velNum'] + 1):
                front_image, side_image = render(denList[i][0, 0, :, :, :], isSample=False,
                                                 flag=batch['isSF'][0].item(),light_positions=[(modelConfig['den_size'][0], modelConfig['den_size'][1], -modelConfig['den_size'][2])])
                imageList.append(front_image)
                imageList.append(side_image)


            sampleDenList = denList
                # print(len(sampleDenList))

            with torch.set_grad_enabled(is_train):
            # with torch.no_grad():
                velLoss = 0.0
                denLoss = 0.0
                denl1Loss = 0.0
                imageLoss = 0.0
                imageL1Loss = 0.0
                divLoss = 0.0
                advectedList = []
                for i in range(modelConfig['velNum']):
                    if i == 0:
                        den0 = sampleDenList[i]  #rho0,
                        den1 = sampleDenList[i+1]    #rho1
                        den = torch.cat((den0,den1),dim=1)
                        loss,div_loss,list,sample_vel = trainer(den, velList[i])  #u0
                        velLoss = velLoss + loss
                        divLoss = divLoss + div_loss
                    else:
                        if i == 1 :
                            den0 = trainAdvection(sampleDenList[i - 1][0, 0], sample_vel[0], source)   #rho0,u0 -> rho'1
                        else:
                            den0 = trainAdvection(den0[0, 0].clone(), sample_vel[0], source)  # rho'1,u1 -> rho'2

                        den0 = set_zero_border(den0).unsqueeze(0).unsqueeze(0)  # rho'1/rho'2
                        advectedList.append(den0)
                        loss = F.mse_loss(den0, sampleDenList[i])
                        l1Loss = F.l1_loss(den0, sampleDenList[i])
                        # if modelConfig['useGrho']:
                        imgLoss,img_l1loss = img_loss(den0, imageList[2*i],imageList[2*i+1],count,is_train)
                        imageLoss = imageLoss + imgLoss
                        imageL1Loss = imageL1Loss + img_l1loss
                        denLoss = denLoss + loss
                        denl1Loss = denl1Loss + l1Loss

                        den1 = sampleDenList[i + 1]  #rho2/rho3
                        den = torch.cat((den0, den1), dim=1)
                        loss, div_loss, list, sample_vel = trainer(den, velList[i])    #u1/u2
                        velLoss = velLoss + loss
                        divLoss = divLoss + div_loss
                # print(den0.shape)
                den0 = trainAdvection(den0[0, 0].clone(), sample_vel[0], source) #rho'2,u2 -> rho'3
                den0 = set_zero_border(den0).unsqueeze(0).unsqueeze(0)  # rho'3
                advectedList.append(den0)
                loss = F.mse_loss(den0, sampleDenList[-1])
                l1Loss = F.l1_loss(den0, sampleDenList[-1])
                # if modelConfig['useGrho']:
                imgLoss,img_l1loss = img_loss(den0, imageList[-2],imageList[-1],count,is_train)
                imageLoss = imageLoss + imgLoss
                imageL1Loss = imageL1Loss + img_l1loss
                denLoss = denLoss + loss
                denl1Loss = denl1Loss + l1Loss

                # denL1Loss = denL1Loss + l1_loss

                allLoss = (modelConfig['weight'][0] * velLoss / modelConfig['velNum'] +
                           modelConfig['weight'][1] * denLoss / modelConfig['velNum'] +
                           modelConfig['weight'][2] * imageLoss / modelConfig['velNum']+
                           modelConfig['weight'][3] * denl1Loss / modelConfig['velNum']+
                           modelConfig['weight'][4] * imageL1Loss / modelConfig['velNum']+
                           divLoss / modelConfig['velNum']

                           )

                if is_train:
                    # allLoss = allLoss.float()
                    allLoss.backward()
                    optimizer.step()

            epoch_loss += allLoss.sum().item()
            epoch_advected_loss += denLoss.item()/ modelConfig['velNum']
            epoch_image_loss += imageLoss.item()/ modelConfig['velNum']
            epoch_image_l1_loss += imageL1Loss.item()/ modelConfig['velNum']
            epoch_l1_density_loss += denl1Loss.item() / modelConfig['velNum']
            # if not modelConfig['trainscalar']:
            epoch_v1_loss += list[0].item()
            epoch_v2_loss += list[1].item()
            epoch_v3_loss += list[2].item()
            epoch_v4_loss += list[3].item()
            epoch_v5_loss += list[4].item()
            epoch_div_loss += divLoss / modelConfig['velNum']
            tqdmDataLoader.set_postfix(ordered_dict={
                "epoch": epoch,
                "loss: ": epoch_loss / count,
                "LR": optimizer.state_dict()['param_groups'][0]["lr"] if is_train else "N/A"
            })

    avg_epoch_loss = epoch_loss / count
    avg_epoch_advected_loss = epoch_advected_loss / count
    avg_epoch_image_loss = epoch_image_loss/ count
    avg_epoch_image_l1_loss = epoch_image_l1_loss/ count
    avg_epoch_density_l1_loss = epoch_l1_density_loss / count
    avg_epoch_div_loss = epoch_div_loss /count
    avg_epoch_v1_loss = epoch_v1_loss / count
    avg_epoch_v2_loss = epoch_v2_loss / count
    avg_epoch_v3_loss = epoch_v3_loss / count
    avg_epoch_v4_loss = epoch_v4_loss / count
    avg_epoch_v5_loss = epoch_v1_loss / count

    if is_train:
        logger.info(f"Epoch {epoch}: Average Train Loss: {avg_epoch_loss}")
        logger.info(f"Epoch {epoch}: Average Train Advected Loss: {avg_epoch_advected_loss}")
        logger.info(f"Epoch {epoch}: Average Train Density L1 Loss: {avg_epoch_density_l1_loss}")
        logger.info(f"Epoch {epoch}: Average Train Image Loss: {avg_epoch_image_loss}")
        logger.info(f"Epoch {epoch}: Average Train Image L1 Loss: {avg_epoch_image_l1_loss}")

        logger.info(f"Epoch {epoch}: Average Train Div Loss: {avg_epoch_div_loss}")
        logger.info(f"Epoch {epoch}: Average v1 loss: {avg_epoch_v1_loss}")
        logger.info(f"Epoch {epoch}: Average v2 loss: {avg_epoch_v2_loss}")
        logger.info(f"Epoch {epoch}: Average v3 loss: {avg_epoch_v3_loss}")
        logger.info(f"Epoch {epoch}: Average v4 loss: {avg_epoch_v4_loss}")
        logger.info(f"Epoch {epoch}: Average v5 loss: {avg_epoch_v5_loss}")
        # logger.info()
    else:
        logger.info(f"Epoch {epoch}: Average Val Loss: {avg_epoch_loss}")
        logger.info(f"Epoch {epoch}: Average Val Advected Loss: {avg_epoch_advected_loss}")
        logger.info(f"Epoch {epoch}: Average Val Density L1 Loss: {avg_epoch_density_l1_loss}")
        logger.info(f"Epoch {epoch}: Average Val Image Loss: {avg_epoch_image_loss}")
        logger.info(f"Epoch {epoch}: Average Val Image L1 Loss: {avg_epoch_image_l1_loss}")

        logger.info(f"Epoch {epoch}: Average Val Div Loss: {avg_epoch_div_loss}")
        logger.info(f"Epoch {epoch}: Val v1 loss: {avg_epoch_v1_loss}")
        logger.info(f"Epoch {epoch}: Val v2 loss: {avg_epoch_v2_loss}")
        logger.info(f"Epoch {epoch}: Val v3 loss: {avg_epoch_v3_loss}")
        logger.info(f"Epoch {epoch}: Val v4 loss: {avg_epoch_v4_loss}")
        logger.info(f"Epoch {epoch}: Val v5 loss: {avg_epoch_v5_loss}")


    return avg_epoch_loss

def train(modelConfig: Dict):
    start_time = time.time()
    savepath = f'{modelConfig["save_vel_weight_dir"]}/{time.strftime("%Y%m%d_%H%M%S", time.localtime(start_time))}'
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    log_filename = f'logfile.log'
    log_path = os.path.join(savepath, log_filename)
    logger = setup_logger(log_path)
    device = torch.device(modelConfig["device"])


    mydata = TrainVelocityMulFrameSynthesisDataset(modelConfig, velpath=velpath, denpath=denpath, use_train=True,scenenum=81,
                                           imagetransform=transforms.Compose([
                                               # transforms.ToTensor(),
                                               transforms.Resize(modelConfig["img_size"]),
                                           ]), )

    dataloader = DataLoader(mydata, batch_size=modelConfig["batch_size"], shuffle=True)

    valdata = TrainVelocityMulFrameSynthesisDataset(modelConfig, velpath=velpath, denpath=denpath, use_train=False,scenenum=19,
                                                    imagetransform=transforms.Compose([
                                                        # transforms.ToTensor(),
                                                        transforms.Resize(modelConfig["img_size"]),
                                                    ]), )

    val_loader = DataLoader(valdata, batch_size=modelConfig["batch_size"], shuffle=False)

    model = UNet3D_trainVelocity(inchannels=2).to(device)
    if modelConfig["training_vel_load_weight"] is not None:
        model.load_state_dict(
            torch.load(os.path.join(modelConfig["save_vel_weight_dir"], modelConfig["training_vel_load_weight"]),
                       map_location=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=3, min_lr=1e-5,verbose=True)
    trainer = train_vel_model_mulFrame(model).to(device)

    min_avg_val_loss = 1e5
    for e in range(modelConfig['start_epoch'],modelConfig["epoch"]):
        folder_name = f'ablation_trainVelocityMulFrame_randomSynthesis_{e}'
        new = os.path.join(savepath, folder_name)
        # resize = transforms.Resize(modelConfig["img_size"])
        if not os.path.exists(new):
            os.mkdir(new)
        start_time = time.time()
        train_loss = train_or_val_mulFrame(logger,modelConfig, model, dataloader, trainer, optimizer=optimizer, is_train=True,
                                  device=device, epoch=e,path=new)
        val_loss = train_or_val_mulFrame(logger,modelConfig,model, val_loader, trainer, optimizer=optimizer, is_train=False,
                                device=device, epoch=e,path=new)
        end_time = time.time()
        epoch_duration = (end_time - start_time) / 60.0

        logger.info(f'Epoch {e} took {epoch_duration:.2f} min')
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {e}, lr={current_lr}")
        logger.info("------------------------------------------------")
        scheduler.step(val_loss)
        if val_loss < min_avg_val_loss or e % 10 == 0:
            if val_loss < min_avg_val_loss:
                min_avg_val_loss = val_loss
            # torch.save(model.state_dict(), os.path.join(
            #     modelConfig["save_vel_weight_dir"], 'velocity_randomSynthesis_wImgLoss_' + str(e) + "_.pt"))
        torch.save(model.state_dict(), os.path.join(
            savepath, 'velocity_' + str(e) + "_.pt"))
