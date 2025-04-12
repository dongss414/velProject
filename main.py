from train import train



def main(model_config = None):
    modelConfig = {
        "state": "train",
        "epoch": 60,
        'start_epoch':0,
        "batch_size": 1,
        "lr": 1e-4,
        "grad_clip": 1.,
        'dropout': 0,
        "device": "cuda:0", ### MAKE SURE YOU HAVE A GPU !!!
        "training_vel_load_weight": "velocity.pt",
        "save_vel_weight_dir": "./weight/",
        'velNum': 3,
        'weight':[1,1,1,0.1,0.1],
        'img_size' : (64,64),
        'den_size' : (64,64,64),
        'vel_size': (3,64,64,64),
        'cube_size': 64,
        'dataset':'syn',
        }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        print("training Gu")
        train(modelConfig)






if __name__ == '__main__':
    main()
