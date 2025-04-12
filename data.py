import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import torch,os

class TrainVelocityMulFrameSynthesisDataset(Dataset):
    def __init__(self, modelConfig, velpath, denpath, use_train,scenenum = 81 ,veltransform=None, dentransform=None,imagetransform=None):
        self.modelConfig = modelConfig
        self.loadVelNum = modelConfig['velNum']
        self.use_train = use_train
        self.densize = modelConfig['den_size']
        self.velsize = modelConfig['vel_size']
        self.sceneSize = 150
        self.available = self.sceneSize - self.loadVelNum
        if self.use_train:
            self.ITEM = self.available * scenenum
        else:
            self.ITEM = self.available * scenenum



        self.velpath = velpath
        self.denpath = denpath
        self.veltransform = veltransform
        self.dentransform = dentransform
        self.imagetransform = imagetransform

    def expand(self,front,side):
        _,width = front.shape
        out1 = front.unsqueeze(0).unsqueeze(0).repeat(1,width,1,1)
        out2 = side.unsqueeze(0).unsqueeze(-1).repeat(1,1,1,width)
        return out1,out2

    def resize3D(self, x, size):
        resized_data = F.interpolate(x, size=size, mode='trilinear',
                                     align_corners=False)
        return resized_data
    def bin2tensor(self, index):
        if not self.use_train:
            index = index + self.available * 81


        scene = index // self.available
        frame = index % self.available
        num = scene * self.sceneSize + frame
        srcnum = scene * self.sceneSize
        # print(srcnum)

        def find_files(path, prefix, base_num, count):
            num_list = [base_num + i for i in range(count)]
            return [os.path.join(path, f) for f in os.listdir(path) if
                    f.endswith('.bin') and any(f'{prefix}{num:06d}' in f for num in num_list)]

        velfile = find_files(self.velpath, 'velocity', num, self.loadVelNum)
        denfile = find_files(self.denpath, 'density', num, self.loadVelNum + 1)
        srcfile = find_files(self.denpath, 'density', srcnum, 1)


        if not velfile:
            raise FileNotFoundError(f"No file found for velocity{num:06d} in {self.velpath}")
        if not denfile:
            raise FileNotFoundError(f"No file found for density{num:06d} in {self.denpath}")
        if not srcfile:
            raise FileNotFoundError(f"No file found for density{srcnum:06d} in {self.denpath}")

        def load_files(file_list, size ,orgsize):
            data_dict = {}
            for file in file_list:
                data_key = next(key for key in range(num, num + len(file_list)) if f'{key:06d}' in file)
                data = np.fromfile(file, dtype=np.float32).reshape(orgsize)
                if data.shape == (3,64,64,64):
                    data = torch.from_numpy(data).unsqueeze(0)
                    data = self.resize3D(data, size)[0]

                else:
                    data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
                    data = self.resize3D(data, size)[0,0]
                data_dict[data_key] = data
            return data_dict

        vel_dict = load_files(velfile, self.densize,(3,64,64,64))
        den_dict = load_files(denfile, self.densize,(64,64,64))
        src = srcfile[0]
        src = np.fromfile(src, dtype=np.float32).reshape((64,64,64))
        src = torch.from_numpy(src).unsqueeze(0).unsqueeze(0)
        src = self.resize3D(src, self.densize)[0,0]
        # src = torch.from_numpy(src)

        velList = [vel_dict[num + i] for i in range(self.loadVelNum)]
        denList = [torch.unsqueeze(den_dict[num + i], 0) for i in range(self.loadVelNum + 1)]


        # def get_views(data):
        #     front_view = torch.mean(data, dim=0)
        #     side_view = torch.mean(data, dim=2)
        #     return self.expand(front_view, side_view)

        images = []
        # for i in range(num, num + self.loadVelNum + 1):
        #     images.extend(get_views(den_dict[i]))


        return num, images,denList,velList,src

    def __getitem__(self, index):
        num, images,dens,vels,src = self.bin2tensor(index)
        # print(num)

        imageList = []
        if self.imagetransform:
            for image in images:
                image = self.imagetransform(image)
                imageList.append(image)
        else:
            imageList = images

        velList = []
        if self.veltransform:
            for vel in vels:
                vel = self.veltransform(vel)
                velList.append(vel)
        else:
            velList = vels

        denList = []
        if self.dentransform:
            for den in dens:
                den = self.dentransform(den)
                denList.append(den)
        else:
            denList = dens


        return {'velList': velList,
                'denList':denList,
                'num': str(f'{num:06d}'),
                'images': imageList,
                'src':src,
                'isSF':False
                }

    def __len__(self):
        return self.ITEM