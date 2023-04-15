# built-in lin
import json
from pathlib import Path
from multiprocessing import Pool
# third-party lib
import cv2
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
# personal lib
from lib.utils.linemod.opengl_renderer import OpenGLRenderer
from .handle_custom_dataset import sample_fps_points, custom_to_coco

class BackImg:
    def __init__(self,root='data/custom/background',rand_seed=1234,ratio=0.6,img_size=(640,480)) -> None:
        self.root = Path(root)
        self.ratio = ratio
        self.rng = np.random.default_rng(seed=rand_seed)
        self.img_size = img_size

    def alloc(self):
        train_imgs = self.root/'train_imgs.txt'
        vali_imgs = self.root/'vali_imgs.txt'
        test_imgs = self.root/'test_imgs.txt'
        paths = self.root/'paths.txt'

        train_imgs.write_text('')
        vali_imgs.write_text('')
        test_imgs.write_text('')
        paths.write_text('')

        subdirs = [subdir for subdir in self.root.iterdir() if subdir.is_dir()]
        for subdir in subdirs:
            imgs_path = [str(path) for path in subdir.iterdir() if path.is_file()]
            with paths.open('+a') as f:
                f.write('\n'.join(imgs_path))
                f.write('\n')

            self.rng.shuffle(imgs_path)
            imgs_num = len(imgs_path)
            train_num = int(imgs_num*self.ratio)
            vali_num = (imgs_num-train_num)//2

            with train_imgs.open('+a') as f:
                f.write('\n'.join([path for path in imgs_path[:train_num]]))
                f.write('\n')
    
            with vali_imgs.open('+a') as f:
                f.write('\n'.join([path for path in imgs_path[train_num:train_num+vali_num]]))
                f.write('\n')

            with test_imgs.open('+a') as f:
                f.write('\n'.join([path for path in imgs_path[train_num+vali_num:]]))
                f.write('\n')

    def __iter__(self):
        if not (self.root/'paths.txt').is_file():
            self.alloc()
        paths = (self.root/'paths.txt').read_text().split('\n')
        for path in paths:
            img = cv2.imread(path)
            img = cv2.resize(img,self.img_size)
            yield img,path

    def __getitem__(self,index):
        if not (self.root/'paths.txt').is_file():
            self.alloc()
        paths = (self.root/'paths.txt').read_text().split('\n')
        path = paths[index]
        img = cv2.imread(path)
        img = cv2.resize(img,self.img_size)
        return img,path

    def __len__(self):
        if not (self.root/'paths.txt').is_file():
            self.alloc()
        paths = (self.root/'paths.txt').read_text().split('\n')
        return len(paths)-1

# ---------------------------------- need improvement -----------------------------
coco = COCO('data/linemod/cat/train.json')
len_ = len(coco.getAnnIds())
offset = 0
for _ in range(len_):
    offset += 1
    anno = coco.loadAnns(offset)[0]
    if anno['type'] == 'render':
        break
def make_pose(index):
    global coco
    global len_
    global offset
    pose = np.array(coco.loadAnns((index+offset)%(len_-1))[0]['pose'])
    pose = np.hstack((pose[:,:3],pose[:,3:]*0.5)) # zoom in
    return pose
# --------------------------------------------------------------------------------

def make_img(root='data/custom',back_root='data/custom/background'):
    # get model and camera instrinsic
    root = Path(root)
    back_root = Path(back_root)
    render = OpenGLRenderer(root/'model.ply')
    K = np.loadtxt(root/'camera.txt')
    img_size = (640,480)

    # make directories
    rgb_dir = root/'rgb'
    pose_dir = root/'pose'
    mask_dir = root/'mask'
    for dir in (rgb_dir,pose_dir,mask_dir):
        for subdir in [str(i) for i in back_root.iterdir() if i.is_dir()]:
            (dir/(subdir.split('/')[-1])).mkdir(parents=True,exist_ok=True)

    # make and save
    background = BackImg(back_root,rand_seed=1234,ratio=0.6,img_size=img_size)
    for index in range(len(background)):
        # make img, pose, mask
        back_img,img_path = background[index]
        img_path = '/'.join(img_path.split('/')[-2:])
        img_size = np.array(img_size)
        pose = make_pose(index)
        render_img = render.render(pose,K,img_size,'rgb')
        mask = (render_img[...,0] != 0).astype(np.uint8)
        img = render_img + back_img*((mask==0)[...,None])

        # save img, pose, mask
        cv2.imwrite(str(rgb_dir)+'/'+img_path,img)
        np.save(str(pose_dir)+'/'+img_path[:-3]+'npy',pose)
        cv2.imwrite(str(mask_dir)+'/'+img_path[:-3]+'png',mask)
        print(index,img_path)


def make_dataset(root='data/custom',back_root='data/custom/background'):
    background = BackImg()
    background.alloc()
    make_img(root,back_root)
    sample_fps_points(root)

    for file in [i for i in Path(back_root).iterdir() if i.is_file()]:
        path_list = file.read_text().split('\n')
        kind = (str(file).split('/')[-1]).split('_')[0]
        custom_to_coco(path_list=path_list[:-1],kind=kind)

if __name__ == "__main__":
    make_dataset()