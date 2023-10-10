from torch.utils import data
from torchvision import transforms as T
from glob import glob
from PIL import Image
import os

class alpha_matte_AB(data.Dataset):
    """ trainning Dataset -- the alpha_matte_AB dataset."""

    def __init__(self, root_train, transform, transform2):

        
        self.root_train = ""
        self.transform = transform
        self.transform2 = transform2
        self.A_files = glob(os.path.join(root_train, 'A_jpg','*.*'))

        #self.B_files = glob(os.path.join(root_train, 'B_jpg','*.*'))
        #self.GT_files = glob(os.path.join(root_train, 'image_jpg','*.*'))
        #self.focus_map = glob(os.path.join(root_train, 'focus_map_png','*.*'))
        self.root_train = root_train


    def __getitem__(self, index):
        imname = self.A_files[index]
        lind = imname.rfind("/")
        rind = imname.rfind(".")
        imname = imname[lind+1:rind]
        ind = imname.find('_')
        imname = imname[ind+1:]
        A_img_path = os.path.join(self.root_train, 'A_jpg', 'A_'+imname+'.jpg')
        B_img_path = os.path.join(self.root_train, 'B_jpg', 'B_'+imname+'.jpg')
        GT_img_path = os.path.join(self.root_train, 'image_jpg', imname+'.jpg')
        #focus_map_path = os.path.join(self.root_train, 'blurred_focus_map_png', 'blurred_'+imname+'.png')
        focus_map_path = os.path.join(self.root_train, 'focus_map_png', imname+'.png')
        A = Image.open(A_img_path)
        B = Image.open(B_img_path)
        GT = Image.open(GT_img_path)
        focus_map = Image.open(focus_map_path)
           
        A = self.transform(A)
        B = self.transform(B)
        GT = self.transform(GT)
        fm = self.transform2(focus_map).split(1,0)[0]
        
        return A, B, fm, GT

    def __len__(self):
        """Return the number of images."""
        return len(self.A_files)
    
class LytroDataset(data.Dataset):
    """test Dataset -- the LytroDataset."""
    def __init__(self, root_test, transform):
        self.transform = transform
        self.root_test = root_test
        self.A_files = sorted(glob(os.path.join(root_test, 'A_jpg','*.*')))
        self.B_files = glob(os.path.join(root_test, 'B_jpg','*.*'))

    def __getitem__(self, index):
        imname = self.A_files[index]
        lind = imname.rfind("/")
        rind = imname.rfind("-")
        imname = imname[lind+1:rind]
        A = Image.open(os.path.join(self.root_test, 'A_jpg',imname+'-A.jpg'))
        B = Image.open(os.path.join(self.root_test, 'B_jpg',imname+'-B.jpg'))
        A = self.transform(A)
        B = self.transform(B)
        
        return A, B

    def __len__(self):
        """Return the number of images."""
        return len(self.A_files)

class MFFW2(data.Dataset):
    """test Dataset -- the MFFW2 Dataset."""
    def __init__(self, root_test, transform):
        self.transform = transform
        self.root_test = root_test
        self.A_files = sorted(glob(os.path.join(root_test, 'A_jpg','*.*')))
        self.B_files = glob(os.path.join(root_test, 'B_jpg','*.*'))

    def __getitem__(self, index):

        imname = self.A_files[index]
        lind = imname.rfind("/")
        rind = imname.rfind("_")
        imname = imname[lind+1:rind]
        A = Image.open(os.path.join(self.root_test, 'A_jpg',imname+'_A.jpg'))
        B = Image.open(os.path.join(self.root_test, 'B_jpg',imname+'_B.jpg'))
        A = self.transform(A)
        B = self.transform(B)
        
        return A, B

    def __len__(self):
        """Return the number of images."""
        return len(self.A_files)
    
class grayscale_jpg(data.Dataset):
    """test Dataset -- the MFFW2 Dataset."""
    def __init__(self, root_test, transform):
        self.transform = transform
        self.root_test = root_test
        self.A_files = sorted(glob(os.path.join(root_test, 'A_jpg','*.*')))
        self.B_files = glob(os.path.join(root_test, 'B_jpg','*.*'))

    def __getitem__(self, index):

        imname = self.A_files[index]
        indr = self.A_files[index].rfind("_")
        indl = self.A_files[index].rfind("/")
        imname = imname[indl+1:indr]
        A = Image.open(self.root_test + "A_jpg/" + imname + "_A.jpg")
        B = Image.open(self.root_test + "B_jpg/" + imname + "_B.jpg")
        A = self.transform(A)
        B = self.transform(B)
        
        return A, B

    def __len__(self):
        """Return the number of images."""
        return len(self.A_files)

class MFIWHU(data.Dataset):
    """test Dataset -- the MFFW2 Dataset."""
    def __init__(self, root_test, transform):
        self.transform = transform
        self.root_test = root_test
        self.A_files = sorted(glob(os.path.join(root_test, 'A_jpg','*.*')))
        self.B_files = glob(os.path.join(root_test, 'B_jpg','*.*'))

    def __getitem__(self, index):

        imname = self.A_files[index]
        lind = imname.rfind("/")
        rind = imname.rfind("_")
        imname = imname[lind+1:rind]
        A = Image.open(os.path.join(self.root_test, 'A_jpg',imname+'_A.jpg'))
        B = Image.open(os.path.join(self.root_test, 'B_jpg',imname+'_B.jpg'))
        A = self.transform(A)
        B = self.transform(B)
        
        return A, B

    def __len__(self):
        """Return the number of images."""
        return len(self.A_files)
