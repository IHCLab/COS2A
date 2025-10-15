from os.path import join
from torchvision.transforms import Compose, ToTensor
from dataset import HSIDataset, HSITestDataset

def input_transform():
    return Compose([
        ToTensor(),
    ])

def get_training_set(patch_size, data_augmentation=True):
    root_dir = "./dataset"
    train_dir = join(root_dir, "Train_Spec")
    
    return HSIDataset(
        train_dir, 
        patch_size=patch_size,
        data_augmentation=data_augmentation,
        input_transform=input_transform()
    )

def get_validation_set():
    root_dir = "./dataset"
    val_dir = join(root_dir, "Valid_Spec")
    
    return HSITestDataset(
        val_dir,
        input_transform=input_transform()
    )

def get_test_set():
    root_dir = "./dataset"
    test_dir = join(root_dir, "Test_Spec")
    
    return HSITestDataset(
        test_dir,
        input_transform=input_transform()
    )
