

# Environment
Python 3.10

Pytroch 2.0.1

Cuda 11.8
# Our Dataset
https://drive.google.com/file/d/1iyOcIvqsTnf0rriuxelXmmIkKUVlKv1A/view?usp=drive_link
# Directory structure description

├── README.md           // Help document
    
├── Dataset_Production    // Dataset creation file
    
    ├── ADD_Noise             
    
    │   ├── utils
    
    │   ├── mytest.m    // matlab document
    
    ├── Low_Frequency             
    
    │   ├── Decompose.m  // matlab document
    
    │   ├── SideWindowBoxFilter.m    // matlab document
    
    ├── Lighting_Adjustment.py   // python document 
    
├── Modules    
    
    │   ├── CA.py
    
    │   ├── CSFI.py
    
    │   ├── CSM.py
    
    │   ├── CSFI.py
    
    │   ├── CVMI.py
  
    │   ├── Encoder_Decoder.py
    
    │   ├── CSFI.py
    
    │   ├── IEM.py
    
    │   ├── Net.py  // Overall network architecture
    
├── src    
    
    │   ├── datasets.py // dataloader
    
    │   ├── getmetrics.py  // calculate PSNR SSIM
    
    │   ├── losses.py   // loss
    
    │   ├── test.py   // test
    
    │   ├── test_dataset_value.py // Calculate the PSNR and SSIM of the predicted images
  
    │   ├── train.py  // train
    
    │   ├── util.py  // save image and set random seed
    
# Training process
Run train.py
train:

    low_left = r'Low light left view path'
    low_right = r'Low light right view path'
    gt_left = r'ground truth left view path'
    gt_right = r'ground truth right view path'
    low_frequency_left = r'low frequency left view path'
    low_frequency_right = r'low frequency right view path'

val:

    val_low_left = r'Low light left view path'
    val_low_right = r'Low light right view path'
    val_gt_left = r'ground truth left view path'
    val_gt_right = r'ground truth right view path'
    val_low_frequency_left = r'low frequency left view path'
    val_low_frequency_right = r'low frequency right view path'
# Testing process
    Run test.py
    
    parser.add_argument('--light_l', type=str, default=r"low frequency left view path")
    parser.add_argument('--light_r', type=str, default=r"low frequency right view path")
    parser.add_argument('--low_l', type=str, default=r"Low light left view path")
    parser.add_argument('--low_r', type=str, default=r"Low light right view path")
    parser.add_argument('--sava_left', type=str, default=r"save left view path")
    parser.add_argument('--save_right', type=str, default=r"save right view path")
    parser.add_argument('--snapshots_pth', type=str, default="../models/111.pth")

    Run test_dataset_value.py to calculate SSIM and PSNR
    
    path1 = r"ground truth path"
    path2 = r"pre path"
