# Environment
Python 3.10

Pytroch 2.0.1

Cuda 11.8
# Dataset

www.xxxxxxxx
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

