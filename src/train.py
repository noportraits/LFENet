from torch.utils.data import DataLoader
from getmetrics import psnr_ssim
from modules.Net import Net
from losses import LossFre, LossSpa
from datasets import Dataset
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
import torch
import time
from util import set_random_seed

epoch_losses = [1]
# train
low_left = r'D:\mydataset\bestbignoise\holopix\train\normal\low\left'
low_right = r'D:\mydataset\bestbignoise\holopix\train\normal\low\right'
gt_left = r'D:\mydataset\bestbignoise\holopix\train\normal\gt\left'
gt_right = r'D:\mydataset\bestbignoise\holopix\train\normal\gt\right'
low_frequency_left = r'D:\mydataset\bestbignoise\holopix\train\low_fre\low\left'
low_frequency_right = r'D:\mydataset\bestbignoise\holopix\train\low_fre\low\right'
seed = 12345
train_batch_size = 7
crop_train = [200, 200]
scheduler_list = [250, 500, 750, 1000]
lr = 0.0001
epochs = 1000
# val
val_low_left = r'D:\mydataset\bestbignoise\holopix\val\normal\low\left'
val_low_right = r'D:\mydataset\bestbignoise\holopix\val\normal\low\right'
val_gt_left = r'D:\mydataset\bestbignoise\holopix\val\normal\gt\left'
val_gt_right = r'D:\mydataset\bestbignoise\holopix\val\normal\gt\right'
val_low_frequency_left = r'D:\mydataset\bestbignoise\holopix\val\low_fre\low\left'
val_low_frequency_right = r'D:\mydataset\bestbignoise\holopix\val\low_fre\low\right'
val_batch_size = 1
crop_val = [400, 400]
val_gap = 1
set_random_seed(seed)
# datasets
train_dataset = Dataset(low_frequency_left, low_frequency_right, low_left, low_right, gt_left, gt_right,
                                 mode='train', crop=crop_train, random_resize=None)
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=1, pin_memory=True)
val_dataset = Dataset(val_low_frequency_left, val_low_frequency_right, val_low_left, val_low_right,
                               val_gt_left, val_gt_right, mode='val', crop=crop_val)
val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=1, pin_memory=True)
ssimepochs = [1]
best_ssim_epoch = 0
best_psnr_epoch = 0
max_ssim_val = 0
max_psnr_val = 0
model = Net().cuda()
freloss = LossFre().cuda()
spaloss = LossSpa().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, scheduler_list, 0.5)

save_dir = "../models"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  
def main():
    start_epoch = 1
    start = time.time()
    for epoch in range(start_epoch, epochs + 1):
        train(epoch)
        if (epoch) % val_gap == 0:
            val(epoch)
        elapsed_time = time.time() - start
        h, rem = divmod(elapsed_time, 3600)
        m, rem = divmod(rem, 60)
        s = rem
        avg_time_per_epoch = elapsed_time / epoch
        remaining_epochs = epochs - epoch
        remaining_time = remaining_epochs * avg_time_per_epoch

        hours, rem = divmod(remaining_time, 3600)
        minutes, _ = divmod(rem, 60)
        print(
            'Epoch [{}/{}], Time Elapsed: {:.2f} hours, {:.2f} minutes, {:.2f} seconds, Expected Time Remaining: {} hours and {} minutes'.format(
                epoch, epochs, int(h), int(m), int(s), int(hours), int(minutes)))


def train(epoch):
    model.train()
    max = len(train_dataloader)
    for i, (light_l, light_r, low_l, low_r, gt_l, gt_r) in enumerate(train_dataloader):
        [light_l, light_r, low_l, low_r, gt_l, gt_r] = [x.cuda() for x in [light_l, light_r, low_l, low_r, gt_l, gt_r]]
        optimizer.zero_grad()
        pre_l, pre_r = model(light_l, light_r, low_l, low_r)
        fre_loss = freloss(pre_l, gt_l) + freloss(pre_r, gt_r)
        spa_loss = spaloss(pre_l, gt_l) + spaloss(pre_r, gt_r)
        loss = fre_loss + spa_loss
        loss.backward()
        optimizer.step()
        if (i + 1) % 1 == 0:
            print(
                'Training: Epoch[{:0>4}/{:0>4}] Iteration[{:0>4}/{:0>4}]  loss: {:.4f} fre_loss: {:.4f} spa_loss: {:.4f}'.format(
                    epoch,
                    epochs, i + 1, max, loss, fre_loss, spa_loss))

    torch.save(model.state_dict(), "../models/current_epoch_200.pth")

    scheduler.step()


def val(epoch):
    a_total = 0
    b_total = 0
    sum = 0
    for i, (light_l, light_r, low_l, low_r, gt_l, gt_r) in enumerate(val_dataloader):
        with torch.no_grad():
            model.eval()
            torch.cuda.empty_cache()
            [light_l, light_r, low_l, low_r, gt_l, gt_r] = [x.cuda() for x in
                                                            [light_l, light_r, low_l, low_r, gt_l, gt_r]]
            val_l, val_r = model(light_l, light_r, low_l, low_r)
        val_l = val_l.squeeze(0)
        val_r = val_r.squeeze(0)
        gt_l = gt_l.squeeze(0)
        gt_r = gt_r.squeeze(0)
        psnr_l, ssim_l = psnr_ssim(val_l, gt_l)
        psnr_r, ssim_r = psnr_ssim(val_r, gt_r)
        a_total = a_total + (psnr_l + psnr_r) / 2
        b_total = b_total + (ssim_l + ssim_r) / 2
        i += 1
        sum = i
        print("第", i, "张:", "psnr:", (psnr_l + psnr_r) / 2, "  ssim:", (ssim_l + ssim_r) / 2)
    print("total: psnr", a_total / sum, " ssim:", b_total / sum)
    ssimepochs.append(1 - b_total / sum)
    global max_ssim_val, max_psnr_val, best_ssim_epoch, best_psnr_epoch
    if a_total / sum > max_psnr_val:
        max_psnr_val = a_total / sum
        best_psnr_epoch = epoch
        torch.save(model.state_dict(), "../models/best_psnr_epoch_200.pth")
    if b_total / sum > max_ssim_val:
        max_ssim_val = b_total / sum
        best_ssim_epoch = epoch
        torch.save(model.state_dict(), "../models/best_ssim_epoch_200.pth")
    plt.figure()
    plt.plot(ssimepochs, color='b')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('SSIM during Validation')
    plt.savefig(f'ssim_during_val.png')
    plt.close()
    torch.cuda.empty_cache()
    print("best ssim epoch: ", best_ssim_epoch, " ssim:", max_ssim_val, "best psnr epoch: ", best_psnr_epoch, " ssim:", max_psnr_val)


if __name__ == '__main__':
    main()
