import argparse
import torch
import numpy as np
import lpips

import FILE
import model

def get_test_data(LR, HR):
    print('Load LR')
    lrs = FILE.load_img(FILE.load_file_path(LR), Normlize=True, Transpose=True)
    print(str(len(lrs))+' low resolution images in folder')
    print('Load HR')
    hrs = FILE.load_img(FILE.load_file_path(HR), Transpose=True)
    print(str(len(hrs))+' GT images in folder')
    return lrs, hrs

def get_model(model_type, model_path):
    if model_type == 'SRRes':
        G = model.SRRes()
    elif model_type == 'RRDB':
        G = model.RRDBNet(3,3,64,23)

    G.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    G.eval()
    G.cuda()
    print('Model Load Success')
    return G

def get_srs(G, lrs):
    srs = []
    for i in range(len(lrs)):
        print('Process Image {}/{}'.format(i+1, len(lrs)))
        lr = torch.Tensor(np.expand_dims(lrs[i], 0)).cuda()
        with torch.no_grad():
            sr = G(lr).squeeze().detach().cpu().numpy()
            sr = np.transpose(sr, [1,2,0])
            sr = np.clip(sr*255,0,255).astype('uint8')
            srs.append(sr)
    print('Process Complete')
    return srs

def get_lpips(srs, hrs):
    lp = lpips.LPIPS(net='alex').cuda()
    tp = 0.
    for i in range(len(srs)):
        sr = np.expand_dims(srs[i], 0).astype('float32')
        sr = sr/127.5 - 1
        sr = torch.Tensor(sr).cuda()
        hr = np.expand_dims(hrs[i], 0).astype('float32')
        hr = hr/127.5 - 1
        hr = torch.Tensor(hr).cuda()
        tp += lp(hr,sr).squeeze().detach().cpu().float()
    tp /= len(srs)
    print('Average LPIPS is: {}'.format(tp))


def test(args):
    lrs, hrs = get_test_data(args.LR_PATH, args.HR_PATH)
    G = get_model(args.Model, args.Model_PATH)
    srs = get_srs(G, lrs)
    FILE.save_imgs(args.Save_PATH, srs)
    srs = FILE.load_img(FILE.load_file_path(args.Save_PATH), Transpose=True)
    get_lpips(srs, hrs)
    print('Test Complete')
    print('PSNR, SSIM, PI calculate by Matlab in Y channel')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Setting')
    parser.add_argument('--LR_PATH', default='D:\\date\\BSD100_X4_LR\\RGB', help='address for input low resolution images folder')
    parser.add_argument('--HR_PATH', default='D:\\date\\BSD100_HR\\RGB', help='corresponding GT folder')
    parser.add_argument('--Save_PATH', default='C:\\Users\\ada\\Desktop\\PXX\\Test', help='folder for result')

    parser.add_argument('--Model', default='RRDB', help='Generator Type', choices=['SRRes', 'RRDB'])
    parser.add_argument('--Model_PATH', default='C:\\Users\\ada\\Desktop\\PXX/IMSRGAN_RRDB_MS.pth', help='Pre-trained model path')

    test_args = parser.parse_args()

    test(test_args)



