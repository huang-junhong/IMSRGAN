# IMSRGAN
implement of paper 'IMSRGAN: Perceptual Single Image Super Resolution With Mixture Images and global sharp'

## Fast Test
we provide 5 per-trained model for fast test, you can download in [Google cloud](https://drive.google.com/drive/folders/1oOotFsmoDGbqitlxDlYb6vmEculsI609?usp=sharing)

* IMSRGAN_RRDB.pth       this model use RRDB as generator, witch result in paper table-VII IMSRAGN
* IMSRGAN_SRRes.pth      this model use SRRes as generator, witch result in paper table-VI  SRRes-IMSRAGN
* IMSRGAN_RRDB_GS.pth    this model use RRDB as generator and use GS, witch result in paper table-VII IMSRAGN-GS
* IMSRGAN_SRRes_GS.pth   this model use SRRes as generator, witch result in paper table-VI  SRRes-IMSRAGN-GS
* IMSRGAN_RRDB_MS.pth    this model use RRDB as generator, witch result in paper table-III Multi Shape

Before Test
* you should pip install cv2 and [lpips]()
* Download our pre-trained model
* vi IMSRGAN_Test.py to change folder address, the default path is absolute address you should change by yourself.
* run IMSRGAN_Test.py
* calculate PSNR, SSIM, PI in MATLAB
