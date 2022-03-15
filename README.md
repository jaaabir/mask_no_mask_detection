# Face Mask Detection

<img src="./assets/face_mask.jpg">

It is a face mask detection, trained using completely on pytorch by using a pre-trained Faster R-CNN with pre-trained RES-NET 50 feature pyramid network ( FPN ) as it's backbone due to low number of images and to get a good accuracy. Was able to achieve `Validation Loss :==> 0.348` and the final epoch losses are

```
{
 "loss_classifier"  : 0.084,
 "loss_box_reg"     : 0.195,
 "loss_objectness"  : 0.013,
 "loss_rpn_box_reg" : 0.004
}
```

### usage

**python live_demo.py --help**

```
usage: live_demo.py [-h] [--vc VC] [--i I] [--s S] [--d D] [--t T] [--verbose]

Video capturing and hyper-parameter tuning

optional arguments:
  -h, --help  show this help message and exit
  --vc VC     cv.videoCapture : 0, 1 or video path, ( default : 0 )
  --i I       iou threshold, should be in range 0.0 - 1.0 , ( default : 0.1 )
  --s S       score threshold, should be in range 0.0 - 1.0, ( default : 0.6 )
  --d D       device to train, [cpu, cuda], ( default : cpu )
  --t T       time to wait after capturing one frame in ms, ( default : 10 )
  --verbose   if verbose , print out the predicted logs
```
