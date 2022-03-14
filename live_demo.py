import torch
import cv2 as cv
import numpy as np
from time import time
import albumentations as A
from torchvision.models import detection
from postprocessing import predict_images, filter_predictions
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def custom_model(num_classes):
    model = detection.fasterrcnn_resnet50_fpn(pretrained = True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    print(f'Box predictor in_features  : {in_features}')
    print(f'Box predictor out_features : {num_classes}')
  
    return model

def load_model_optimizer(path, device, return_optimizer = True):
    checkpoint = torch.load(path, map_location = device)
    model = custom_model(3 + 1)
    
    learning_rate = 1e-2
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr = learning_rate, weight_decay=0.0005)
    
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    print(" Loading Model ".center(60, '='))
    for key in checkpoint.keys():
        if 'state' not in key:
            print(f'Best {key} :===> {checkpoint[key]}')
    print('=' * 60)
            
    if return_optimizer:
        return model, optimizer
    return model

def close_cam(cap):
    cap.release()
    cv.destroyAllWindows()
    
def transform(image, target, size):
    h, w = size
    T = A.Compose([
        A.Resize(h, w)
    ], A.BboxParams(format = 'pascal_voc', label_fields = ['class_labels']))
    
    transformed      = T(image = image, bboxes = target['boxes'], class_labels = target['labels'])
    target['boxes']  = transformed['bboxes']
    target['labels'] = transformed['class_labels']
    return transformed['image'], target

def display_pred_in_cam(cap, image, target):
    
    if target:
        
        label_enc = {
            1 : ['with mask', (0, 0, 255)],
            2 : ['without mask', (255, 0, 0)],
            3 : ['mask worn incorrectly', (0, 255, 0)]    
        }
        
        boxes = target['boxes']
        scores = target['scores']
        labels = target['labels']
        for ind in range(len(labels)):
            x1,y1, x2,y2 = boxes[ind]
            label, color = label_enc[labels[ind].item()]
            cv.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
            cv.putText(image, label, (int(x1), int(y1) + 2), cv.FONT_HERSHEY_PLAIN, 1, color, 1, cv.LINE_AA)
            
    cv.imshow('Mask Detection', image)
    
def main():
    H, W = (480, 480)
    device = torch.device('cpu')
    model = load_model_optimizer('best_model.pth', device, False)

    iou_threshold = 0.1
    score_threshold = 0.60

    cap = cv.VideoCapture(0)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frames = 0
    old_fps, new_fps = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        orig_img = np.array(frame)
        img_resized = np.array(cv.resize(orig_img, (H, W)))
        
        img_tensor = torch.from_numpy(img_resized).permute(-1, 0, 1)
        img_tensor = (img_tensor / 255)
        loader     = [
            [[img_tensor] , [None]]
        ]
        
        new_fps = time()
        predictions  = predict_images(model, loader, device, iou_threshold)
        fpredictions = filter_predictions(predictions, score_threshold)
        
        frames += 1
        print(f' Frame ( {frames} ) prediction '.center(60, '='))
        print()
        print(fpredictions[0])
        print()
    
        image, target = transform(img_resized, fpredictions[0], (height, width))
        
        fps  = 1/(new_fps - old_fps)
        old_fps = new_fps
        cv.putText(frame, f'fps : {fps}', (int(10), int(20) + 1), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1, cv.LINE_AA)
        display_pred_in_cam(cap, image, target)
        
        if cv.waitKey(10) == ord('q'):
            close_cam(cap)
            break
        
        
        
main()
        
        