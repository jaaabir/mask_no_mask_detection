
import torch
import torchvision
import matplotlib.pyplot as plt
from visualization_utils import plot_img_bbox
 


def get_row(num, col):
    if num % col == 0:
        return num // col
    return (num // col) + 1

def apply_nms(orig_prediction, iou_thresh=0.3):
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
  
    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]
  
    return final_prediction

def predict_images(MODEL, test_loader, device, iou_threshold):
    MODEL.eval()
    MODEL.to(device)

    predictions = []
    with torch.no_grad():
        for test_images, _ in test_loader:
            images  = list(img.to(device) for img in test_images)

            prediction  = MODEL(images)
            prediction  = [{k : v.to(torch.device('cpu')) for k, v in apply_nms(pred_dict, iou_threshold).items()} 
                                                                                   for pred_dict in prediction]
            predictions += prediction

    return predictions

def filter_predictions(predictions, score_threshold):
    filtered_predictions = []
    for pred_dict in predictions:
        scores = pred_dict['scores']
        idx = []
        for ind, score in enumerate(scores):
            if score >= score_threshold:
                idx.append(ind)
                
        pred_dict['boxes'] = pred_dict['boxes'][idx, :]
        pred_dict['labels'] = pred_dict['labels'][idx]
        pred_dict['scores'] = pred_dict['scores'][idx]
        
        filtered_predictions.append(pred_dict)
        
    return filtered_predictions

def pred_show_bbox(MODEL, test_loader, device, iou_threshold, 
                       score_threshold, col = 5, channel_after = False, show_labels = True):
    
    predictions = predict_images(MODEL, test_loader, device, iou_threshold)
    predictions = filter_predictions(predictions, score_threshold)
    images      = []
    for imgs, _ in test_loader:
        images += imgs
    num         = len(images)
    row = get_row(num, col)

    fig, ax = plt.subplots(row, col, figsize = (col * 4, row * 4))

    ind   = 0
    for i in range(row):
        for j in range(col):
            if row == 1:
                axes = ax[j]
            else:
                axes = ax[i, j] 
                
            plot_img_bbox(axes, images[ind], predictions[ind], channel_after, show_labels)
            ind += 1