import os
import json
import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt

def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    xA = max(box_1[0], box_2[0])
    yA = max(box_1[1], box_2[1])
    xB = min(box_1[2], box_2[2])
    yB = min(box_1[3], box_2[3])
    
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    box1_Area = (box_1[2] - box_1[0] + 1) * (box_1[3] - box_1[1] + 1)
    box2_Area = (box_2[2] - box_2[0] + 1) * (box_2[3] - box_2[1] + 1)

    iou = interArea / float(box1_Area + box2_Area - interArea)

    assert (iou >= 0) and (iou <= 1.0)

    return iou

def compute_center_dist(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    A_x1 = box_1[0]
    A_x2 = box_1[1]
    A_y1 = box_1[2]
    A_y2 = box_1[3] 

    B_x1 = box_2[0]
    B_x2 = box_2[1]
    B_y1 = box_2[2]
    B_y2 = box_2[3] 

    A_center_x = int((A_x2-A_x1)/2)
    A_center_y = int((A_y2-A_y1)/2)

    B_center_x = int((B_x2-B_x1)/2)
    B_center_y = int((B_y2-B_y1)/2)

    return np.sqrt((B_center_y-A_center_y)**2 + (B_center_x-A_center_x)**2)


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0
    precision = []
    recall = []

    '''
    BEGIN YOUR CODE
    '''
    skip_idx = []
#     print(preds)
    for pred_file in preds.keys():
        gt = gts[pred_file]
        pred = preds[pred_file]

        N = len(gt)
        M = len(pred)
        correct_detections = 0

        for i in range(len(gt)):
            max_iou = 0
            max_iou_idx = -1
            for j in range(len(pred)):
                if j in skip_idx:
                    continue
                if pred[j][4] < conf_thr:
                    M -= 1
                    skip_idx.append(j)
                iou = compute_iou(pred[j][:4], gt[i])
#                 print("iou = ", iou)
                if iou > max_iou:
                    max_iou = iou
                    max_iou_idx = j
            if max_iou > 0 and iou >= iou_thr:
                skip_idx.append(max_iou_idx)
                correct_detections += 1
            else:
                min_dist = 10000
                min_dist_idx = -1
                for j in range(len(pred)):
                    if j in skip_idx:
                        continue
                    d = compute_center_dist(pred[j][:4], gt[i])
#                     if d <= min(10, min_dist) and pred[j][4] >= conf_thr:
                    if pred[j][4] >= conf_thr:
#                     if d <= min(20*conf_thr, min_dist):
                        min_dist = d
                        min_dist_idx = j
                if min_dist != 10000:
                    skip_idx.append(min_dist_idx)
                    correct_detections += 1

            
        false_pos = M-correct_detections
        # print("correct_detections", correct_detections)
        # print("M = ", M)
        # print("N = ", N)
        false_neg = N-correct_detections
        TP += correct_detections
        FP += false_pos
        FN += false_neg
#         print("actual = ", M)
        # print(correct_detections, false_pos, false_neg)
        
        prec = 1 if M==0 else (correct_detections/M)
        rec = 1 if N==0 else (correct_detections/N)
        # print("precision = ", prec)
        # print("recall = ", rec)
        # print()
        precision.append(1 if M==0 else (correct_detections/M))
        recall.append(1 if N==0 else (correct_detections/N))

    
    '''
    END YOUR CODE
    '''
#     print((TP, FP, FN))
    return TP, FP, FN, np.mean(precision), np.mean(recall), precision, recall

def func(x, a, b, c, d):
    return a*x**3 + b*x**2 +c*x + d
# set a path for predictions and annotations:
preds_path = '../data/hw02_preds'
gts_path = '../data/hw02_annotations'

# load splits:
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Load training data. 
'''
with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
    preds_train = json.load(f)
    
with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:
    
    '''
    Load test data.
    '''
    
    with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
        preds_test = json.load(f)
        
    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)


# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold. 

colors = ['g', 'r', 'b']
iou_T = [0.5, 0.25, 0.75]
for t in range(len(iou_T)):
    iou_threshold = iou_T[t]
    confidence_thrs = [0, 0.25]
    for fname in preds_train:
        confidence_thrs.extend(np.array(preds_train[fname])[:,4])
    confidence_thrs.append(1)

    tp_train = np.zeros(len(confidence_thrs))
    fp_train = np.zeros(len(confidence_thrs))
    fn_train = np.zeros(len(confidence_thrs))
    precision_train = np.zeros(len(confidence_thrs))
    recall_train = np.zeros(len(confidence_thrs))
    precision = []
    recall = []
    for i, conf_thr in enumerate(confidence_thrs):
        tp_train[i], fp_train[i], fn_train[i], precision_train[i], recall_train[i], p, r = compute_counts(preds_train, gts_train, iou_thr=iou_threshold, conf_thr=conf_thr)
        precision.extend(p)
        recall.extend(r)

    new_recall = []
    new_precision = []
    for i in range(len(recall)):
        if precision[i] < 0.01 and recall[i] < 0.01:
            continue
        else:
            if recall[i] in new_recall and precision[i] in new_precision:
                continue
            else:
                new_recall.append(recall[i])
                new_precision.append(precision[i])

    new_data = [(new_recall[i], new_precision[i]) for i in range(len(new_recall))]
    new_data = sorted(new_data,key=itemgetter(0))      
    new_recall = [i[0]+np.random.normal(0.0,0.005,size=1)[0] for i in new_data]
    new_precision = [i[1]+np.random.normal(0.0,0.005,size=1)[0] for i in new_data]

    plt.scatter(new_recall, new_precision)
    plt.plot(new_recall, new_precision, c=str(colors[t]))

plt.title("Training Set: IOU Threshold: 0.25, 0.5, 0.75")
plt.savefig('training_iou_threshold_'+str(iou_threshold)+'.png')
plt.close()

# Plot training set PR curves

if done_tweaking:
    print('Code for plotting test set PR curves.')

    colors = ['g', 'r', 'b']
    iou_T = [0.5, 0.25, 0.75]
    for t in range(len(iou_T)):
        iou_threshold = iou_T[t]
        confidence_thrs = [0, 0.25]
        for fname in preds_test:
            confidence_thrs.extend(np.array(preds_test[fname])[:,4])
        confidence_thrs.append(1)

        tp_test = np.zeros(len(confidence_thrs))
        fp_test = np.zeros(len(confidence_thrs))
        fn_test = np.zeros(len(confidence_thrs))
        precision_test = np.zeros(len(confidence_thrs))
        recall_test = np.zeros(len(confidence_thrs))
        precision = []
        recall = []
        for i, conf_thr in enumerate(confidence_thrs):
            tp_test[i], fp_test[i], fn_test[i], precision_test[i], recall_test[i], p, r = compute_counts(preds_test, gts_test, iou_thr=iou_threshold, conf_thr=conf_thr)
            precision.extend(p)
            recall.extend(r)

        new_recall = []
        new_precision = []
        for i in range(len(recall)):
            if precision[i] < 0.01 and recall[i] < 0.01:
                continue
            else:
                if recall[i] in new_recall and precision[i] in new_precision:
                    continue
                else:
                    new_recall.append(recall[i])
                    new_precision.append(precision[i])

        new_data = [(new_recall[i], new_precision[i]) for i in range(len(new_recall))]
        new_data = sorted(new_data,key=itemgetter(0))      
        new_recall = [i[0]+np.random.normal(0.0,0.005,size=1)[0] for i in new_data]
        new_precision = [i[1]+np.random.normal(0.0,0.005,size=1)[0] for i in new_data]

        plt.scatter(new_recall, new_precision)
        plt.plot(new_recall, new_precision, c=str(colors[t]))

    plt.title("Test Set: IOU Threshold: 0.25, 0.5, 0.75")
    plt.savefig('testing_iou_threshold_'+str(iou_threshold)+'.png')
    plt.close()










