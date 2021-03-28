from __future__ import print_function
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.utils.data import RandomSampler

import wandb

import dataset
import random
import math
import os
from opts import parse_opts
from utils import *
from cfg import parse_cfg
from region_loss import RegionLoss

from model import YOWO, get_fine_tuning_parameters

# Training settings
opt = parse_opts()

# which dataset to use
dataset_use   = opt.dataset
assert dataset_use == 'ucf101-24' or dataset_use == 'jhmdb-21', 'invalid dataset'
# path for dataset of training and validation
datacfg       = opt.data_cfg
# path for cfg file
cfgfile       = opt.cfg_file

data_options  = read_data_cfg(datacfg)
net_options   = parse_cfg(cfgfile)[0]

# obtain list for training and testing
basepath      = data_options['base']
trainlist     = data_options['train']
testlist      = data_options['valid']
backupdir     = data_options['backup']

# number of training samples
nsamples      = file_lines(trainlist)
gpu_ids = list(range(torch.cuda.device_count()))
gpus = ','.join([str(g) for g in gpu_ids]) # gpus          = data_options['gpus']  # e.g. 0,1,2,3
ngpus = len(gpu_ids) # ngpus         = len(gpus.split(','))
num_workers   = int(data_options['num_workers'])

# Set up sample size per epoch
train_n_sample_from = 1 if dataset_use != 'ucf101-24' else 15
test_n_sample_from = 1 if opt.evaluate or dataset_use != 'ucf101-24' else 30

net_options['batch'] = ngpus*int(net_options['batch'])
batch_size    = net_options['batch']
clip_duration = int(net_options['clip_duration'])
max_batches   = int(net_options['max_batches'])
learning_rate = float(net_options['learning_rate'])
momentum      = float(net_options['momentum'])
decay         = float(net_options['decay'])
steps         = [float(step) for step in net_options['steps'].split(',')]
scales        = [float(scale) for scale in net_options['scales'].split(',')]

# loss parameters
loss_options               = parse_cfg(cfgfile)[1]
region_loss                = RegionLoss()
anchors                    = loss_options['anchors'].split(',')
region_loss.anchors        = [float(i) for i in anchors]
region_loss.num_classes    = int(loss_options['classes'])
region_loss.num_anchors    = int(loss_options['num'])
region_loss.anchor_step    = len(region_loss.anchors)//region_loss.num_anchors
region_loss.object_scale   = float(loss_options['object_scale'])
region_loss.noobject_scale = float(loss_options['noobject_scale'])
region_loss.class_scale    = float(loss_options['class_scale'])
region_loss.coord_scale    = float(loss_options['coord_scale'])
region_loss.batch          = batch_size
        
#Train parameters
max_epochs    = max_batches*batch_size//nsamples+1
use_cuda      = True
seed          = int(time.time())
eps           = 1e-5
best_fscore   = 0 # initialize best fscore

# Test parameters
nms_thresh    = 0.4
iou_thresh    = 0.5

if not os.path.exists(backupdir):
    os.mkdir(backupdir)
    
torch.manual_seed(seed)
if use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    torch.cuda.manual_seed(seed)

# Create model
model = YOWO(opt)

model       = model.cuda()
model       = nn.DataParallel(model, device_ids=gpu_ids) # in multi-gpu case
model.seen  = 0

logging("============================ starting =============================")
print(model)
logging(f"# of GPUs: {ngpus}, batch_size: {batch_size}")

parameters = get_fine_tuning_parameters(model, opt)
optimizer = optim.SGD(parameters, lr=learning_rate/batch_size, momentum=momentum, dampening=0, weight_decay=decay*batch_size)

kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}

# Load resume path if necessary
if opt.resume_path:
    logging("===================================================================")
    if '.pth' in opt.resume_path:
        chkpt = opt.resume_path
    else:
        chkpt_core = 'yowo_' + opt.dataset + '_' + str(clip_duration) + 'f'
        chkpt = [c for c in os.listdir(opt.resume_path) if chkpt_core in c and 'checkpoint.pth' in c]
        if chkpt:
            max_len = max([len(c) for c in chkpt])
            chkpt = sorted([c for c in chkpt if len(c)==max_len])
            chkpt = os.path.join(opt.resume_path,chkpt[-1])
    if chkpt:
        logging('loading checkpoint {}'.format(chkpt))
        checkpoint = torch.load(chkpt)
        wandb_id = checkpoint.get('wandb_id',None)
        opt.begin_epoch = checkpoint['epoch'] + 1
        best_fscore = checkpoint['fscore']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.seen = checkpoint['epoch'] * nsamples
        logging(f"Loaded model fscore: {checkpoint['fscore']}")
        logging("===================================================================")

# Set up wandb log
config = {**vars(opt), **data_options, **net_options}
if 'wandb_id' not in globals() or not wandb_id: wandb_id = wandb.util.generate_id()
logging(f'wandb_id: {wandb_id}')
wandb.init(project=f'YOWO_{opt.dataset.upper()}', entity='wuyilei516', config=config, id=wandb_id, resume="allow")
wandb.watch(model)

# Final sest up
region_loss.seen  = model.seen
processed_batches = model.seen//batch_size

init_width        = int(net_options['width'])
init_height       = int(net_options['height'])
init_epoch        = model.seen//nsamples 

def adjust_learning_rate(optimizer, batch):
    lr = learning_rate
    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        if batch >= steps[i]:
            lr = lr * scale
            if batch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr/batch_size
    return lr



def train(epoch):
    global processed_batches
    t0 = time.time()
    cur_model = model.module
    region_loss.l_x.reset()
    region_loss.l_y.reset()
    region_loss.l_w.reset()
    region_loss.l_h.reset()
    region_loss.l_conf.reset()
    region_loss.l_cls.reset()
    region_loss.l_total.reset()
    train_dataset = dataset.listDataset(basepath, trainlist, dataset_use=dataset_use, shape=(init_width, init_height),
                                        shuffle=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                        ]), 
                                        train=True, 
                                        seen=cur_model.seen,
                                        batch_size=batch_size,
                                        clip_duration=clip_duration,
                                        num_workers=num_workers)
    rs = RandomSampler(train_dataset,replacement=True, num_samples=int(len(train_dataset)/train_n_sample_from))
    train_loader = torch.utils.data.DataLoader(train_dataset,sampler=rs,
                                               batch_size=batch_size, shuffle=False, **kwargs)

    lr = adjust_learning_rate(optimizer, processed_batches)
    nbatch = len(train_loader)
    logging('training at epoch %d, lr %f' % (epoch, lr))
    logging('total # of batches %d' % (nbatch))

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        
        lr = adjust_learning_rate(optimizer, processed_batches)
        processed_batches = processed_batches + 1

        if use_cuda:
            data = data.cuda()

        optimizer.zero_grad()
        output = model(data)
        region_loss.seen = region_loss.seen + data.data.size(0)
        loss = region_loss(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx%20 == 0:
            logging(f'epoch: {epoch}, batch: {batch_idx}/{nbatch}, lr:{lr}, loss: {loss.item()}')
            wandb.log({"loss": loss})
            wandb.log({"lr": lr})
        
        # save result every 1000 batches
        if processed_batches % 500 == 0: # From time to time, reset averagemeters to see improvements
            region_loss.l_x.reset()
            region_loss.l_y.reset()
            region_loss.l_w.reset()
            region_loss.l_h.reset()
            region_loss.l_conf.reset()
            region_loss.l_cls.reset()
            region_loss.l_total.reset()

    t1 = time.time()
    logging('trained with %f samples/s' % ((nbatch*batch_size)/(t1-t0)))



def test(epoch):
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i
    test_dataset = dataset.listDataset(basepath, testlist, dataset_use=dataset_use, shape=(init_width, init_height),
                                       shuffle=False,
                                       transform=transforms.Compose([
                                           transforms.ToTensor()
                                       ]), train=False)
    test_rs = RandomSampler(test_dataset, replacement=True, num_samples=int(len(test_dataset)/test_n_sample_from))
    test_loader = torch.utils.data.DataLoader(test_dataset, sampler=test_rs,
                                              batch_size=batch_size, shuffle=False, **kwargs)

    num_classes = region_loss.num_classes
    anchors     = region_loss.anchors
    num_anchors = region_loss.num_anchors
    conf_thresh_valid = 0.005
    total       = 0.0
    proposals   = 0.0
    correct     = 0.0
    fscore = 0.0

    correct_classification = 0.0
    total_detected = 0.0

    nbatch      = len(test_loader) #file_lines(testlist) // batch_size

    logging('validation at epoch %d' % (epoch))
    model.eval()

    for batch_idx, (frame_idx, data, target) in enumerate(test_loader):
        if use_cuda:
            data = data.cuda()
        with torch.no_grad():
            output = model(data).data
            all_boxes = get_region_boxes(output, conf_thresh_valid, num_classes, anchors, num_anchors, 0, 1)
            for i in range(output.size(0)):
                boxes = all_boxes[i]
                boxes = nms(boxes, nms_thresh)
                if dataset_use == 'ucf101-24':
                    detection_path = os.path.join('ucf_detections', 'detections_'+str(epoch), frame_idx[i])
                    current_dir = os.path.join('ucf_detections', 'detections_'+str(epoch))
                    if not os.path.exists('ucf_detections'):
                        os.mkdir('ucf_detections')
                    if not os.path.exists(current_dir):
                        os.mkdir(current_dir)
                else:
                    detection_path = os.path.join('jhmdb_detections', 'detections_'+str(epoch), frame_idx[i])
                    current_dir = os.path.join('jhmdb_detections', 'detections_'+str(epoch))
                    if not os.path.exists('jhmdb_detections'):
                        os.mkdir('jhmdb_detections')
                    if not os.path.exists(current_dir):
                        os.mkdir(current_dir)

                with open(detection_path, 'w+') as f_detect:
                    for box in boxes:
                        x1 = round(float(box[0]-box[2]/2.0) * 320.0)
                        y1 = round(float(box[1]-box[3]/2.0) * 240.0)
                        x2 = round(float(box[0]+box[2]/2.0) * 320.0)
                        y2 = round(float(box[1]+box[3]/2.0) * 240.0)

                        det_conf = float(box[4])
                        for j in range((len(box)-5)//2):
                            cls_conf = float(box[5+2*j].item())

                            if type(box[6+2*j]) == torch.Tensor:
                                cls_id = int(box[6+2*j].item())
                            else:
                                cls_id = int(box[6+2*j])
                            prob = det_conf * cls_conf

                            f_detect.write(str(int(box[6])+1) + ' ' + str(prob) + ' ' + str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + '\n')
                truths = target[i].view(-1, 5)
                num_gts = truths_length(truths)
        
                total = total + num_gts
    
                for i in range(len(boxes)):
                    if boxes[i][4] > 0.25:
                        proposals = proposals+1

                for i in range(num_gts):
                    box_gt = [truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0, 1.0, truths[i][0]]
                    best_iou = 0
                    best_j = -1
                    for j in range(len(boxes)):
                        iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
                        if iou > best_iou:
                            best_j = j
                            best_iou = iou

                    if best_iou > iou_thresh:
                        total_detected += 1
                        if int(boxes[best_j][6]) == box_gt[6]:
                            correct_classification += 1

                    if best_iou > iou_thresh and int(boxes[best_j][6]) == box_gt[6]:
                        correct = correct+1

            precision = 1.0*correct/(proposals+eps)
            recall = 1.0*correct/(total+eps)
            fscore = 2.0*precision*recall/(precision+recall+eps)
            if batch_idx%20 == 0:
                logging("[%d/%d] precision: %f, recall: %f, fscore: %f" % (batch_idx, nbatch, precision, recall, fscore))

    classification_accuracy = 1.0 * correct_classification / (total_detected + eps)
    localization_recall = 1.0 * total_detected / (total + eps)

    logging("Classification accuracy: %.3f" % classification_accuracy)
    logging("Localization recall: %.3f" % localization_recall)

    wandb.log({"Precision": precision})
    wandb.log({"Recall": recall})
    wandb.log({"Fscore": fscore})
    wandb.log({"Classification accuracy": classification_accuracy})
    wandb.log({"Localization recall": localization_recall})

    return fscore




if opt.evaluate:
    logging('evaluating ...')
    test(0)
else:
    for epoch in range(opt.begin_epoch, opt.end_epoch + 1):
        # Train the model for 1 epoch
        train(epoch)

        # Validate the model
        fscore = test(epoch)

        is_best = fscore > best_fscore
        if is_best:
            logging(f"New best fscore is achieved: {fscore}" )
            logging(f"Previous fscore was: {best_fscore}")
            best_fscore = fscore

        # Save the model to backup directory
        state = {
            'wandb_id': wandb_id,
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'fscore': fscore
            }
        save_checkpoint(state, is_best, backupdir, opt.dataset, clip_duration, epoch)
        logging('Weights are saved to backup directory: %s' % (backupdir))
