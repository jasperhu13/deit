from models import mvit_kinetics600
import torch, torchvision
import numpy as np
import torch.nn as nn
import os, json

def get_imagenet_inds():
  with open('/content/drive/MyDrive/Colab Notebooks/research/multiscale/Data/imagenet-30/classes.json') as json_file:
    classes = json.load(json_file)
  indices = []
  root_dir = '/content/drive/MyDrive/Colab Notebooks/research/multiscale/Data/imagenet-30/val'


  verify_dict = {}
  classes_1k = []
  with open('/content/drive/MyDrive/Colab Notebooks/research/multiscale/map_clsloc.txt') as file:
    for line in file:
      (fn, idx, lab) = line.split()
      classes_1k.append(fn)
      verify_dict[fn] = lab
  classes_1k = sorted(classes_1k)
  cls_all_map = {}
  i = 0
  for cls in classes_1k:
    cls_all_map[cls] = i
    i += 1

  cls_idx_map = {}
  for cls in classes.keys():
    cls_idx_map[cls] = cls_all_map[cls]
  inds = np.array(sorted(list(set(cls_idx_map.values()))))
  return inds
def get_imagenet30_idxmap():
  with open('/home/jasper/imagenet-30/classes.json') as json_file:
    classes = json.load(json_file)

  verify_dict = {}
  classes_1k = []
  with open('/home/jasper/deit/map_clsloc.txt') as file:
    for line in file:
      (fn, idx, lab) = line.split()
      classes_1k.append(fn)
      verify_dict[fn] = lab
  classes_1k = sorted(classes_1k)
  cls_all_map = {}
  i = 0
  for cls in classes_1k:
    cls_all_map[cls] = i
    i += 1

  cls_idx_map = {}
  cls_ind_pair = []
  for cls in classes.keys():
    cls_ind_pair.append((cls, cls_all_map[cls]))
  cls_idx_map = {pair[0]:i for i, pair in enumerate(sorted(cls_ind_pair, key = lambda x:x[1]))}

  return cls_idx_map
def find_classes(directory) :
    corruptions = sorted([(entry.name, entry.path) for entry in os.scandir(directory) if entry.is_dir()], key = lambda x:x[0])
    all_splits = []
    for corruption in corruptions:
        splits = sorted([(os.path.join(corruption[1],  entry.name), entry.path) for entry in os.scandir(corruption[1]) if entry.is_dir()], key = lambda x:x[0])
        all_splits.extend(splits)
    classes = get_imagenet30_idxmap()
    all_classes = []
    for split in all_splits:
        some_classes = sorted([(os.path.join(split[1], entry.name), entry.name, entry.path) for entry in os.scandir(split[0]) if entry.is_dir() and entry.name in classes.keys()], key = lambda x:x[0])
        all_classes.extend(some_classes)
    if not all_classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
    
    all_class_to_idx = {cls_name[0]: classes[cls_name[1]] for cls_name in all_classes}
    return list(classes.keys()), all_class_to_idx
def create_symlinks():
    corruptions = sorted([(entry.name, entry.path) for entry in os.scandir(directory) if entry.is_dir()], key = lambda x:x[0])
    all_splits = []
    for corruption in corruptions:
        splits = sorted([(os.path.join(corruption[1],  entry.name), entry.path) for entry in os.scandir(corruption[1]) if entry.is_dir()], key = lambda x:x[0])
        all_splits.extend(splits)
    with open('/home/jasper/imagenet-30/classes.json') as json_file:
        classes = json.load(json_file)
    for split in all_splits:
        for entry in os.scandir(split[0]):
            if entry.is_dir() and entry.name in classes.keys():
                print(entry.path)

def validate(val_loader, model, device):
  model.eval()
  num_top1 = 0
  num_top5 = 0
  num_total = len(val_loader.dataset)

  with torch.no_grad():
    for i, (images, target) in enumerate(val_loader):
      images = images.to(device)
      target = target.to(device)
      output = model(images.unsqueeze(0))
      #output = output[:, inds]
      _, top1 = torch.topk(output, 1, dim = 1)
    #  _, top5 = torch.topk(output, 5, dim = 1)
     # print(top1[:,0])
     # print(target)
      top1_cts = torch.sum(torch.eq(top1[:,0], target))
     # top5_cts = torch.sum(torch.eq(top5, target))
  
      num_top1 += top1_cts
     # print(num_top1)
     # num_top5 += top5_cts
      
  return num_top1/num_total