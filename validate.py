from torchvision import transforms
from models import mvit_kinetics600, mvit_imagenet
import torch, torchvision
from torchvision.datasets.folder import ImageFolder
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import os, json
from timm.data import create_transform
import matplotlib.pyplot as plt
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
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
def eval_corruptions(root = "/home/jasper/imagenet-30/corruptions"):
    transforms = create_transform(224)
    model1 = mvit_kinetics600(pretrained=True)

    model2 = mvit_imagenet(pretrained=True)


    k_accs = []
    i_accs = []
    corruptions = []
    for corruption in os.scandir(root):
        if corruption.is_dir():
            kcorr_accs = []
            icorr_accs = []
            splits = []
            for split in os.scandir(corruption):
                if split.is_dir():
                    dataset = ImageFolder(split.path, transform=transforms)
                    loader = DataLoader(dataset, batch_size = 64)
                    acc1 = validate(loader, model1, 'cuda')

                    acc2 = validate(loader, model2, 'cuda')
                    print("Model: Kinetics, corruption:", corruption.name, "difficulty: ", split.name, "accuracy: ", acc1)
                    print("Model: Imagenet, corruption:", corruption.name, "difficulty: ", split.name, "accuracy: ", acc2)
                    splits.append(split.name)
                    kcorr_accs.append(acc1)
                    icorr_accs.append(acc2)
            
            print("Kinetics Average "+corruption.name + " Accuracy", np.mean(kcorr_accs) )
            print("Imagenet Average "+corruption.name + " Accuracy", np.mean(icorr_accs) )
            plt.figure()
            index = np.arange(5)
            bar_width = 0.35
            plt.bar(index, kcorr_accs, width = bar_width)
            plt.bar(index+bar_width, icorr_accs, width = bar_width)
            plt.xticks(index + bar_width/2, labels = splits)
            plt.xlabel("difficulty")
            plt.ylabel("accuracy")
            plt.title(corruption.name + "model accuracy comparison")
            plt.legend(["Kinetics", "Imagenet"])
            plt.savefig(os.path.join("/home/jasper/data/eval/corruptions",corruption.name))

            corruptions.append(corruption.name)
            k_accs.append(kcorr_accs)
            i_accs.append(icorr_accs)
    k_means = [np.mean(x) for x in k_accs]
    i_means = [np.mean(x) for x in i_accs]
    plt.figure()
    index = np.arange(5)
    bar_width = 0.35
    plt.bar(index, k_means, width = bar_width)
    plt.bar(index+bar_width, i_means, width = bar_width)
    plt.xticks(index + bar_width/2, labels = corruptions)
    plt.xlabel("difficulty")
    plt.ylabel("average corruptionaccuracy")
    plt.title("Corruptions model accuracy comparison")
    plt.legend(["Kinetics", "Imagenet"])
    plt.savefig("/home/jasper/data/eval/corruptions/overall")



    print("Kinetics Average Overall Accuracy: ", np.mean(k_accs))
    print("Imagenet Average Overall Accuracy: ", np.mean(i_accs))
def make_dataset(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                if is_valid_file(fname):
                    path = os.path.join(root, fname)
                    item = path, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

    return instances

class RenditionsFolder(ImageFolder):
    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        if class_to_idx is None:
            raise ValueError(
                "The class_to_idx parameter cannot be None."
            )
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)
        
def eval_renditions(root = "/home/jasper/imagenet-30/renditions"):
    transforms = create_transform(224)
    model1 = mvit_kinetics600(pretrained=True)

    model2 = mvit_imagenet(pretrained=True)


    dataset = RenditionsFolder(root, transform=transforms, is_valid_file = lambda x: True)
    loader = DataLoader(dataset, batch_size = 64)
    acc1 = validate(loader, model1, 'cuda')

    acc2 = validate(loader, model2, 'cuda')
   
    print("Kinetics Average Overall Accuracy: ", acc1)
    print("Imagenet Average Overall Accuracy: ", acc2)
        




def create_symlinks(directory):
    root = "/home/jasper/imagenet-30/corruptions"
    corruptions = sorted([(entry.name, entry.path,os.path.join(root, entry.name)) for entry in os.scandir(directory) if entry.is_dir()], key = lambda x:x[0])
    all_splits = []
    for corruption in corruptions:
        splits = sorted([(os.path.join(corruption[1], entry.name), os.path.join(corruption[2],  entry.name), entry.path) for entry in os.scandir(corruption[1]) if entry.is_dir()], key = lambda x:x[0])
        all_splits.extend(splits)
    with open('/home/jasper/imagenet-30/classes.json') as json_file:
        classes = json.load(json_file)
    for split in all_splits:

        os.makedirs(split[1],exist_ok =True)
        for entry in os.scandir(split[0]):
            if entry.is_dir() and entry.name in classes.keys():
#                print(os.path.join(split[1], entry.name))
                os.symlink(entry.path, os.path.join(split[1], entry.name))
def create_renditions(directory):
    root = "/home/jasper/imagenet-30/renditions"
    with open('/home/jasper/imagenet-30/classes.json') as json_file:
        classes = json.load(json_file)

    for entry in os.scandir(directory):
        if entry.is_dir() and entry.name in classes.keys():
               # print(os.path.join(root, entry.name))
            os.symlink(entry.path, os.path.join(root, entry.name))
def fill_classes(root = "/home/jasper/imagenet-30/renditions"):
    with open('/home/jasper/imagenet-30/classes.json') as json_file:
        classes = json.load(json_file)
    for cls in classes.keys():
        path = os.path.join(root, cls)
        if not os.path.isdir(path):
            os.mkdir(path)


def validate(val_loader, model, device):
  model.eval()
  num_top1 = 0
  num_top5 = 0
  model = model.to(device)
  num_total = len(val_loader.dataset)

  with torch.no_grad():
    for i, (images, target) in enumerate(val_loader):
      images = images.to(device)
      target = target.to(device)
      output = model(images)
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
  return (num_top1/num_total).item()

#create_symlinks("/home/data/imagenet/corruptions/")
#create_renditions("/home/data/imagenet/renditions/")
#eval_corruptions()
eval_renditions()
#fill_classes()
