import os
import cv2
import numpy as np
import random
import albumentations as A
from pathlib import Path


transforms_obj = A.Compose([
    A.Rotate(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.GaussNoise(var_limit=(5.0, 10.0), p=0.5),
    A.GaussianBlur(blur_limit=(3,7), p=0.3),
    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.2),
                               contrast_limit=0.1,
                               brightness_by_max=True,
                               always_apply=False,
                               p=0.5)
])


def generate(backgrounds_dir: Path,
    objects_dir: Path,
    masks_dir: Path,
    compositions_dir: Path,
    yolo_labels_dir: Path,
    n_samples: int
):
    object_paths = [os.path.join(root, f) for root, dirs, files in os.walk(backgrounds_dir) for f in files]
    label_img_dict = _create_label_img_dict(objects_dir)

    for i in range(n_samples):
        for object_path in object_paths:
            img_bg = cv2.imread(object_path)
            img_comp, mask_comp, labels_comp, obj_areas = _create_composition(img_bg,
                                                                             objects_dir,
                                                                             masks_dir,
                                                                             label_img_dict,
                                                                             max_objs=30,
                                                                             overlap_degree=0.1,
                                                                             max_attempts_per_obj=20)

            object_filename = str(i) + '.' + object_path.split('.')[-1]
            cv2.imwrite(os.path.join(compositions_dir, object_filename), img_comp)

            annotation_filename = object_filename.split('.')[0] + '.txt'
            annotation_file = open(os.path.join(yolo_labels_dir, annotation_filename), 'w') #annotation file
            yolo_labels_list = _create_yolo_annotations(mask_comp, labels_comp)
            for label in yolo_labels_list:
                annotation_file.write(f"{label[0]+1} {label[1]} {label[2]} {label[3]} {label[4]}\n")
            annotation_file.close()

def _create_label_img_dict(objects_dir):
    '''
    Create dictionary with yolo classes as keys and belonging image filenames as values
    :param objects_dir: dir with objects
    :return: dictionary
    '''
    label_img_dict = {}
    object_filenames = [f for root, dirs, files in os.walk(objects_dir) for f in files]
    targets = [filename.split('_')[0] for filename in object_filenames]

    for target in set(targets):
        label_img_dict[int(target)] =[]

    for filename in object_filenames:
        label_img_dict[int(filename.split('_')[0])].append(filename)

    return label_img_dict

def _get_img_and_mask(img_path, mask_path):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)

    mask_b = mask[:, :, 0] == 1  # This is boolean mask
    mask = mask_b.astype(np.uint8)  # This is binary mask

    return img, mask


def _resize_transform_obj(img, mask, transforms=False):
    h, w = mask.shape[0], mask.shape[1]

    resize_koef = random.uniform(0.5, 1)
    h_new = int(h * resize_koef)
    w_new = int(w * resize_koef)
    transform_resize = A.Resize(h_new, w_new, interpolation=1, p=1)

    transformed_resized = transform_resize(image=img, mask=mask)
    img_t = transformed_resized["image"]
    mask_t = transformed_resized["mask"]

    if transforms:
        transformed = transforms_obj(image=img_t, mask=mask_t)
        img_t = transformed["image"]
        mask_t = transformed["mask"] #Mask doesn't have to be transformed if augmentations are not geometric

    return img_t, mask_t


def _add_obj(img_comp, mask_comp, img, mask, x, y, idx):
    '''
    img_comp - composition of objects (background + objects)
    mask_comp - composition of objects` masks
    img - image of object
    mask - binary mask of object
    x, y - coordinates where center of img is placed
    Function returns img_comp in CV2 RGB format + mask_comp
    '''
    h_comp, w_comp = img_comp.shape[0], img_comp.shape[1]
    h, w = img.shape[0], img.shape[1]

    x = x - int(w / 2)
    y = y - int(h / 2)

    mask_b = mask == 1
    mask_rgb_b = np.stack([mask_b, mask_b, mask_b], axis=2)

    if x >= 0 and y >= 0:
        h_part = h - max(0, y + h - h_comp)  # h_part - part of the image which gets into the frame of img_comp along y-axis
        w_part = w - max(0, x + w - w_comp)  # w_part - part of the image which gets into the frame of img_comp along x-axis

        img_comp[y:y + h_part, x:x + w_part, :] = img_comp[y:y + h_part, x:x + w_part, :] * \
                                                  ~mask_rgb_b[0:h_part, 0:w_part, :] + \
                                                  (img * mask_rgb_b)[0:h_part, 0:w_part, :]
        mask_comp[y:y + h_part, x:x + w_part] = mask_comp[y:y + h_part, x:x + w_part] * \
                                                ~mask_b[0:h_part, 0:w_part] + \
                                                (idx * mask_b)[0:h_part, 0:w_part]
        mask_added = mask[0:h_part, 0:w_part]

    elif x < 0 and y < 0:
        h_part = h + y
        w_part = w + x

        img_comp[0:0 + h_part, 0:0 + w_part, :] = img_comp[0:0 + h_part, 0:0 + w_part, :] * \
                                                  ~mask_rgb_b[h - h_part:h, w - w_part:w, :] + \
                                                  (img * mask_rgb_b)[h - h_part:h, - w_part:w,:]
        mask_comp[0:0 + h_part, 0:0 + w_part] = mask_comp[0:0 + h_part, 0:0 + w_part] * \
                                                ~mask_b[h - h_part:h, w - w_part:w] + \
                                                (idx * mask_b)[h - h_part:h, w - w_part:w]
        mask_added = mask[h - h_part:h, w - w_part:w]

    elif x < 0 and y >= 0:
        h_part = h - max(0, y + h - h_comp)
        w_part = w + x

        img_comp[y:y + h_part, 0:0 + w_part, :] = img_comp[y:y + h_part, 0:0 + w_part, :] * \
                                                  ~mask_rgb_b[0:h_part, w - w_part:w, :] + \
                                                  (img * mask_rgb_b)[0:h_part, w - w_part:w,:]
        mask_comp[y:y + h_part, 0:0 + w_part] = mask_comp[y:y + h_part, 0:0 + w_part] * \
                                                ~mask_b[0:h_part, w - w_part:w] + \
                                                (idx * mask_b)[0:h_part, w - w_part:w]
        mask_added = mask[0:h_part, w - w_part:w]

    elif x >= 0 and y < 0:
        h_part = h + y
        w_part = w - max(0, x + w - w_comp)

        img_comp[0:0 + h_part, x:x + w_part, :] = img_comp[0:0 + h_part, x:x + w_part, :] * \
                                                  ~mask_rgb_b[h - h_part:h,0:w_part, :] + \
                                                  (img * mask_rgb_b)[h - h_part:h,0:w_part, :]
        mask_comp[0:0 + h_part, x:x + w_part] = mask_comp[0:0 + h_part, x:x + w_part] * \
                                                ~mask_b[h - h_part:h,0:w_part] + \
                                                (idx * mask_b)[h - h_part:h,0:w_part]
        mask_added = mask[h - h_part:h, 0:w_part]

    return img_comp, mask_comp, mask_added


def _check_areas(mask_comp, obj_areas, i, overlap_degree=0.0):
    '''
    Check if new object overlaps with already existing objects.
    :param mask_comp: Composition of masks and background
    :param obj_areas: Areas of objects
    :i: Counter of labels
    :param overlap_degree: Ratio of allowed overlap
    :return: Boolean value describing if there is overlap or not
    '''
    obj_ids = np.unique(mask_comp).astype(np.uint8)[1:-1]
    masks = mask_comp == obj_ids[:, None, None]

    ok = True

    if i == 1:
        ok = True
    else:
        if len(np.unique(mask_comp)) != np.max(mask_comp) + 1:
            ok = False
            return ok

        for idx, mask in enumerate(masks):
            if np.count_nonzero(mask) / obj_areas[idx] < 1 - overlap_degree:
                ok = False
                break

    return ok


def _create_composition(img_comp_bg,
                       path_imgs,
                       path_masks,
                       label_img_dict,
                       max_objs=20,
                       overlap_degree=0.0,
                       max_attempts_per_obj=10):
    '''

    :param img_comp_bg: Composition of the background and previously added  objects
    :param path_imgs: Path to objects
    :param path_masks: Path to masks
    :param label_img_dict: Dictionary with object/masks filenames as values and their indexes as keys
    :param max_objs: Maximum number of objects added to the background
    :param overlap_degree: Degree of overlap of objects - default = 0
    :param max_attempts_per_obj: Maximum number of attempts for object to be added to the background. If object
    overlaps with other already existing objects or background boundary, the attempt is unsuccessful
    :return: img_comp - composition of objects and background, mask_comp - composition of masks and background,
    label_comp - composition of labels
    '''
    img_comp = img_comp_bg.copy()
    h, w = img_comp.shape[0], img_comp.shape[1]
    mask_comp = np.zeros((h, w), dtype=np.uint8)

    obj_areas = []
    labels_comp = []
    num_objs = np.random.randint(max_objs) + 2

    i = 1 # _add_obj use i to create masks with labeled pixels

    for _ in range(1, num_objs):
        idx = random.choice(list(label_img_dict.keys()))

        for _ in range(max_attempts_per_obj):


            img_path = os.path.join(path_imgs, random.choice(label_img_dict[idx]))
            mask_path = os.path.join(path_masks, random.choice(label_img_dict[idx]))
            img, mask = _get_img_and_mask(img_path, mask_path)

            x, y = np.random.randint(w), np.random.randint(h)
            img, mask = _resize_transform_obj(img,
                                             mask,
                                             transforms=transforms_obj)
            h_img = img.shape[0]
            w_img = img.shape[1]
            x_top = x - int(h_img / 2)
            y_top = y - int(w_img / 2)

            if x_top >= 0 and y_top >= 0 and (x_top + w_img) <= w and (y_top + h_img) <= h:
                img_comp_prev, mask_comp_prev = img_comp.copy(), mask_comp.copy()
                img_comp, mask_comp, mask_added = _add_obj(img_comp,
                                                          mask_comp,
                                                          img,
                                                          mask,
                                                          x,
                                                          y,
                                                          i)
                ok = _check_areas(mask_comp, obj_areas, i, overlap_degree)
                if ok:
                    obj_areas.append(np.count_nonzero(mask_added))
                    labels_comp.append(idx)
                    i += 1
                    break
                else:
                    img_comp, mask_comp = img_comp_prev.copy(), mask_comp_prev.copy()

    return img_comp, mask_comp, labels_comp, obj_areas


def _create_yolo_annotations(mask_comp, labels_comp):
    '''
    Create annotations in format acceptable by yolo object detection model.
    :param mask_comp: Composition of masks
    :param labels_comp: Composition of labels
    :return: List of annotations list per object placed in composition of background and object
    '''
    comp_w, comp_h = mask_comp.shape[1], mask_comp.shape[0]

    obj_ids = np.unique(mask_comp).astype(np.uint8)[1:]
    masks = mask_comp == obj_ids[:, None, None]

    annotations_yolo = []
    for idx, val in enumerate(labels_comp):
        pos = np.where(masks[idx])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])

        xc = (xmin + xmax) / 2
        yc = (ymin + ymax) / 2
        w = xmax - xmin
        h = ymax - ymin

        annotations_yolo.append([val - 1,
                                 round(xc / comp_w, 5),
                                 round(yc / comp_h, 5),
                                 round(w / comp_w, 5),
                                 round(h / comp_h, 5)])

    return annotations_yolo