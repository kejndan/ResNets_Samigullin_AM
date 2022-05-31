from PIL import Image
import os
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
def print_dataset_stats(path):
    for type_set in os.listdir(path):
        path_type_set = os.path.join(path, type_set)
        if not os.path.isdir(path_type_set):
            continue
        count_nb_img_type_set = 0
        stats  = []
        for cls in os.listdir(path_type_set):
            path_cls = os.path.join(path_type_set, cls)
            if not os.path.isdir(path_cls):
                continue
            print(f'Class {cls} - {len(os.listdir(path_cls))} images')
            count_nb_img_type_set += len(os.listdir(path_cls))
            stats.append(len(os.listdir(path_cls)))
        print(f'{type_set} - {count_nb_img_type_set} images')
        sns.barplot(x=np.arange(12),y=stats)
        plt.show()


def delete_small_images(path, img_thr):
    for cls in os.listdir(path):
        path_cls = os.path.join(path, cls)
        if os.path.isdir(path_cls):
            for name_img in os.listdir(path_cls):
                path_img = os.path.join(path_cls, name_img)
                img = Image.open(os.path.join(path_cls, path_img))
                if not img.height >= img_thr[0] and img.width >= img_thr[1]:
                    os.remove(path_img)

def delete_img_by_std(path, thr):
    for cls in os.listdir(path):
        path_cls = os.path.join(path, cls)
        if os.path.isdir(path_cls):
            for name_img in os.listdir(path_cls):
                path_img = os.path.join(path_cls, name_img)
                img = np.asarray(Image.open(os.path.join(path_cls, path_img)))
                if np.std(img) < thr:
                    os.remove(path_img)

def plot_dist_square(path):
    sqrs = []
    c = 0
    for cls in os.listdir(path):
        path_cls = os.path.join(path, cls)
        if os.path.isdir(path_cls):
            for name_img in os.listdir(path_cls):
                path_img = os.path.join(path_cls, name_img)
                img = np.asarray(Image.open(os.path.join(path_cls, path_img)))

                sqrs.append(np.std(img))

    sns.displot(x=sqrs)
    plt.show()
    print(c)


def del_by_conf(path_to_ds, conf, thr):
    with open(conf, 'rb') as f:
        confidences_error = pickle.load(f)
    c = 0
    for conf,path in confidences_error:
        if conf > thr:
            if os.path.exists(os.path.join(path_to_ds, path.split('/')[-2],path.split('/')[-1])):
                os.remove(os.path.join(path_to_ds, path.split('/')[-2],path.split('/')[-1]))
                c += 1
    print(c)



if __name__ == '__main__':
    # print_dataset_stats('/Users/adels/PycharmProjects/ResNets_Samigullin_AM/data/custom_set')
    # delete_img_by_std('/Users/adels/PycharmProjects/ResNets_Samigullin_AM/data/custom_set_clean/train', 5)
    # delete_small_images('/Users/adels/PycharmProjects/ResNets_Samigullin_AM/data/custom_set_clean/train', (200, 200))
    # print_dataset_stats('/Users/adels/PycharmProjects/ResNets_Samigullin_AM/data/custom_set_clean')
    # plot_dist_square('/Users/adels/PycharmProjects/ResNets_Samigullin_AM/data/custom_set_clean/train')
    # print_dataset_stats('/Users/adels/PycharmProjects/ResNets_Samigullin_AM/data/custom_set_clean')
    # del_by_conf('/Users/adels/PycharmProjects/ResNets_Samigullin_AM/data/custom_set_clean/train','/Users/adels/PycharmProjects/ResNets_Samigullin_AM/confidences_error_w_path_train.pickle', 0.7)
    print_dataset_stats('/home/kirito/Datasets/custom_set_clean')