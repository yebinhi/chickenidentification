# # Creating Train / Val / Test folders (One time use)
import os
import random
import shutil

root_dir = '../../data/chicken_state/state_2_frolicking'  # data root path
classes_dir = ['sitting', 'standing']  # total labels
data_set_dir = ['train', 'val', 'test']

train_ratio = 0.8
val_ratio = 0.15
test_ratio = 0.05

# total frame
total_frame_end = 500
train_frame_end = total_frame_end*train_ratio
val_frame_end = total_frame_end * (train_ratio + val_ratio)
test_ratio_end = total_frame_end

# create folders for train, val and test
for ds_dir in data_set_dir:
    for cls in classes_dir:
        dir_path = root_dir + '/' + ds_dir + '/' + cls
        print(dir_path)
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

# src = root_dir + 'state/sitting'  # Folder to copy images from

# remove some image from standing
num_sitting = len(os.listdir(root_dir+'/sitting'))
num_standing = len(os.listdir(root_dir+'/standing'))
print(num_sitting)
print(num_standing)
items = os.listdir(root_dir+'/standing')
number = num_standing - num_sitting
print(number)
#
for i in range(number):
    r_item = random.choice(items)
    items.remove(r_item)
    os.remove(root_dir+'/standing/'+r_item)


num_sitting = len(os.listdir(root_dir+'/sitting'))
num_standing = len(os.listdir(root_dir+'/standing'))
print(num_sitting)
print(num_standing)


for cls in classes_dir:
    src = root_dir + '/' + cls
    allFileNames = os.listdir(src)
    i = 0
    for f in allFileNames:
        unique_path = src + '/' + f
        print(f.split('_'))
        frame_number = f.split('_')[1]
        f_n = int(frame_number)
        if f_n <= train_frame_end:
            shutil.copy(unique_path, root_dir + '/train/' + cls)
        if train_frame_end < f_n <= val_frame_end:
            shutil.copy(unique_path, root_dir + '/val/' + cls)
        if val_frame_end < f_n <= test_ratio_end:
            shutil.copy(unique_path, root_dir + '/test/' + cls)

# generate a report
train_sitting = len(os.listdir(root_dir+'/sitting'))
train_standing = len(os.listdir(root_dir+'/standing'))
train_total = train_sitting + train_standing
print('train images total: %d' % train_total)
print('train_sitting: %d' % train_sitting)
print('train_standing: %d' % train_standing)
val_sitting = len(os.listdir(root_dir+'/val/sitting'))
val_standing = len(os.listdir(root_dir+'/val/standing'))
val_total = val_sitting + val_standing
print('val images: %d' % val_total)
print('val_sitting: %d' % val_sitting)
print('val_standing: %d' % val_standing)
test_sitting = len(os.listdir(root_dir+'/test/sitting'))
test_standing = len(os.listdir(root_dir+'/test/standing'))
test_total = test_sitting + test_standing
print('test images: %d' % test_total)
print('test_sitting: %d' % test_sitting)
print('test_standing: %d' % test_standing)


