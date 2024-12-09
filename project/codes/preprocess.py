from utils.preprocess import *

# define important paths
local = False
if local:
    data_path = '/Users/masoud/Documents/Education/Alphanumerics Lab/Projects/data/cell_tracking'
else:
    data_path = '/labs/samenilab/team/masoud_nateghi/data/cell_tracking'

dataset_path = '../../dataset'
os.makedirs(dataset_path, exist_ok=True)

# preprocessing parameters
frame_size = 2048
patch_size = 128
threshold = 20
kernel_size = 9

# check for the previous preprocessed images in order to skip preprocessing
preprocessed_files = os.listdir(dataset_path)

for root, dirs, files in os.walk(data_path):
    if 'CH2.tif' in files and 'CH3.tif' in files:
            cell_type, experiment, position = root.split(os.sep)[-3:]
            file_dir = f'{cell_type[0]}e{int(experiment[-1]):02d}p{int(position[-1]):02d}'
            if file_dir in preprocessed_files:  # check if the file is already preprocessed
                print(f'{cell_type}----{experiment}----{position} is already preprocessed.')
                continue


            # load the image and mask (shape: (n_frames, frame_size, frame_size))
            print(f'{cell_type}----{experiment}----{position} is loading...')
            frames_mask = load_tiff(os.path.join(root, 'CH2.tif'))
            frames_img = load_tiff(os.path.join(root, 'CH3.tif'))

            # preprocess the image and mask (shape: (n_frames, n_patch**2, patch_size, patch_size, 2))
            print(f'{cell_type}----{experiment}----{position} is being preprocessed...')
            data = preprocess(frames_img=frames_img,
                              frames_mask=frames_mask,
                              frame_size=frame_size,
                              patch_size=patch_size,
                              threshold=threshold,
                              kernel_size=kernel_size)

            # save each frame of the data as a separate file (shape: (n_patch**2, patch_size, patch_size, 2))
            print(f'{cell_type}----{experiment}----{position} is being saved...')
            save_path = os.path.join(dataset_path, file_dir)
            os.makedirs(save_path, exist_ok=True)
            save_data(data, save_path)
