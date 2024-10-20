import numpy as np
import os
import tifffile as tiff
import scipy.io

# Function to load ground truth data from .mat files
def load_ground_truth(gt_file_path):
    # Load the .mat file using scipy.io
    ground_truth = scipy.io.loadmat(gt_file_path)
    
    # Extract the 'gt_frame' field which contains the frame-level annotations for abnormal events
    return ground_truth['gt_frame'].flatten()  # Flatten it to a 1D list if it's nested

# Main dataset loader function
def get_dataset(data_dir, seed=None):
    if seed is not None:
        np.random.seed(seed)

    def load_ucsd_data(data_dir, train=True):
        data = []
        labels = []
        
        # Set directory for train or test
        if train:
            path = os.path.join(data_dir, 'Train')
        else:
            path = os.path.join(data_dir, 'Test')

        # Loop through each folder (each clip)
        for folder in sorted(os.listdir(path)):
            folder_path = os.path.join(path, folder)
            if os.path.isdir(folder_path):
                # Load ground truth only for the test set
                gt_file_path = os.path.join(data_dir, "Test", f"{folder}_gt.mat") if not train else None
                if not train and os.path.exists(gt_file_path):
                    gt_frames = load_ground_truth(gt_file_path)
                else:
                    gt_frames = None

                for img_file in sorted(os.listdir(folder_path)):
                    if img_file.endswith('.tif'):
                        img_path = os.path.join(folder_path, img_file)
                        try:
                            # Load image using tifffile
                            img = tiff.imread(img_path)
                            img = img / 255.0  # Normalize pixel values
                            img_flattened = img.flatten()
                            data.append(img_flattened)

                            # Assign labels: 0 for normal, 1 for anomaly (for the test set)
                            if train:
                                labels.append(0)  # Normal for training set
                            else:
                                labels.append(get_label_for_frame(folder, img_file, gt_frames))  # Use ground truth for test
                        except tiff.TiffFileError as e:
                            print(f"Error reading {img_file}: {e}")
                            continue

        return np.array(data), np.array(labels)

    # Function to get test set labels (0 for normal, 1 for anomaly)
    def get_label_for_frame(folder, img_file, gt_frames):
        if gt_frames is None:
            return 0  # No ground truth file, assume normal
        
        # Extract the frame number from the image file (assuming filenames like '001.tif')
        frame_number = int(img_file.split('.')[0])
        
        # Return 1 (anomaly) if the frame number is in the ground truth list, else return 0 (normal)
        if frame_number in gt_frames:
            return 1  # Anomaly
        else:
            return 0  # Normal

    # Load train and test data
    data_train_id, labels_train_id = load_ucsd_data(data_dir, train=True)
    data_test, labels_test = load_ucsd_data(data_dir, train=False)

    # Ensure the return structure is consistent with the toy dataset return
    id_to_type = {
        0: "normal",
        1: "anomaly"
    }

    return data_train_id, labels_train_id, data_test, labels_test, id_to_type

# Function to create a meshgrid from the data
def create_meshgrid_from_data(data, n_points=100, meshgrid_offset=1):
    x_min, x_max = data[:, 0].min() - meshgrid_offset, data[:, 0].max() + meshgrid_offset
    y_min, y_max = data[:, 1].min() - meshgrid_offset, data[:, 1].max() + meshgrid_offset
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, n_points), np.linspace(y_min, y_max, n_points))
    return xx, yy

# Main directory path
data_dir = "UCSD_Anomaly_Dataset.v1p2/UCSDped1"
data_train_id, labels_train_id, data_test, labels_test, id_to_type = get_dataset(data_dir)
