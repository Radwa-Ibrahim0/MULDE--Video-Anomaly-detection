import numpy as np
import os
import tifffile as tiff

# Ground truth data for test videos
TestVideoFile = [
    {"gt_frame": list(range(60, 153))},
    {"gt_frame": list(range(50, 176))},
    {"gt_frame": list(range(91, 201))},
    {"gt_frame": list(range(31, 169))},
    {"gt_frame": list(range(5, 91)) + list(range(140, 201))},
    {"gt_frame": list(range(1, 101)) + list(range(110, 201))},
    {"gt_frame": list(range(1, 176))},
    {"gt_frame": list(range(1, 95))},
    {"gt_frame": list(range(1, 49))},
    {"gt_frame": list(range(1, 141))},
    {"gt_frame": list(range(70, 166))},
    {"gt_frame": list(range(130, 201))},
    {"gt_frame": list(range(1, 157))},
    {"gt_frame": list(range(1, 201))},
    {"gt_frame": list(range(138, 201))},
    {"gt_frame": list(range(123, 201))},
    {"gt_frame": list(range(1, 48))},
    {"gt_frame": list(range(54, 121))},
    {"gt_frame": list(range(64, 139))},
    {"gt_frame": list(range(45, 176))},
    {"gt_frame": list(range(31, 201))},
    {"gt_frame": list(range(16, 108))},
    {"gt_frame": list(range(8, 166))},
    {"gt_frame": list(range(50, 172))},
    {"gt_frame": list(range(40, 136))},
    {"gt_frame": list(range(77, 145))},
    {"gt_frame": list(range(10, 123))},
    {"gt_frame": list(range(105, 201))},
    {"gt_frame": list(range(1, 16)) + list(range(45, 114))},
    {"gt_frame": list(range(175, 201))},
    {"gt_frame": list(range(1, 181))},
    {"gt_frame": list(range(1, 53)) + list(range(65, 116))},
    {"gt_frame": list(range(5, 166))},
    {"gt_frame": list(range(1, 122))},
    {"gt_frame": list(range(86, 201))},
    {"gt_frame": list(range(15, 109))}
]

# Function to get test set labels (0 for normal, 1 for anomaly)
def get_label_for_frame(folder, img_file, gt_frames):
    # Extract the frame number from the image file (assuming filenames like '001.tif')
    frame_number = int(img_file.split('.')[0])
    
    # Return 1 (anomaly) if the frame number is in the ground truth list, else return 0 (normal)
    if frame_number in gt_frames:
        return 1  # Anomaly
    else:
        return 0  # Normal

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

        # Loop through each folder (each test clip)
        for folder in sorted(os.listdir(path)):
            folder_path = os.path.join(path, folder)
            if os.path.isdir(folder_path):
                # Identify which test video folder this is (e.g., Test001 corresponds to index 0)
                if(not train):
                    video_index = int(folder.replace("Test", "")) - 1 
                gt_frames = TestVideoFile[video_index]['gt_frame'] if not train else None

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
                                labels.append(0)  # All training frames are normal
                            else:
                                labels.append(get_label_for_frame(folder, img_file, gt_frames))
                        except tiff.TiffFileError as e:
                            print(f"Error reading {img_file}: {e}")
                            continue

        return np.array(data), np.array(labels)

    # Load train and test data
    data_train_id, labels_train_id = load_ucsd_data(data_dir, train=True)
    data_test, labels_test = load_ucsd_data(data_dir, train=False)

    # Ensure the return structure is consistent with the toy dataset return
    id_to_type = {
        0: "normal",
        1: "anomaly"
    } 
    # print (sum(labels_test))


    return data_train_id, labels_train_id, data_test, labels_test, id_to_type

def create_meshgrid_from_data(data, n_points=100, meshgrid_offset=1):
    x_min, x_max = data[:, 0].min() - meshgrid_offset, data[:, 0].max() + meshgrid_offset
    y_min, y_max = data[:, 1].min() - meshgrid_offset, data[:, 1].max() + meshgrid_offset
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, n_points), np.linspace(y_min, y_max, n_points))
    return xx, yy

# Example usage
data_dir = "UCSD_Anomaly_Dataset.v1p2/UCSDped1"
data_train_id, labels_train_id, data_test, labels_test, id_to_type = get_dataset(data_dir)

