import numpy as np
import os
import tifffile as tiff
import re  # For parsing the .m file

def parse_ground_truth_m_file(m_file_path):
    """
    Parses a .m file to extract ground truth frame ranges for test videos.

    Parameters:
        m_file_path (str): Path to the .m file containing TestVideoFile definitions.

    Returns:
        list of dict: A list where each dictionary contains the 'gt_frame' key with a list of frame numbers.
    """
    TestVideoFile = []

    # Regular expression pattern to match lines defining gt_frame
    pattern = r'TestVideoFile\{end\+1\}\.gt_frame\s*=\s*\[(\d+):(\d+)\];'

    try:
        with open(m_file_path, 'r') as file:
            for line in file:
                match = re.search(pattern, line)
                if match:
                    start, end = map(int, match.groups())
                    gt_frames = list(range(start, end + 1))  # Inclusive range
                    TestVideoFile.append({'gt_frame': gt_frames})
    except FileNotFoundError:
        print(f"Error: The file {m_file_path} was not found.")
    except Exception as e:
        print(f"An error occurred while parsing the .m file: {e}")

    return TestVideoFile

def get_label_for_frame(img_file, gt_frames):
    """
    Assigns a label to a frame based on ground truth.

    Parameters:
        img_file (str): The image file name (e.g., '001.tif').
        gt_frames (list of int): List of frame numbers labeled as anomalies.

    Returns:
        int: 1 if the frame is an anomaly, 0 otherwise.
    """
    frame_number = int(os.path.splitext(img_file)[0])  # Extract frame number from filename

    return 1 if frame_number in gt_frames else 0

def get_dataset(data_dir, m_file_path, seed=None):
    """
    Loads the UCSD dataset with dynamic ground truth extraction from a .m file.

    Parameters:
        data_dir (str): Directory where the UCSD dataset is located.
        m_file_path (str): Path to the .m file containing ground truth definitions.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        tuple: Contains training data, training labels, test data, test labels, and label types.
    """
    def load_ucsd_data(path, TestVideoFile, train=True):
        data = []
        labels = []

        # Each folder represents each video
        for folder in sorted(os.listdir(path)):
            folder_path = os.path.join(path, folder)
            if os.path.isdir(folder_path):
                if not train:
                    # Extract video index from folder name, e.g., 'Test1' -> 0
                    video_match = re.search(r'\d+', folder)
                    if video_match:
                        video_index = int(video_match.group()) - 1
                        # Ensure video_index is within bounds
                        if video_index < len(TestVideoFile):
                            gt_frames = TestVideoFile[video_index]['gt_frame']
                        else:
                            print(f"Warning: Video index {video_index} out of range for folder '{folder}'.")
                            gt_frames = []
                    else:
                        print(f"Warning: Could not extract video index from folder '{folder}'.")
                        gt_frames = []
                else:
                    gt_frames = None

                for img_file in sorted(os.listdir(folder_path)):
                    if img_file.endswith('.tif'):
                        img_path = os.path.join(folder_path, img_file)
                        try:
                            img = tiff.imread(img_path)
                            img = img / 255.0  # Normalize pixel values
                            img_flattened = img.flatten()
                            data.append(img_flattened)

                            # Assign labels
                            if train:
                                labels.append(0)  # All training frames are normal
                            else:
                                label = get_label_for_frame(img_file, gt_frames)
                                labels.append(label)
                        except tiff.TiffFileError as e:
                            print(f"Error reading {img_file}: {e}")
                            continue

        return np.array(data), np.array(labels)

    # Parse the .m file to get TestVideoFile
    TestVideoFile = parse_ground_truth_m_file(m_file_path)

    # Define paths for training and testing data
    train_path = os.path.join(data_dir, 'Train')
    test_path = os.path.join(data_dir, 'Test')

    # Load train and test data
    data_train_id, labels_train_id = load_ucsd_data(train_path, TestVideoFile, train=True)
    data_test, labels_test = load_ucsd_data(test_path, TestVideoFile, train=False)

    id_to_type = {
        0: "normal",
        1: "anomaly"
    }

    return data_train_id, labels_train_id, data_test, labels_test, id_to_type

def create_meshgrid_from_data(data, n_points=100, meshgrid_offset=1):
    """
    Creates a meshgrid from the given data.

    Parameters:
        data (np.ndarray): The dataset.
        n_points (int): Number of points in each dimension.
        meshgrid_offset (int): Offset added to the min and max of each dimension.

    Returns:
        tuple: Two 2D arrays representing the meshgrid.
    """
    x_min, x_max = data[:, 0].min() - meshgrid_offset, data[:, 0].max() + meshgrid_offset
    y_min, y_max = data[:, 1].min() - meshgrid_offset, data[:, 1].max() + meshgrid_offset
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, n_points), np.linspace(y_min, y_max, n_points))
    return xx, yy


if __name__ == "__main__":
    # Define the path to your UCSD dataset and the .m file
    data_dir = "UCSD_Anomaly_Dataset.v1p2/UCSDped2"
    m_file_path = os.path.join(data_dir, 'Test/UCSDped2.m')  # Replace with the actual .m file name

    # Load the dataset
    data_train_id, labels_train_id, data_test, labels_test, id_to_type = get_dataset(data_dir, m_file_path)


