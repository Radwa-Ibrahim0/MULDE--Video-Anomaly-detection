# test_ped1_loader.py
from ucsd_dataset_loader import load_ucsd_ped1_data, load_ucsd_ped1_test_with_labels

data_dir = "UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train"
gt_file_path = "UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/UCSDped1.m"

# Test the training data loader
data_train, labels_train = load_ucsd_ped1_data(data_dir, is_train=True)
print("Train data shape:", data_train.shape)
print("Train labels shape:", labels_train.shape)

# Test the test data loader
data_test, labels_test = load_ucsd_ped1_test_with_labels(data_dir, gt_file_path=gt_file_path)
print("Test data shape:", data_test.shape)
print("Test labels shape:", labels_test.shape)

# Check the label distribution
print(f"Number of normal samples in test set: {sum(labels_test == 0)}")
print(f"Number of anomaly samples in test set: {sum(labels_test == 1)}")
