import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KDTree
import pickle
import random

base_path = "/home/wzj/pan2/Chilean_Underground_Mine_Dataset_Many_Times/"
runs_folder = "chilean/"
filename = "pointcloud_locations_20m_10overlap.csv"
# 使用NDT文件夹
pointcloud_fols = "/pointcloud_20m_10overlap/"

all_folders = sorted(os.listdir(os.path.join(base_path, runs_folder)))
print(f"Total folders found: {len(all_folders)}")

# 过滤出有效的session folders（假设是数字命名）
valid_folders = []
for folder in all_folders:
    if not folder.startswith('.') and folder.isdigit():
        valid_folders.append(folder)

valid_folders.sort(key=int)  # 按数字大小排序
print(f"Valid session folders: {len(valid_folders)}")
print(f"Session range: {valid_folders[0]} - {valid_folders[-1]}")

# 基于时间/session的划分策略
# 训练集：session 100-179
# 测试集：session 180-209
TRAIN_SESSION_START = 100
TRAIN_SESSION_END = 179
TEST_SESSION_START = 180
TEST_SESSION_END = 209

def check_in_test_set_by_session(session_id):
    """基于session ID判断是否属于测试集"""
    try:
        session_num = int(session_id)
        return TEST_SESSION_START <= session_num <= TEST_SESSION_END
    except ValueError:
        return False

def check_in_train_set_by_session(session_id):
    """基于session ID判断是否属于训练集"""
    try:
        session_num = int(session_id)
        return TRAIN_SESSION_START <= session_num <= TRAIN_SESSION_END
    except ValueError:
        return False

def construct_query_dict(df_centroids, filename):
    tree = KDTree(df_centroids[['northing', 'easting']])
    # index
    ind_nn = tree.query_radius(df_centroids[['northing', 'easting']], r=7)
    ind_r = tree.query_radius(df_centroids[['northing', 'easting']], r=35)
    queries = {}
    for i in range(len(ind_nn)):
        query = df_centroids.iloc[i]["file"]
        # np.setdiff1d: Return the unique values in ar1 that are not in ar2.
        positives = np.setdiff1d(ind_nn[i], [i]).tolist()
        negatives = np.setdiff1d(df_centroids.index.values.tolist(), ind_r[i]).tolist()
        random.shuffle(negatives)
        queries[i] = {"query": query, "positives": positives, "negatives": negatives}
    with open(filename, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", filename)

# 筛选出训练和测试相关的sessions
train_folders = []
test_folders = []

for folder in valid_folders:
    session_num = int(folder)
    if TRAIN_SESSION_START <= session_num <= TRAIN_SESSION_END:
        train_folders.append(folder)
    elif TEST_SESSION_START <= session_num <= TEST_SESSION_END:
        test_folders.append(folder)

print(f"\n=== Session Distribution ===")
print(f"Training sessions ({TRAIN_SESSION_START}-{TRAIN_SESSION_END}): {len(train_folders)} sessions")
print(f"Test sessions ({TEST_SESSION_START}-{TEST_SESSION_END}): {len(test_folders)} sessions")

if len(train_folders) > 0:
    print(f"Training session range: {train_folders[0]} - {train_folders[-1]}")
if len(test_folders) > 0:
    print(f"Test session range: {test_folders[0]} - {test_folders[-1]}")

# Initialize pandas DataFrame
df_train = pd.DataFrame(columns=['file', 'northing', 'easting'], dtype=object)
df_test = pd.DataFrame(columns=['file', 'northing', 'easting'], dtype=object)

processed_folders = 0
total_processed_files = 0
train_files_count = 0
test_files_count = 0

# 处理所有相关的sessions（训练+测试）
all_relevant_folders = train_folders + test_folders

for folder in all_relevant_folders:
    csv_path = os.path.join(base_path, runs_folder, folder, filename)

    if not os.path.exists(csv_path):
        print(f"Skipping {folder}: CSV file not found at {csv_path}")
        continue

    # 检查对应的NDT文件夹是否存在
    ndt_folder_path = os.path.join(base_path, runs_folder, folder, pointcloud_fols.strip('/'))
    if not os.path.exists(ndt_folder_path):
        print(f"Skipping {folder}: NDT folder {ndt_folder_path} not found")
        continue

    print(f"Processing session {folder}...")

    df_locations = pd.read_csv(csv_path, sep=',')

    # 简化版：直接使用timestamp构建NDT文件路径
    df_locations['timestamp'] = runs_folder + folder + pointcloud_fols + df_locations['timestamp'].astype(
        str) + '.bin'
    df_locations = df_locations.rename(columns={'timestamp': 'file'})

    folder_processed_files = 0
    session_num = int(folder)

    for index, row in df_locations.iterrows():
        # 验证NDT文件是否真实存在
        full_path = os.path.join(base_path, row['file'])
        if not os.path.exists(full_path):
            print(f"Warning: file does not exist: {full_path}")
            continue

        # 根据session ID分配到训练集或测试集
        if check_in_test_set_by_session(session_num):
            df_test = pd.concat([df_test, pd.DataFrame([row])], ignore_index=True)
            test_files_count += 1
        elif check_in_train_set_by_session(session_num):
            df_train = pd.concat([df_train, pd.DataFrame([row])], ignore_index=True)
            train_files_count += 1

        folder_processed_files += 1
        total_processed_files += 1

    print(f"  Processed {folder_processed_files} files from session {folder}")
    processed_folders += 1

print(f"\n=== Processing Summary ===")
print(f"Processed sessions: {processed_folders}")
print(f"Total processed files: {total_processed_files}")
print(f"Training files: {train_files_count} (from sessions {TRAIN_SESSION_START}-{TRAIN_SESSION_END})")
print(f"Test files: {test_files_count} (from sessions {TEST_SESSION_START}-{TEST_SESSION_END})")

# 验证生成的文件路径
if len(df_train) > 0:
    sample_file = df_train.iloc[0]['file']
    print(f"Sample training file path: {sample_file}")

    # 验证NDT文件可以正确加载
    try:
        sample_full_path = os.path.join(base_path, sample_file)
        ndt_data = np.fromfile(sample_full_path, dtype=np.float64)
        ndt_data = ndt_data.reshape((-1, 12))
        print(f"Sample NDT data shape: {ndt_data.shape}")
        print(f"Expected shape: (100, 12)")
        if ndt_data.shape[1] == 12:
            print("✓ NDT file format verification successful!")
        else:
            print("✗ NDT file format verification failed!")
    except Exception as e:
        print(f"✗ Error loading sample NDT file: {e}")

# 生成查询字典文件
if len(df_train) > 0:
    construct_query_dict(df_train, "training_queries_chilean_period.pickle")
    print(f"Generated training queries with {len(df_train)} samples")
else:
    print("Warning: No training data found!")

if len(df_test) > 0:
    construct_query_dict(df_test, "test_queries_chilean_period.pickle")
    print(f"Generated test queries with {len(df_test)} samples")
else:
    print("Warning: No test data found!")

print(f"\n=== Generated Files ===")
print(f"Training queries: training_queries_chilean_period.pickle")
print(f"Test queries: test_queries_chilean_period.pickle")
print(f"\n=== Final Data Split Summary ===")
print(f"Training data: {len(df_train)} point clouds from {len(train_folders)} sessions")
print(f"Test data: {len(df_test)} point clouds from {len(test_folders)} sessions")
print(f"No data leakage: Training sessions ({TRAIN_SESSION_START}-{TRAIN_SESSION_END}) and test sessions ({TEST_SESSION_START}-{TEST_SESSION_END}) are completely separate")