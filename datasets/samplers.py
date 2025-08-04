# Author: Jacek Komorowski
# Warsaw University of Technology
# Modified for Chilean dataset geographic constraint validation

import random
import copy
import os
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader, Sampler

from datasets.oxford import OxfordDataset

VERBOSE = False


class BatchSampler(Sampler):
    # Sampler returning list of indices to form a mini-batch
    # Samples elements in groups consisting of k=2 similar elements (positives)
    # Batch has the following structure: item1_1, ..., item1_k, item2_1, ... item2_k, itemn_1, ..., itemn_k
    def __init__(self, dataset: OxfordDataset, batch_size: int, batch_size_limit: int = None,
                 batch_expansion_rate: float = None):
        if batch_expansion_rate is not None:
            assert batch_expansion_rate > 1., 'batch_expansion_rate must be greater than 1'
            assert batch_size <= batch_size_limit, 'batch_size_limit must be greater or equal to batch_size'

        self.batch_size = batch_size
        self.batch_size_limit = batch_size_limit
        self.batch_expansion_rate = batch_expansion_rate
        self.dataset = dataset
        self.k = 2  # Number of positive examples per group must be 2
        if self.batch_size < 2 * self.k:
            self.batch_size = 2 * self.k
            print('WARNING: Batch too small. Batch size increased to {}.'.format(self.batch_size))

        self.batch_idx = []     # Index of elements in each batch (re-generated every epoch)

        self.elems_ndx = {}    # Dictionary of point cloud indexes
        for ndx in self.dataset.queries:
            self.elems_ndx[ndx] = True
            
        # Initialize geographic coordinate cache for Chilean dataset
        self.coord_cache = {}
        self.is_chilean = hasattr(dataset, 'dataset_name') and getattr(dataset, 'dataset_name', '') == 'Chilean'
        
        if self.is_chilean:
            print("Initializing geographic coordinates cache for Chilean dataset...")
            self._load_geographic_coordinates()
            print(f"Loaded coordinates for {len(self.coord_cache)} samples")

    def _load_geographic_coordinates(self):
        """Load geographic coordinates for all samples and cache them"""
        base_path = "/home/wzj/pan2/Chilean_Underground_Mine_Dataset_Many_Times/"
        runs_folder = "chilean/"
        csv_filename = "pointcloud_locations_20m_10overlap.csv"
        
        # Cache for CSV data to avoid repeated loading
        csv_cache = {}
        
        for idx in self.dataset.queries:
            try:
                query_file = self.dataset.queries[idx]['query']
                coords = self._parse_coords_from_path(query_file, base_path, runs_folder, csv_filename, csv_cache)
                if coords is not None:
                    self.coord_cache[idx] = coords
                else:
                    print(f"Warning: Could not load coordinates for sample {idx}")
            except Exception as e:
                print(f"Error loading coordinates for sample {idx}: {e}")
                
    def _parse_coords_from_path(self, query_file, base_path, runs_folder, csv_filename, csv_cache):
        """Parse coordinates from query file path"""
        try:
            # Extract session and timestamp from path
            # Expected format: chilean/SESSION/pointcloud_20m_10overlap/TIMESTAMP.bin
            path_parts = query_file.split('/')
            if len(path_parts) < 3:
                return None
                
            session = path_parts[1]
            timestamp_with_ext = path_parts[-1]
            timestamp = timestamp_with_ext.replace('.bin', '')
            
            # Load CSV if not cached
            csv_key = session
            if csv_key not in csv_cache:
                csv_path = os.path.join(base_path, runs_folder, session, csv_filename)
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    csv_cache[csv_key] = df
                else:
                    return None
            
            df = csv_cache[csv_key]
            
            # Find the row with matching timestamp
            timestamp_int = int(timestamp)
            matching_rows = df[df['timestamp'] == timestamp_int]
            
            if len(matching_rows) > 0:
                row = matching_rows.iloc[0]
                return (float(row['northing']), float(row['easting']))
            else:
                return None
                
        except Exception as e:
            if VERBOSE:
                print(f"Error parsing coordinates from {query_file}: {e}")
            return None

    def calculate_distance(self, idx1, idx2):
        """Calculate geographic distance between two samples"""
        if not self.is_chilean:
            return float('inf')  # For non-Chilean datasets, assume all are valid
            
        if idx1 not in self.coord_cache or idx2 not in self.coord_cache:
            return float('inf')  # If coordinates not available, assume invalid
            
        coord1 = self.coord_cache[idx1]
        coord2 = self.coord_cache[idx2]
        
        # Calculate Euclidean distance
        distance = np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
        return distance

    def is_valid_negative(self, anchor_idx, candidate_idx, min_distance=35):
        """Check if candidate is a valid negative sample for anchor"""
        if not self.is_chilean:
            # For non-Chilean datasets, use original logic
            return candidate_idx not in self.dataset.get_positives_ndx(anchor_idx)
            
        distance = self.calculate_distance(anchor_idx, candidate_idx)
        return distance > min_distance

    def is_valid_positive(self, anchor_idx, candidate_idx, max_distance=7):
        """Check if candidate is a valid positive sample for anchor"""
        if not self.is_chilean:
            # For non-Chilean datasets, use original logic
            return candidate_idx in self.dataset.get_positives_ndx(anchor_idx)
            
        distance = self.calculate_distance(anchor_idx, candidate_idx)
        return distance < max_distance

    def __iter__(self):
        # Re-generate batches every epoch
        self.generate_batches()
        for batch in self.batch_idx:
            yield batch

    def __len(self):
        return len(self.batch_idx)

    def expand_batch(self):
        if self.batch_expansion_rate is None:
            print('WARNING: batch_expansion_rate is None')
            return

        if self.batch_size >= self.batch_size_limit:
            return

        old_batch_size = self.batch_size
        self.batch_size = int(self.batch_size * self.batch_expansion_rate)
        self.batch_size = min(self.batch_size, self.batch_size_limit)
        print('=> Batch size increased from: {} to {}'.format(old_batch_size, self.batch_size))

    def generate_batches(self):
        # Generate training/evaluation batches.
        # batch_idx holds indexes of elements in each batch as a list of lists
        self.batch_idx = []

        unused_elements_ndx = copy.deepcopy(self.elems_ndx)
        current_batch = []

        assert self.k == 2, 'sampler can sample only k=2 elements from the same class'

        batch_count = 0
        failed_attempts = 0
        max_failed_attempts = 1000

        while len(unused_elements_ndx) > 0 and failed_attempts < max_failed_attempts:
            if len(current_batch) >= self.batch_size:
                # Flush out a new batch
                if len(current_batch) >= 2*self.k:
                    # Ensure there're at least two groups of similar elements
                    assert len(current_batch) % self.k == 0, 'Incorrect batch size: {}'.format(len(current_batch))
                    self.batch_idx.append(current_batch)
                    batch_count += 1
                    
                    if VERBOSE and self.is_chilean and batch_count <= 10:
                        self._validate_batch_geography(current_batch, batch_count)
                        
                current_batch = []
                failed_attempts = 0
                
            if len(unused_elements_ndx) == 0:
                break

            # Add k=2 similar elements to the batch
            try:
                # 1. Select first anchor element A
                selected_element = random.choice(list(unused_elements_ndx))
                unused_elements_ndx.pop(selected_element)
                
                # 2. Find positive sample B for A
                positives = list(self.dataset.get_positives_ndx(selected_element))
                if len(positives) == 0:
                    failed_attempts += 1
                    continue

                unused_positives = [e for e in positives if e in unused_elements_ndx]
                
                if len(unused_positives) > 0:
                    second_positive = random.choice(unused_positives)
                    unused_elements_ndx.pop(second_positive)
                else:
                    second_positive = random.choice(positives)

                # Add A-B pair to batch
                current_batch += [selected_element, second_positive]

                # If batch needs more samples and we have elements left
                if len(current_batch) < self.batch_size and len(unused_elements_ndx) >= 2:
                    # 3. Select second anchor element C, ensure it's a valid negative for A and B
                    candidate_negatives = []
                    
                    for candidate in list(unused_elements_ndx):
                        if (self.is_valid_negative(selected_element, candidate) and 
                            self.is_valid_negative(second_positive, candidate)):
                            candidate_negatives.append(candidate)
                    
                    if len(candidate_negatives) == 0:
                        # Fallback: use any remaining elements
                        candidate_negatives = list(unused_elements_ndx)
                    
                    if len(candidate_negatives) > 0:
                        third_element = random.choice(candidate_negatives)
                        unused_elements_ndx.pop(third_element)
                        
                        # 4. Find positive sample D for C, ensure D is also negative for A and B
                        third_positives = list(self.dataset.get_positives_ndx(third_element))
                        
                        if len(third_positives) > 0:
                            valid_fourth_elements = []
                            
                            for pos in third_positives:
                                if (self.is_valid_negative(selected_element, pos) and 
                                    self.is_valid_negative(second_positive, pos)):
                                    valid_fourth_elements.append(pos)
                            
                            if len(valid_fourth_elements) == 0:
                                # Fallback: use any positive of third_element
                                valid_fourth_elements = third_positives
                            
                            unused_valid = [e for e in valid_fourth_elements if e in unused_elements_ndx]
                            if len(unused_valid) > 0:
                                fourth_element = random.choice(unused_valid)
                                unused_elements_ndx.pop(fourth_element)
                            else:
                                fourth_element = random.choice(valid_fourth_elements)
                            
                            current_batch += [third_element, fourth_element]
                        else:
                            # If third_element has no positives, just add it alone
                            current_batch += [third_element]
                            
                failed_attempts = 0
                
            except Exception as e:
                if VERBOSE:
                    print(f"Error in batch generation: {e}")
                failed_attempts += 1
                continue

        # Add remaining batch if it has enough elements
        if len(current_batch) >= 2*self.k:
            assert len(current_batch) % self.k == 0, 'Incorrect batch size: {}'.format(len(current_batch))
            self.batch_idx.append(current_batch)
            batch_count += 1
            
            if VERBOSE and self.is_chilean and batch_count <= 10:
                self._validate_batch_geography(current_batch, batch_count)

        print(f"Generated {len(self.batch_idx)} batches")
        if failed_attempts >= max_failed_attempts:
            print("Warning: Reached maximum failed attempts in batch generation")

    def _validate_batch_geography(self, batch, batch_num):
        """Validate geographic relationships in a batch for debugging"""
        print(f"\n=== Batch {batch_num} Geographic Validation ===")
        
        for i in range(0, len(batch), 2):
            if i + 1 < len(batch):
                anchor = batch[i]
                positive = batch[i + 1]
                distance = self.calculate_distance(anchor, positive)
                print(f"Positive pair ({anchor}, {positive}): {distance:.2f}m")
                
                # Check distances to other elements
                for j in range(len(batch)):
                    if j != i and j != i + 1:
                        other = batch[j]
                        dist_anchor_other = self.calculate_distance(anchor, other)
                        dist_positive_other = self.calculate_distance(positive, other)
                        print(f"  {anchor} -> {other}: {dist_anchor_other:.2f}m")
                        print(f"  {positive} -> {other}: {dist_positive_other:.2f}m")

        for batch in self.batch_idx:
            assert len(batch) % self.k == 0, 'Incorrect batch size: {}'.format(len(batch))


if __name__ == '__main__':
    dataset_path = '/media/sf_Datasets/PointNetVLAD'
    query_filename = 'test_queries_baseline.pickle'

    ds = OxfordDataset(dataset_path, query_filename)
    sampler = BatchSampler(ds, batch_size=16)
    dataloader = DataLoader(ds, batch_sampler=sampler)
    e = ds[0]
    res = next(iter(dataloader))
    print(res)
