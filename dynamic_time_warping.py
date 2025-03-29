import numpy as np
from dtaidistance import dtw, dtw_ndim

# Assume we have embeddings for two sequences (query vs candidate match)
query_video_embeddings = np.random.rand(1200, 512)  # 1000 frames
candidate_video_embeddings = np.random.rand(1000, 512)  # 1200 frames (different length)

# Compute pairwise distances (Euclidean distance)
# distance_matrix = np.linalg.norm(query_video_embeddings[:, None] - candidate_video_embeddings[None, :], axis=2)

# Apply DTW to find the best alignment
dtw_distance, path = dtw_ndim.warping_paths(query_video_embeddings, candidate_video_embeddings)
# print(dtw_distance,dtw_distance.shape)
# Retrieve the optimal alignment path
alignment_path = dtw.best_path(path)
print(alignment_path)