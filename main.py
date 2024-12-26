import pandas as pd
import numpy as np
import scipy as sp
import gc
from utils import *
from evaluation import *
from time import time
from sklearn.metrics.pairwise import cosine_similarity
import pickle


if __name__ == '__main__':
    recommender_sys = False   # Only one of them should be true.
    system_evaluation = True   # Turn this to true, after the recommender system is completed.

    if recommender_sys:
        print("\n=====================\nLoad Data...")
        # columns: user_idx, paper_idx, normal_rating
        ratings_df = load_normal_ratings('newdir/normalized_ratings_small.csv')
        number_of_users = ratings_df['user_idx'].max() + 1     # because first user ID is 0
        number_of_papers = ratings_df['paper_idx'].max() + 1   # because first paper ID is 0
        print(f"number_of_users = {number_of_users}, number_of_papers = {number_of_papers}")

        print("\n=====================\nCreate papers vectors for the system's users liked papers")
        # create_users_file_with_paperIdxShrink("newdir/paper_idx2paper_idx_shrink.csv", "newdir/users_with_paperidx.csv")

        all_users_papers_ratings_lists = create_users_papers_vectors('newdir/users_with_paperidx_shrink.csv',
                                                                     'newdir/normalized_ratings_small.csv')

        print("\n=====================\nItem-to-Item CF Reccommender System")
        k = 50  # Number of recommendations
        print(f'k = {k}')
        knn = 8  # Minimum number of common users provided ratings for a user's paper AND a paper in the dataset
        print(f'knn = {knn}')

        # If true the system creates vectors for the knn-filtered data; if false it loads the vectors pickle file
        # If you run the RS for the first time (or changed knn variable) it should be true.
        create_vectors = True

        cos_sim_threshold = 0.01
        print(f'cos_sim_threshold = {cos_sim_threshold}')

        index_start = 1   # define the system's users i.e. 1 to 10
        index_stop = 11
        # The minimum number of occurrences is equal to knn
        # Get the values from 'paper_idx' that appear more than knn times
        values = ratings_df['paper_idx'].value_counts()
        values = values[values > knn].index.tolist()
        print(f"Papers w more ratings than KNN = {len(values)}\n")
        # make a small copy of the original dataframe
        df_small = ratings_df[ratings_df.paper_idx.isin(values)].copy()
        print(df_small.head())

        unique_pids = df_small["paper_idx"].unique()
        print(f"unique_pids = {len(unique_pids)}")

        # Delete the old DataFrame to release memory
        del ratings_df
        # Perform garbage collection
        gc.collect()

        # a list of dictionaries (we need a cell for each user, but we create a bigger empty list, to start from 1)
        generic_list = [{}] * 12

        # record start time
        time_start = time()

        if create_vectors:  # create the vectors of all the papers to use them later
            print("\nCreate the vectors of all the papers in the KNN-filtered-dataset")
            pid_ratings_dict = {}    # key (paperid) : value (np.array with ratings)
            count1 = 0
            for pid in unique_pids:   # for each paperid in the small dataset
                count1 += 1
                if count1 % 100000 == 0:
                    print(count1)
                dense_pid_ratings_np_array = create_paper_ratings_np_array(pid, df_small, number_of_users)
                # convert to a sparse matrix of COO format (to save lots of memory compared to normal np arrays)
                sparse = sp.sparse.coo_array(dense_pid_ratings_np_array)
                pid_ratings_dict[pid] = sparse

            # Save the dictionary to pickle file for future use
            with open('data/paperid_ratings_knn_' + str(knn) + '_dict.pkl', 'wb') as fp:
                pickle.dump(pid_ratings_dict, fp)
        else:
            print("\nLoad the vectors of all the papers in the KNN-filtered-dataset...")
            pid_ratings_dict = pickle.load(open('newdir/paperid_ratings_knn_' + str(knn) + '_dict.pkl', 'rb'))
            print("Loaded successfully.")

        # record end time
        time_end = time()
        # calculate the duration
        time_duration = time_end - time_start
        # report the duration
        print(f"Took {time_duration} seconds\n")


        # record start time
        time_start = time()
        print("Calculating pairwise cosine similarity...")
        for i in range(index_start, index_stop):  # for each system's user
            print(f"Process Papers of User No.{i}")
            # holds all the papers' ratings of one user
            current_user_papers_ratings_lists = all_users_papers_ratings_lists[i - 1]
            for user_paper in current_user_papers_ratings_lists:  # for each paper of the current user
                # check how many users have rated the paper (rating 1-5)
                arr2_non_zero_indices = np.nonzero(user_paper)[0]
                arr2_length = arr2_non_zero_indices.size
                if arr2_length >= knn:
                    # Find the dataset's users that have rated both the current paper (pid) and this user's paper
                    # Check k nearest-neighbors limitation, using NumPy array and the intersect1d() method
                    i_count = 0
                    for pid in unique_pids:  # for each paperid in the small dataset
                        i_count += 1
                        # just print something useful
                        if i_count % 100000 == 0:
                            print(f"{i_count} / {len(unique_pids)} papers")
                        # retrieve from the dictionary the ratings of the current paper
                        pid_ratings_array_sparse = pid_ratings_dict[pid]
                        pid_ratings_array = pid_ratings_array_sparse.todense()[0]
                        # Find the indices of non-zero elements using numpy. Check how many users have rated the paper.
                        arr1_non_zero_indices = np.nonzero(pid_ratings_array)[0]
                        common_elements = np.intersect1d(arr1_non_zero_indices, arr2_non_zero_indices)
                        common_length = common_elements.size
                        cos = 0.0
                        if common_length >= knn:
                            # print(f"Found paper w common users = {common_length}")
                            user_paper_array = np.array(user_paper, dtype=int)
                            cos = calculate_cosine_sim(user_paper_array, pid_ratings_array)
                        if cos >= cos_sim_threshold:   # Don't process papers that have very low cos.sim.
                            # Load the relevant dictionary with recommendations for the current user
                            item_similarity_dict = generic_list[i]
                            # Round max to five decimal points
                            item_similarity_dict[pid] = round(cos, 4)
                            # print(f'cos sim = {round(cos, 4)}')
                            # Sort the dictionary by value in descending order (biggest similarity will be on the top)
                            sorted_list = sorted(item_similarity_dict.items(), key=lambda x: x[1], reverse=True)
                            # keep only the top k items of the sorted list
                            part = sorted_list[:k]
                            # convert to a dict
                            converted_dict = dict(part)
                            # update the list of dictionaries
                            generic_list[i] = converted_dict

        print("\n==================\nPaper Recommendation...")
        for i in range(index_start, index_stop):  # for each user
            # load the users data
            data_dict = generic_list[i]
            print(f"User No.{i} Recommendations: ")
            print(data_dict)
            keys_list = list(data_dict.keys())  # actually a list of papers with the highest cos similarity
            # Using list comprehension, we convert a list of integers to a list of strings
            output_keys_list = [str(x) for x in keys_list]
            # Specify the file path
            file_path = 'data/user' + str(i) + '_recommendations.txt'
            # Using "with open" syntax to automatically close the file
            with open(file_path, 'w') as file:
                # Join the list elements into a single string with a newline character
                data_to_write = '\n'.join(output_keys_list)
                # Write the data to the file
                file.write(data_to_write)

        print(f"\nRecommendation completed.\n")
        # record end time
        time_end = time()
        # calculate the duration
        time_duration = time_end - time_start
        # report the duration
        print(f"Took {time_duration} seconds")


    # EVALUATION OF THE SYSTEM
    if system_evaluation:
        print("EVALUATION OF THE SYSTEM")
        print("Calculating Recall...")
        K = 50
        for i in range(1,11):
            # Load predictions for the specific user
            predictions = []
            file1 = open('newdir/user' + str(i) + '_recommendations.txt', 'r')
            Lines = file1.readlines()
            # Strips the newline character
            for line in Lines:
                predictions.append(int(line))
            # Load the ground truth for the specific user
            gt = []  # gt or relevant items
            file2 = open('newdir/user' + str(i) + 'gt_edited_shrink.csv', 'r')
            Lines = file2.readlines()
            for line in Lines:
                x = line.split(",")
                gt.append(int(x[0]))

            r = calculate_recall(K, predictions, gt)
            print(f'User No.{i} Recall@{K} = {r}')

        print("\nCalculating NDCG...")
        K = 50
        for i in range(1,11):
            # Load predictions for the specific user
            predictions = []
            file1 = open('newdir/user' + str(i) + '_recommendations.txt', 'r')
            Lines = file1.readlines()
            # Strips the newline character
            for line in Lines[:K]:
                predictions.append(int(line.strip()))
            # Load the ground truth for the specific user
            gt = {}  # gt or relevant items, we make a dict to hold the ratings as the value
            file2 = open('newdir/user' + str(i) + 'gt_edited_shrink.csv', 'r')
            Lines = file2.readlines()
            for line in Lines:
                x = line.strip().split(",")
                pidx = int(x[0])
                rat = int(x[1])
                gt[pidx] = rat   # the format of the dict: paper_idx_shrink -> rating
            the_ratings = []
            ndcg = 0.0
            non_zero_element = False
            for pr in predictions:
                if pr in gt.keys():
                    the_ratings.append(gt[pr])   # the rating
                    non_zero_element = True
                else:
                    the_ratings.append(0)
            if non_zero_element:
                ndcg = calculate_ndcg(the_ratings)
            print(f'User No.{i} NDCG@{K} = {ndcg}')

