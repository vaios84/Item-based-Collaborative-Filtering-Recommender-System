import pandas as pd
import numpy as np
from numpy.linalg import norm


def load_normal_ratings(acsv):
    # Load the dataset into a Pandas dataframe called ratings
    ratings = pd.read_csv(acsv)
    # Save an original copy of the dataframe
    ratings_original = ratings.copy(deep=True)

    # Check the head of the dataset
    # print(ratings.head())
    # Check the tail of the dataset
    # print(ratings.tail())
    # Get the shape and size of the dataset
    # print("\nNumber of rows    :", ratings.shape[0])
    # print("Number of columns :", ratings.shape[1])
    print("\nData info")
    print(ratings.info())
    # Check for any Null values in the dataset
    print("\nCheck for Null values")
    print(ratings.isnull().sum())
    return ratings


def create_users_file_with_paperIdxShrink(new_paper_idx_shrink_mapping, users_with_paperidx):
    # this method creates a new users file that has a new column 'paper_idx_shrink'
    # 'papers_idx' translation to the new mappings, in order to use it later in the Recommender and the evaluation
    df1 = pd.read_csv(new_paper_idx_shrink_mapping, delimiter=',')
    print(df1.head())
    print(df1.tail())

    df2 = pd.read_csv(users_with_paperidx, delimiter=',')
    print(df2.head())
    print(df2.tail())

    paper_idx = df2.paper_idx.values.tolist()
    full_pidx = []
    for p in paper_idx:  # p is a string holding all the paperidx for one user
        li = list(p.split(" "))  # li is a list holding the paperidx for one user's papers
        full_pidx += li
    # Converting all strings in list to integers Using eval()
    res = [eval(i) for i in full_pidx]
    a_dict = {}  # create a translation dictionary
    for p in res:
        # Get the row number of value based on column
        row_num = df1[df1['paper_idx'] == p].index
        if not row_num.empty:
            full_row = df1.loc[row_num, :].values.flatten().tolist()
            pidx_shrink = full_row[1]
            a_dict[p] = pidx_shrink
            print(str(p) + '->' + str(pidx_shrink))
        else:
            print("row empty. paper idx not found")

    # change dataframe column to a list using Series.values.tolist()
    pidx = df2.paper_idx.values.tolist()  # now paper_idx is a list of strings

    # create the new column 'paperIdsNew' to add it later to the dataframe
    paperIdxNew = []
    for x in pidx:  # x  is the paperids of a user
        temp = ''
        tempy = ''
        li = list(x.split(" "))  # x is a list holding the paperids for one user's papers
        # Converting all strings in list to integers using eval()
        res = [eval(i) for i in li]
        # we create a string holding the relevant new paperIdx from the dictionary
        for z in res:  # z is a paperIdx
            pIdx = a_dict.get(z)
            if pIdx is not None:
                temp = temp + str(pIdx) + ' '
                tempy = temp.strip()  # to remove the last blank space

        paperIdxNew.append(tempy)

    # add the new column to the 'users' dataframe
    df2['paper_idx_shrink'] = paperIdxNew
    # saving the new dataframe
    df2.to_csv('data/users_with_paperidx_shrink.csv', header=True, index=False)
    print("users_with_paperidx_shrink.csv was created successfully")


def create_users_papers_vectors(users_with_paperidx_shrink_csv, normal_ratings_csv):
    # delete (overwrite) any previously created files, just in case
    for u in range(1, 12):
        f = open('data/user' + str(u) + 'papers_ratings.txt', 'w')
        f.close()
    # load the users_with_paperidx file and for each user create the relevant papers ratings-vectors file
    # creating a data frame
    df = pd.read_csv(users_with_paperidx_shrink_csv, delimiter=',')
    # print(df.head())
    # print(df.tail())
    # Use the `shape` property
    # print(df.shape)
    # print("Number of rows :", df.shape[0])
    # print("Number of columns :", df.shape[1])
    normal_ratings_df = pd.read_csv(normal_ratings_csv, delimiter=',')
    number_of_users = normal_ratings_df['user_idx'].max() + 1

    # change dataframe column to a list using Series.values.tolist()
    # pidx_list is a list of strings (holding the pidx of user's likes)
    pidx_list = df.paper_idx_shrink.values.tolist()
    user_num = 1
    all_users_result_list = []
    for p in pidx_list:  # p actually is the user likes  (user_num = 1,2,3 ...)
        a_user_papers_ratings_list = []
        li = list(p.split(" "))  # li is a list holding the pidx for one user's papers
        print(f"Processing papers of User No.{user_num}")
        for a_pidx in li:  # a_pidx is one paper that the user liked
            temp_d = {}  # holds the users and ratings of the current paper, retrieved for the full normal_ratings
            # condition mask. we select only the rows where the paperid is the current pid we need to process
            mask = normal_ratings_df['paper_idx'] == int(a_pidx)
            # we create a new dataframe with the selected rows (columns remain the same as in the original df)
            df_new = pd.DataFrame(normal_ratings_df[mask])
            # print(df_new)
            for i in range(len(df_new)):  # iterate the new df per row
                temp_d[df_new.iloc[i, 0]] = df_new.iloc[i, 2]

            line = str(a_pidx) + ','
            # converting input dictionary items(key-value pair) to a list of tuples
            resultList = list(temp_d.items())
            # create a list and fill all cells with zero (the ratings of the paper)
            current_paper_ratings_list = [0] * number_of_users
            for x in resultList:  # x is a tuple  (user_idx, normal_rating)
                line = line + str(x[0]) + ' ' + str(x[1]) + ','
                current_paper_ratings_list[x[0]] = x[1]

            a_user_papers_ratings_list.append(current_paper_ratings_list)

            with open('data/user' + str(user_num) + 'papers_ratings.txt', 'a') as f:
                f.write(line)
                f.write('\n')

        all_users_result_list.append(a_user_papers_ratings_list)
        user_num += 1
    # all_users_result_list is a list (a cell for each user of the system, now in this case: 1,2,...,11),
    # of lists (a cell for each paper),
    # of lists (the ratings for the relevant paper given by the total number of users in the dataset)
    print("Users papers-ratings files completed.")
    return all_users_result_list


def calculate_cosine_sim(a, b):  # a, b are two np.arrays with ratings, for paper a and paper b
    np.seterr(divide='ignore', invalid='ignore')  # ignore if divisor is 0 or NaN
    # compute cosine similarity
    cosine = np.dot(a, b) / (norm(a) * norm(b))
    # print("Cosine similarity:", cosine)
    return cosine


def create_paper_ratings_np_array(paper, df, total_users):
    newlist = [0] * total_users  # create a list and fill all cells with zero (the ratings of a paper)
    # condition mask. we select only the rows where the paperid is the current pid we need to process
    mask = df['paper_idx'] == int(paper)
    # we create a new dataframe with the selected rows (columns remain the same as in the original df)
    df_new = pd.DataFrame(df[mask])
    # print(df_new)
    for i in range(len(df_new)):  # iterate the new df per row
        # insert the actual ratings in the relevant indexes of newlist
        # for example, in index 34800 of the newlist we insert the rating of the user 34800 for the current pid
        newlist[df_new.iloc[i, 0]] = df_new.iloc[i, 2]
    # converting list to numpy array
    arr = np.array(newlist)
    return arr
