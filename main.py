import numpy as np

from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score

# Load the data file and return the # of rating for each movie with a mapping between each movie and their id. 
def load_rating_data(path, n_users, n_movies):
    """
    Load rating data from a specified file. 

    This function reads a file from the given path and creates a matrix of user-movide ratings. 
    It also generates a mapping of movie IDs to their indices and a count of ratings for each movie. 
    
    Parameters: 
    path (str): The file path to the ratings data. 
    n_users (int): The number of users in the dataset.
    n_movies (int): The number of movies in the dataste.

    Returns: 
    tuple: A tuple containing three elements: 
        - data (numpy.ndarray): A 2D array of ratings with users as rows and movies as columns. 
        - movie_n_rating (defaultdict(int)): A dictionary with movie IDs as keys and the number of ratings as values. 
        - movie_id_mapping (dict): A dictionary mapping movie IDs to their corresponding index in the data array.
    """

    try: 
        data = np.zeros([n_users, n_movies], dtype=np.float32)

        movie_id_mapping = {}
        movie_n_rating = defaultdict(int)

        with open(path, 'r') as file:
            #  Skip the first element of the list of str rep lines from the file because of headers. 
            for line in file.readlines()[1:]:
                user_id, movie_id, rating, _ = line.split("::")
                
                # Adjust the IDs to be zero-based.
                user_id = int(user_id) - 1
                
                # Assigns the next avaialble index to movie_id.
                if movie_id not in movie_id_mapping:
                    movie_id_mapping[movie_id] = len(movie_id_mapping)

                rating = float(rating)

                # Assign user_i rating for the movie corresponding to the movie_id.
                data[user_id, movie_id_mapping[movie_id]] = rating

                # If the rating is greater than 0, increment the stored number of ratings for the movie corresponding to movie_id.
                if rating > 0:
                    movie_n_rating[movie_id] += 1
    except FileNotFoundError:
        print(f"Error: The file {path} was not found.")
        return None, None, None
    except Exception as e: 
        print("An error occurred: {e}")
        return None, None, None

    return data, movie_n_rating, movie_id_mapping

def prepare_data(data, movie_id_mapping, movie_id_most):
    """
    Prepare the data for training the Bayes Naive classifier.

    This function processes the rating data to create a feature matrix and a target vector for model training. 
    It removes the column corresponding to the most rated movie from the data array to form the feature matrix (X). 
    The target vector (Y) consists of all ratings for the most rated movie. 
    Rows where the most rated movie is not rated (rating = 0) are removed.
    The ratings are converted to binary values (0 or 1) based on a specified threshold.
    
    Parameters: 
    data (numpy.ndarray): A 2D array of ratings with users as rows and movies as columns. 
    movie_id_mapping (dict): A dictionary mapping movie IDs to their corresponding index in the data array.
    movie_id_most (int): The index of the movie in the data array with the most ratings. 

    Returns: 
    tuple: A tuple containing two elements: 
        - X (numpy.ndarray): A 2D array of user ratings for all movies except the most rated movie. 
        - Y (numpy.ndarray): A 1D target vector containing the binary ratings (0 or 1) for the most rated movie. 
    """

    if data is None or movie_id_mapping is None or movie_id_most is None: 
        print("Invalid input data. Cannot prepare data for training. ")
        return None, None

    # Remove the column corresponding to the most rated movie.
    X_raw = np.delete(data, movie_id_mapping[movie_id_most], axis=1)

    # Extract all of the ratings for the most rated movie by selecting the column corresponding to the most rated movie. This is the target vector.
    Y_raw = data[:, movie_id_mapping[movie_id_most]]

    # Filter X_raw to include only those rows where the corresponding entry in Y_raw is greater than 0.
    X = X_raw[Y_raw > 0]

    # Filter Y_raw to only include rows that have a rating greater than 0. 
    Y = Y_raw[Y_raw > 0]

    # Ratings equal to 3 or below are considered "dislike" (or don't recommend), 
    # and ratings above 3 are considered "like" (or recommend).
    recommend = 3

    # Change all ratings less tham 3 to 0. 
    Y[Y <= recommend] = 0

    # Change all ratings greater than 3 to 1. 
    Y[Y > recommend] = 1

    return X, Y

def get_most_rated_movie(movie_n_rating): 
    # Sort the items by the number of raitngs in descending order and selects the first, most rated movie.
    return sorted(movie_n_rating.items(), key=lambda d: d[1], reverse=1)[0]

def train_model(X_train, Y_train):
    """
    Train a Multinomial Bayes Naive classifier.

    This function trains a Multinomial Naive Bayes classifier using the provided training data.
    The model uses Laplace smoothing (alpha=1.0) to handle the problem of zero probability in categorical data. 
    It learns the likelihood of each class based on the frequencies of the features in the training data. 
    
    Parameters: 
    X_train (numpy.ndarray): A 2D array of user ratings for movies, used as features for training. 
    Y_train (numpy.ndarray): A 1D target vector containing the binary ratings (0 or 1) indicating user preferences for the most rated movie.

    Returns: 
        - clf (MultinomialNB): A trained Multinomial Naive Bayes classifier. The classifier is trained to predict the ratings for the most rated movie. 
    """ 

    if X_train is None or Y_train is None: 
        print("Invalid training data. Cannot train model.")
        return None

    clf = MultinomialNB(alpha=1.0, fit_prior=True)
    
    # Train the Naive Bayes classifier.
    clf.fit(X_train, Y_train)

    return clf

def evaluate_model(clf, X_test, Y_test):
    """
    Evaluate the trained Multinomial Bayes Naive classifier.

    This function evaluates the trained Multinomial Naive Bayes classifier using the provided test data.
    It calculates and prints the model's accuracy, precision, recall, and F1 scores (both positive and negative).
    
    Parameters:  
    clf (MultinomialNB): A trained Multinomial Naive Bayes classifier. The classifier is trained to predict the ratings for the most rated movie based on user ratings for other movies..
    X_test (numpy.ndarray): A 2D array of user ratings for movies, used as features for testing. 
    Y_test (numpy.ndarray): A 1D target vector containing the binary ratings (0 or 1) indicating user preferences for the most rated movie.
    
    Returns:
    dict: A dictionary containing the following evaluation metrics:
        - 'accuracy': The accuracy score of the model on the test data.
        - 'precision': The precision score of the model for the positive class.
        - 'recall': The recall score of the model for the positive class.
        - 'f1_positive': The F1 score of the model for the positive class.
        - 'f1_negative': The F1 score of the model for the negative class.
    """ 

    if clf is None or X_test is None or Y_test is None: 
        print("Invalid inputs for evaluation. Cannot evaluate model.")
        return None

    # The score method makes predictions for X_test and then compares these predictions against Y_test to calc the accuracy.
    accuracy = clf.score(X_test, Y_test)

    # Calculate the most likely class label for each sample. It picks the class with the highest probability for each sample based on the previously calcualated probabilities. 
    # Output: an array where each element is the predicted class label.
    prediction = clf.predict(X_test)

    # Computes the ration of true positive predictions to the toal number of positive predictions made. 
    # It measurs the accuracy of positive rpedictions. 
    precision = precision_score(Y_test, prediction, pos_label=1)

    # Computes the ratio of true positive predictions to the total number of actual positisve. 
    # It measures the model's abilitiy to detect positive instances.
    recall = recall_score(Y_test, prediction, pos_label=1)

    # A balance beyween precision and recall for the positive and negative classes. 
    f1_positive = f1_score(Y_test, prediction, pos_label=1)
    f1_negative = f1_score(Y_test, prediction, pos_label=0)

    print(f'The accuracy is: {accuracy*100:.1f}%')
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score (Positive): {f1_positive:.2f}")
    print(f"F1 Score (Negative): {f1_negative:.2f}")

    return {
        'precision': precision,
        'recall': recall, 
        'f1_positive': f1_positive,
        'f1_negative': f1_negative
    }

if __name__ == '__main__':
    path = 'ratings.dat'
    n_users = 100000
    n_movies = 100000

    # movie_n_rating: (movie_i ID, movie_i number of ratings).
    data, movie_n_rating, movie_id_mapping = load_rating_data(path, n_users, n_movies)
    if data is None or movie_n_rating is None or movie_id_mapping is None: 
        print("Failed to load data. Exiting")
        exit(1)

    movie_id_most, n_rating_most = get_most_rated_movie(movie_n_rating)
    X, Y = prepare_data(data, movie_id_mapping, movie_id_most)
    if X is None or Y is None:
        print("Failed to prepare data. Exiting.")    
        exit(1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) # Randomly select 20% of the data to form the test set.  
    clf = train_model(X_train, Y_train)
    if clf is None: 
        print("Failed to train model. Exiting.")
        exit(1)

    evaluate_model(clf, X_test, Y_test)