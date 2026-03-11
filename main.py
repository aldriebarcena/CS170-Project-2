from math import dist

def main():
  print("Welcome to Aldrie Barcena's Feature Selection Algorithm\n")

  filename = input("Type in the name of the file to test: ")
  
  data = read_data(filename)

  print("Type the number of the algorithm you want to run:\n")
  print("\t1. Forward Selection")
  print("\t2. Backward Elimination\n")
  algorithm = input("Enter your choice: ")
  print()

  print(f"This dataset has {len(data[0]) - 1} features (not including the class attribute), with {len(data)} instances.")
  all_features = set(range(1, len(data[0])))
  print(f"Runinng nearest neighbors with all {len(data[0]) - 1} features, using \"leaving-one-out\" evaluation, I get an accuracy of {leave_one_out_cross_validation(data, all_features, 0):.2f}%\n")

  print("Beginning search.\n")
  if algorithm == "1":
    forward_search(data)
  elif algorithm == "2":
    backward_elimination(data)
  else:
    print("Invalid input")
    return

def forward_search(data):

  current_set_of_features = set()

  best_overall_features = set()
  best_overall_accuracy = 0

  # num of columns - 1 since the first column is the class label
  num_features = len(data[0]) - 1

  # loop through each feature level
  # loop excludes first column since it is the class label
  for i in range(1, num_features + 1):
    feature_to_add_at_this_level = 0
    best_so_far_accuracy = 0

    for k in range(1, num_features + 1):
      # only consider adding features that are not already in the current set
      if k in current_set_of_features:
        continue
      
      # calculate accuracy when adding feature k
      accuracy = leave_one_out_cross_validation(data, current_set_of_features, k)

      # create temp set of features for print
      temp = set(current_set_of_features)
      temp.add(k)
      print(f"\tUsing feature(s) {temp} accuracy is {accuracy:.2f}%")

      # update best accuracy and feature to add at this level if needed
      if accuracy > best_so_far_accuracy:
        best_so_far_accuracy = accuracy
        feature_to_add_at_this_level = k
    
    # add the best feature at this level to the current set
    if feature_to_add_at_this_level > 0:
      current_set_of_features.add(feature_to_add_at_this_level)

    print(f"\nFeature set {current_set_of_features} was best, accuracy is {best_so_far_accuracy:.2f}%\n")

    # update best overall subset
    if best_so_far_accuracy > best_overall_accuracy:
      best_overall_accuracy = best_so_far_accuracy
      best_overall_features = set(current_set_of_features)

  print(f"Finished search!! The best feature subset is {best_overall_features}, which has an accuracy of {best_overall_accuracy:.2f}%")
  

def backward_elimination(data):

  # start with all features
  num_features = len(data[0]) - 1
  current_set_of_features = set(range(1, num_features + 1))

  best_overall_features = set(current_set_of_features)
  best_overall_accuracy = leave_one_out_cross_validation(data, current_set_of_features, 0)

  for i in range(num_features, 0, -1):

    feature_to_remove_at_this_level = 0
    best_so_far_accuracy = 0

    for k in current_set_of_features:
      
      # create a copy of current set and remove k to calculate accuracy without it
      temp = current_set_of_features.copy()
      temp.remove(k)

      accuracy = leave_one_out_cross_validation(data, temp, 0)

      print(f"Using feature(s) {temp} accuracy is {accuracy:.2f}%")

      if accuracy > best_so_far_accuracy:
        best_so_far_accuracy = accuracy
        feature_to_remove_at_this_level = k

    if feature_to_remove_at_this_level > 0:
      current_set_of_features.remove(feature_to_remove_at_this_level)

    print(f"Feature set {current_set_of_features} was best, accuracy is {best_so_far_accuracy:.2f}%\n")

    if best_so_far_accuracy > best_overall_accuracy:
      best_overall_accuracy = best_so_far_accuracy
      best_overall_features = set(current_set_of_features)

  print(f"Finished search!! The best feature subset is {best_overall_features}, which has an accuracy of {best_overall_accuracy:.2f}%")

def leave_one_out_cross_validation(data, current_set_of_features, feature_to_add):

  num_correctly_classified = 0

  # create set of features, including feature to add if forward selection
  features_to_check = set(current_set_of_features)
  if feature_to_add > 0:
    features_to_check.add(feature_to_add)

  # for each instance in the data, test against rest of data
  for i in range(len(data)):
    
    # store all features of current instance
    object_to_classify = data[i][1:]
    # store class label of current instance
    label_object_to_classify = int(data[i][0])
    
    # initialize nearest neighbor variables
    nearest_neighbor_distance = float("inf")
    nearest_neighbor_location = -1

    # compare current instance to all other instances in the data
    for k in range(len(data)):

      # only check against other instances, not itself
      if k != i:

        # create array for only features to consider
        # need two arrays of same size to use dist function
        features_1 = []
        features_2 = []

        # loop through all features and add to arrays if in features to check, otherwise add 0
        for j in range(len(object_to_classify)):
          if (j+1) in features_to_check:
            features_1.append(object_to_classify[j])
            features_2.append(data[k][j+1])
          else:
            features_1.append(0)
            features_2.append(0)

        # formula: sqrt(sum(object_to_classify[j] - data[k][j+1])^2)
        # dist calculates the euclidean distance between the two arrays
        distance = dist(features_1, features_2)

        # update nearest neighbor variables if needed
        if distance < nearest_neighbor_distance:
          nearest_neighbor_distance = distance
          nearest_neighbor_location = k
          nearest_neighbor_label = int(data[nearest_neighbor_location][0])
    
    # check if nearest neighbor label matches label of object to classify
    if nearest_neighbor_label == label_object_to_classify:
      num_correctly_classified += 1
  
  # return accuracy as percentage
  return (num_correctly_classified / len(data)) * 100

def read_data(filename):
  data = []

  # open the file and read each line
  with open(filename, "r") as file:
    for line in file:

      # removes whitespace characters
      line = line.strip()

      # skips empty lines
      if line == "":
        continue

      # convert numbers to floats
      numbers = []

      for value in line.split():
        numbers.append(float(value))

      data.append(numbers)

  return data


main()