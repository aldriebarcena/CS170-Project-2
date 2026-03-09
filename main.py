
# 1. first create a function that can walk down the tree of features
# 2. once this is carefully tested, create a loop inside that considers each feature separately

# 3. implement leave_one_out cross validation for each feature

def main():
  print("Welcome to Aldrie Barcena's Feature Selection Algorithm")
  # filename = input("Type in the name of the file to test: ")
  filename = "test.txt"
  data = read_data(filename)

  print("Type the number of the algorithm you want to run:")
  print("1. Forward Selection")
  print("2. Backward Elimination")
  algorithm = input()

  if algorithm == "1":
    forward_search(data)
  elif algorithm == "2":
    backward_elimination(data)
  else:
    print("Invalid input")
    return

def forward_search(data):

  current_set_of_features = set()

  # num of columns - 1 since the first column is the class label
  num_features = len(data[0]) - 1

  # loop through each feature level
  # loop excludes first column since it is the class label
  for i in range(1, num_features + 1):
    print("On the", i, "th level of the search tree")
    feature_to_add_at_this_level = 0
    best_so_far_accuracy = 0

    for k in range(1, num_features + 1):
      # only consider adding features that are not already in the current set
      if k in current_set_of_features:
        continue

      print("--Considering adding the", k, "th feature")
      accuracy = leave_one_out_cross_validation(data, current_set_of_features, k)

      # update best accuracy and feature to add at this level if needed
      if accuracy > best_so_far_accuracy:
        best_so_far_accuracy = accuracy
        feature_to_add_at_this_level = k
    
    if feature_to_add_at_this_level > 0:
      current_set_of_features.add(feature_to_add_at_this_level)

    print("On level", i, "I added feature", feature_to_add_at_this_level, "to current set")

def backward_elimination(data):

  current_set_of_features = set()

  # number of columns - 1 since column 0 is the class label
  num_features = len(data[0]) - 1

  # start with all features
  for i in range(1, num_features + 1):
    current_set_of_features.add(i)

  # iterate through levels of the search tree
  for i in range(num_features, 0, -1):
    print("On the", i, "th level of the search tree")

    feature_to_remove_at_this_level = 0
    best_so_far_accuracy = 0

    for k in current_set_of_features:

      print("--Considering removing the", k, "th feature")

      # temporarily remove feature k to test accuracy without it
      temp = current_set_of_features.copy()
      temp.remove(k)

      accuracy = leave_one_out_cross_validation(data, temp, 0)

      # update best accuracy and feature to remove
      if accuracy > best_so_far_accuracy:
        best_so_far_accuracy = accuracy
        feature_to_remove_at_this_level = k

    if feature_to_remove_at_this_level > 0:
      current_set_of_features.remove(feature_to_remove_at_this_level)

    print("On level", i, "I removed feature", feature_to_remove_at_this_level, "from current set")

def leave_one_out_cross_validation(data, current_set_of_features, feature_to_add):
  # for now return a random numer between 0 and 0.99 to test the algorithm
  import random
  return random.uniform(0, 0.99)


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