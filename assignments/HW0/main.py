# import libraries
import os
import shutil
import numpy as np

#%% Question 1
numbers = np.arange(1, 101).tolist()
for i, number in enumerate(numbers):
    if number % 3 == 0 and number % 5 == 0:
        numbers[i] = 'ab'
    elif number % 3 == 0:
        numbers[i] = 'a'
    elif number % 5 == 0:
        numbers[i] = 'b'

print(numbers)

#%% Question 2
# Create a 3x3 matrix with random integer values ranging from 1 to 10
a = np.random.randint(1, 11, size=(3, 3))
print('Matrix a:')
print(a)
print('Sum of each row: ')
print(np.sum(a, axis=1).reshape(-1, 1))

# Find and print the maximum and minimum values in the matrix
max_value = np.max(a)
min_value = np.min(a)

# Find their positions
max_position = np.unravel_index(np.argmax(a), a.shape)
min_position = np.unravel_index(np.argmin(a), a.shape)

print("\nMaximum value:")
print(f"Value: {max_value}, Position: {max_position}")

print("\nMinimum value:")
print(f"Value: {min_value}, Position: {min_position}")


#%% Question 3
# check whether a given number N belongs to fibbonacci sequence
def is_fibonacci(N):
    # check whether the number is a perfect square
    def is_perfect_square(x):
        return int(x ** 0.5) ** 2 == x

    # check whether the number is a fibonacci number
    return is_perfect_square(5 * N ** 2 + 4) or is_perfect_square(5 * N ** 2 - 4)


print(is_fibonacci(5))
print(is_fibonacci(7))


#%% Question 4
def copy_files(source_folder, destination_folder, filename_string):
    try:
        # Ensure source and destination folders exist
        if not os.path.isdir(source_folder):
            raise FileNotFoundError(f"Source folder '{source_folder}' does not exist.")
        if not os.path.isdir(destination_folder):
            raise FileNotFoundError(f"Destination folder '{destination_folder}' does not exist.")

        # List all files in the source folder
        files = os.listdir(source_folder)
        print(files)
        # Filter files that start with the filename_string
        files_to_copy = [f for f in files if f.startswith(filename_string)]

        if not files_to_copy:
            print(f"No files starting with '{filename_string}' found in the source folder.")
            return

        # Copy each file to the destination folder
        for file_name in files_to_copy:
            src_file = os.path.join(source_folder, file_name)
            dst_file = os.path.join(destination_folder, file_name)
            shutil.copy(src_file, dst_file)
            print(f"Copied '{file_name}' to '{destination_folder}'")

    except Exception as e:
        print("Error")
        print(f"Exception: {e}")


source_dir = input("Enter the path to the source folder: ")
destination_dir = input("Enter the path to the destination folder: ")
filename = input("Enter the filename string to match: ")
copy_files(source_dir, destination_dir, filename)


#%% Acknolwedgements
# I have used the following resources to complete this assignment:
# Chat GPT-3.5, Stack Overflow, Python Documentation
