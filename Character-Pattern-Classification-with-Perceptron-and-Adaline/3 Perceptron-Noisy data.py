import numpy as np
import time

characters = ['A', 'B', 'C', 'D', 'E', 'J', 'K']

# Function to read character files
def read_character_file(path):
    with open(path, "r") as data:
        vector = []
        for line in data:
            # Convert characters into numerical values
            vector.extend([-1 if char == '.' else 1 if char == '#' else 0 for char in line.strip()])
        vector.append(1)  # Bias
        return vector

# Define activation function
def activation_function(threshold, y_in):
    return 1 if y_in > threshold else -1 if y_in < -threshold else 0

def stop_point_check(output, target):
    return np.all(output == target)

# Training function using Perceptron Rule
def Train(character, w0, b, threshold, learning_rate):
    
    path = []
    target = np.full(21, -1)
    index = characters.index(character)

    for i in range(3):
        target[index * 3 + i] = 1

    for i in range(21):
        char_index = i // 3
        num = i % 3 + 1
        char_path = characters[char_index]
        address = f"C:\\Users\\Parmida\\Desktop\\Characters-TrainSet\\{char_path}{num}.txt"
        path.append(address)

    input = np.array([read_character_file(path[i]) for i in range(len(path))])

    w = np.full(64, w0)
    w[-1] = b

    output_vector = np.full(len(path), 2)

    stop_point = False
    updating_num = 0
    target_check = 0

    # Training loop
    while not stop_point and updating_num < 1000000:
        for i in range(len(target)):
            if output_vector[i] != target[i]:
                y_in = np.dot(input[i], w)
                output = activation_function(threshold, y_in)
                output_vector[i] = output

            if output_vector[i] != target[i]:
                # Compute the weight change using Perceptron rule
                w_change = learning_rate * np.dot(input[i], target[i])
                w_change = w_change.astype(w.dtype)  # Cast w_change to the same data type as w
                w += w_change
                updating_num += 1

        target_check += 1

        stop_point = stop_point_check(output_vector, target)

    err_rate1 = updating_num / target_check
    print(f"\nNumber of updating weights for training character {character}: {updating_num}")
    print(f"Error Rate for Training: {err_rate1}")
    return w, err_rate1

def more_probability(output_vector, characters):
    character_counts = {char: 0 for char in characters}
    
    for i, character in enumerate(characters):
        character_counts[character] = np.count_nonzero(output_vector[i * 3: (i + 1) * 3] == 1)
    
    max_probability = max(character_counts.values())
    result = [char for char, count in character_counts.items() if count == max_probability]
    
    return result


# Testing function for given characters
def Test(character, threshold, w):
    path = []
    target = np.full(21, -1)
    index = characters.index(character)

    for i in range(3):
        target[index * 3 + i] = 1

    for i in range(21):
        char_index = i // 3
        num = i % 3 + 1
        char_path = characters[char_index]
        address = f"C:\\Users\\Parmida\\Desktop\\Characters-TestSet\\{char_path}{num}.txt"
        path.append(address)

    input = np.array([read_character_file(path[i]) for i in range(len(path))])
    y_in = np.dot(input, w)
    output_vector = np.array([activation_function(threshold, y_in[i]) for i in range(len(y_in))])

    # Simple voting between possible characters
    possible_characters = more_probability(output_vector, characters)

    FN = 0
    FP = 0

    # Calculating false negatives (FN) and false positives (FP)
    for i in range(len(target)):
        if target[i] != output_vector[i]:
            if target[i] == 1:
                FN += 1
            else:
                FP += 1

    print(f"number of mistakes = {FN + FP}")
    err_rate = (FN + FP) / 21 * 100
    print(f"Error Rate for testing character {character} is {err_rate}\n")
    
    if len(possible_characters) == 1:
        final_character = possible_characters[0]
        print(f"Final result between pridicted characters: {final_character}\n-------------------------------")
    else:
        print("Multiple characters match the pattern.")

    return err_rate

# Main function to perform training and testing
def run(w0, b, threshold, learning_rate):
    train_err_rate = 0
    test_err_rate = 0
    err_num = 0
    start_time = time.time() # Record start time


    # Loop through characters for training and testing
    for character in characters:
        w, err_rate1 = Train(character, w0, b, threshold, learning_rate)
        err_rate2 = Test(character, threshold, w)
        train_err_rate += err_rate1
        test_err_rate += err_rate2

        FN = err_rate2 * 0.01 * 21
        FP = err_rate2 * 0.01 * 21 
        err_num +=  (FN + FP ) / 2

    end_time = time.time()  # Record end time
    run_time = end_time - start_time  # Calculate runtime

    # Calculate and print average error rates for training and testing, runtime
    train_err_rate /= len(characters)
    test_err_rate /= len(characters)
    print(f"\nFor \'initial weight={w0}\', \'bias={b}\', \'threshold={threshold}\', \'learning_rate={learning_rate}\':")
    print(f"Train error rate is {train_err_rate}")
    print(f"Test error rate is {test_err_rate}\n")
    print(f"Total number of mistakes: {err_num}\n")
    print(f"Runtime: {run_time}\n")


'''=================================================================='''
# Run main function with different parameters

# run(0.1, 0.1, 0, 0.1)
# run(0.5, -0.2, 0, 0.1)
# run(-2, 2, 0, 0.1)
run(0.3, 0.9, 0, 0.1)     # =======> best values
