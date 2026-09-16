import numpy as np

def read_character_file(address):
    data = open(address, "r")
    vector = []
    for line in data.readlines():
        stripped_line = line.strip('\n')
        for i in range(0, len(stripped_line)):
            if '#' in stripped_line[i]:
                vector.append(1)
            elif '.' in stripped_line[i]:
                vector.append(0)
            else:
                vector.append(0.5)

    final_vector = []

    # Calculate the sum of rows
    row_sums = [sum(vector[i * 7:(i + 1) * 7]) for i in range(9)]
    final_vector.extend(row_sums)

    # Calculate the sum of columns
    column_sums = [sum(vector[i + j * 7] for j in range(9)) for i in range(7)]
    final_vector.extend(column_sums)

    # Normalizing final_vector
    max_value = max(final_vector)
    final_vector = [x / max_value for x in final_vector]
  
    final_vector.append(1)  # bias
    return final_vector


def activation_function(threshold, y_in):
    return 1 if y_in > threshold else -1 if y_in < -threshold else 0

def stop_point_check(output, target):
    return np.all(output == target)

def Train(character, w0, b, threshold, learning_rate):
    path = []
    target = np.full(21, -1)
    characters = ['A', 'B', 'C', 'D', 'E', 'J', 'K']
    index = characters.index(character)

    for i in range(3):
        target[index * 3 + i] = 1

    for i in range(21):
        char_index = i // 3
        num = i % 3 + 1
        char_path = characters[char_index]
        address = f"C:\\Users\\Parmida\\Desktop\\Characters-TrainSet\\{char_path}{num}.txt"
        path.append(address)

    # Initialize the weights as floats for each feature
    input = np.array([read_character_file(path[i]) for i in range(len(path))])
    w = np.full(len(input[0]), float(w0))
    w[-1] = float(b)  # Set the last element as bias

    output_vector = np.full(len(path), 2)

    stop_point = False
    updating_num = 0
    target_check = 0

    while not stop_point and updating_num < 1000000:
        for i in range(len(target)):
            if output_vector[i] != target[i]:
                y_in = np.dot(input[i], w)
                output = activation_function(threshold, y_in)
                output_vector[i] = output

            if output_vector[i] != target[i]:
                # Compute the error
                error = target[i] - output_vector[i]

                # Compute the weight change using the Delta rule
                w_change = learning_rate * error * input[i]

                # Update the weights
                w += w_change

                updating_num += 1

        target_check += 1

        stop_point = stop_point_check(output_vector, target)

    err_rate1 = updating_num / target_check
    print(f"\nNumber of updating weights for training character {character}: {updating_num}")
    print(f"Error Rate for Training: {err_rate1}")
    return w, err_rate1

def Test(character, threshold, w):
    path = []
    target = np.full(21, -1)
    characters = ['A', 'B', 'C', 'D', 'E', 'J', 'K']
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

    FN = 0
    FP = 0

    for i in range(len(target)):
        if target[i] != output_vector[i]:
            if target[i] == 1:
                FN += 1
            else:
                FP += 1
    print(f"number of mistakes = {FN+FP}")
    err_rate = (FN + FP) / 21 * 100
    print(f"Error Rate for testing character {character} is {err_rate}\n\n-------------------------------")
    return err_rate

def run(w0, b, threshold, learning_rate):
    characters = ['A', 'B', 'C', 'D', 'E', 'J', 'K']
    train_err_rate = 0
    test_err_rate = 0

    for character in characters:
        w, err_rate1 = Train(character, w0, b, threshold, learning_rate)
        err_rate2 = Test(character, threshold, w)
        train_err_rate += err_rate1
        test_err_rate += err_rate2

    train_err_rate /= len(characters)
    test_err_rate /= len(characters)
    print(f"\nFor initial weight={w0}, bias={b}, threshold={threshold}, learning_rate={learning_rate}:")
    print(f"Train error rate is {train_err_rate}")
    print(f"Test error rate is {test_err_rate}\n")

'''=================================================================='''
# Run main function with different parameters

# run(0.1, 0.1, 0, 0.1)
# run(0.5, -0.2, 0, 0.1)
# run(-2, 2, 0, 0.1)
run(0.3, 0.9, 0, 0.1)     # =======> best values