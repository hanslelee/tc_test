def read_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def calculate_f1_score(input_file, output_file):
    input_lines = read_lines(input_file)
    output_lines = read_lines(output_file)

    # 각 줄을 한 문장씩 비교하며 예측값과 실제값을 생성
    predictions = ['1' if output_line.strip().split('\t')[0] == '국방' else '0' for output_line in output_lines]
    targets = ['1' if input_line.strip().split('\t')[0] == '국방' else '0' for input_line in input_lines]

    # True Positives, False Positives, False Negatives 계산
    true_positives = sum(p == t and p == '1' for p, t in zip(predictions, targets))
    false_positives = sum(p == '1' and t == '0' for p, t in zip(predictions, targets))
    false_negatives = sum(p == '0' and t == '1' for p, t in zip(predictions, targets))
    
    # Precision, Recall 계산
    precision = true_positives / (true_positives + false_positives + 1e-9)
    recall = true_positives / (true_positives + false_negatives + 1e-9)
    
    # F1 Score 계산
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)
    
    return true_positives, false_positives, false_negatives, precision, recall, f1_score

# if __name__ == "__main__":
#     # input_file = '../output/rnn_test_output.txt'
#     input_file = '../output/cnn_test_output.txt'
#     output_file = '../data/test_answer.tsv'

#     # F1 점수 계산
#     true_positives, false_positives, false_negatives, f1_score = calculate_f1_score(input_file, output_file)
#     print("True Positives:", true_positives)
#     print("False Positives:", false_positives)
#     print("False Negatives:", false_negatives)
#     print("F1 Score:", f1_score)

# def read_lines(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         lines = file.readlines()
#     return [line.strip() for line in lines]

# def calculate_f1_score(predictions, targets):
#     true_positives = sum(p == t and p == '1' for p, t in zip(predictions, targets))
#     print("true_positives:", true_positives)
#     false_positives = sum(p == '1' and t == '0' for p, t in zip(predictions, targets))
#     print("false_positives:", false_positives)
#     false_negatives = sum(p == '0' and t == '1' for p, t in zip(predictions, targets))
#     print("false_negatives:", false_negatives)
    
#     precision = true_positives / (true_positives + false_positives + 1e-9)
#     recall = true_positives / (true_positives + false_negatives + 1e-9)
    
#     f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)
    
#     return f1_score

# def main():
#     # input_file = '../output/rnn_test_output.txt'
#     input_file = '../output/cnn_test_output.txt'
#     output_file = '../data/test_answer.tsv'

#     input_lines = read_lines(input_file)
#     output_lines = read_lines(output_file)

#     # 각 줄을 한 문장씩 비교하며 예측값과 실제값을 생성
#     predictions = ['1' if output_line.strip().split('\t')[0] == '국방' else '0' for output_line in output_lines]
#     targets = ['1' if input_line.strip().split('\t')[0] == '국방' else '0' for input_line in input_lines]

#     # F1 점수 계산
#     f1_score = calculate_f1_score(predictions, targets)
#     print("F1 Score:", f1_score)

# if __name__ == "__main__":
#     main()