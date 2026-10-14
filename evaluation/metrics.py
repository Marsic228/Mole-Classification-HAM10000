def calculate_class_metrics(matrix, class_names):
    result = {}

    for class_index, class_name in enumerate(class_names):
        true_positive = matrix[class_index][class_index]

        false_positive = 0
        for row_index in range(len(matrix)):
            if row_index != class_index:
                false_positive += matrix[row_index][class_index]

        false_negative = 0
        for column_index in range(len(matrix[class_index])):
            if column_index != class_index:
                false_negative += matrix[class_index][column_index]

        if true_positive + false_positive == 0:
            precision = 0.0
        else:
            precision = true_positive / (true_positive + false_positive)
        if true_positive + false_negative == 0:
            recall = 0.0
        else:
            recall = true_positive / (true_positive + false_negative)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        result[class_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        }

    return result

if __name__ == "__main__":
    test_matrix = [
        [5, 2, 0],
        [1, 4, 1],
        [0, 2, 6],
    ]
    test_class_names = ["A", "B", "C"]

    metrics = calculate_class_metrics(test_matrix, test_class_names)
    print(metrics)
