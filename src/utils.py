# src/utils.py

import random

# ئەم فەنگشنە لەیبڵەکانی کلاسەکان دەکات بە ژمارە  
def encode_label(label):
    if label == 'L':
        return 0
    elif label == 'B':
        return 1
    elif label == 'R':
        return 2

# بەم فانگشنە دوبارە ژمارە دەکات بە لەیبڵی کلاسەکە
def decode_label(value):
    return ['L', 'B', 'R'][value]

# داتاکە لۆد دەکات بۆ UCI فایل
def load_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 5:
                continue
            label = encode_label(parts[0])
            features = list(map(int, parts[1:]))  # Convert features to int
            data.append((features, label))
    return data

# داتاکە دابەش دەکات بۆ دوو بەش تاکو بە جیا ترەین و تیس بکات
def train_test_split(data, ratio=0.66):
    random.shuffle(data)
    split_index = int(len(data) * ratio)
    train_set = data[:split_index]
    test_set = data[split_index:]
    return train_set, test_set

# پێوانەکانی هەڵسەنگاندن
def precision_recall_f1(y_true, y_pred):
    classes = [0, 1, 2]
    result = {}

    for cls in classes:
        tp = sum((yt == cls and yp == cls) for yt, yp in zip(y_true, y_pred))
        fp = sum((yt != cls and yp == cls) for yt, yp in zip(y_true, y_pred))
        fn = sum((yt == cls and yp != cls) for yt, yp in zip(y_true, y_pred))

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall    = tp / (tp + fn) if tp + fn > 0 else 0
        f1        = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

        result[cls] = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4)
        }

    return result


# ئەم میسۆدە یارمەتیدەرە تاکو فیچەر و لەبڵەکانی کلاسەکە لە ناو داتاسێتەکە دەربهێنیت
def separate_features_labels(data):
    features = [x[0] for x in data]
    labels = [x[1] for x in data]
    return features, labels



