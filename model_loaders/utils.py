def format_predictions(probabilities, labels):
    """ 확률과 라벨을 기반으로 정렬된 예측 결과를 반환합니다. """
    label_probabilities = {label: prob for label, prob in zip(labels, probabilities)}
    sorted_label_probabilities = dict(sorted(label_probabilities.items(), key=lambda item: item[1], reverse=True))

    return sorted_label_probabilities
