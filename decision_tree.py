import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector, min_samples_leaf=None):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов, len(feature_vector) == len(target_vector)
    :param min_samples_leaf: минимальное количество объектов, которое должно остаться в каждом подмножестве после разбиения

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """

    x, y = feature_vector, target_vector
    n = x.size

    if np.all(x == x[0]):
        return None, None, None, None

    sorted_indices = np.argsort(x)
    x = x[sorted_indices]
    y = y[sorted_indices]

    left_positive = np.cumsum(y)[:-1]
    right_positive = np.sum(y) - left_positive

    left_size = np.arange(1, n)
    right_size = n - left_size

    left_proba = left_positive / left_size
    right_proba = right_positive / right_size

    h_left = 1 - left_proba**2 - (1 - left_proba)**2
    h_right = 1 - right_proba**2 - (1 - right_proba)**2

    left_ratio = left_size / n
    right_ratio = right_size / n

    thresholds = ((x[:-1] + x[1:]) / 2)
    ginis = (- left_ratio * h_left - right_ratio * h_right)

    split_mask = x[:-1] != x[1:]
    if min_samples_leaf is not None:
        mask = (
            (left_size >= min_samples_leaf) &
            (right_size >= min_samples_leaf)
        )
        split_mask = split_mask & mask

    thresholds = thresholds[split_mask]
    ginis = ginis[split_mask]

    if thresholds.size < 1:
        return None, None, None, None

    best_index = np.argmax(ginis)
    best_threshold = thresholds[best_index]
    best_gini = ginis[best_index]

    return thresholds, ginis, best_threshold, best_gini


class DecisionTree:

    # Конструктор
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {"level": 0}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    # Обучаем вершину
    def _fit_node(self, sub_X, sub_y, node):

        most_common_class = Counter(sub_y).most_common(1)[0][0]

        # Останавливаем, если все y одинаковые
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        # Останавливаем по max_depth
        if (self._max_depth is not None) and (node["level"] >= self._max_depth):
            node["type"] = "terminal"
            node["class"] = most_common_class
            return

        # Останавливаем по min_samples_leaf
        if (self._min_samples_leaf is not None) and (sub_X.shape[0] < 2 * self._min_samples_leaf):
            node["type"] = "terminal"
            node["class"] = most_common_class
            return

        # Останавливаем по min_samples_split
        if (self._min_samples_split is not None) and (sub_X.shape[0] < self._min_samples_split):
            node["type"] = "terminal"
            node["class"] = most_common_class
            return

        # Ищем предикат
        feature_best, threshold_best, gini_best, split = None, None, None, None
        # Перебираем признаки
        for feature in range(0, sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            # Если признак числовой, сохраняем его как есть
            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            # Если признак категориальный:
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])             # Считаем количества каждой категории
                clicks = Counter(sub_X[sub_y == 1, feature])    # Считаем количества каждой категории, у которой таргет = 1
                # Для каждой категории считаем долю положительного класса
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count

                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))    # Сортируем по доле положительного класса
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))          # Кодируем категории

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))              # Собираем признаковый вектор

            else:
                raise ValueError

            # Выбираем лучшее разбиение
            _, _, threshold, gini = find_best_split(feature_vector, sub_y, self._min_samples_leaf)
            
            if threshold is None:
                continue
                
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0], filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        # Если не нашли предикат, завершаем разбиение
        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = most_common_class
            return

        # Объявляем вершину
        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError

        # Разбиваем и обучаем дочерние
        current_level = node["level"]
        node["left_child"], node["right_child"] = {"level": current_level+1}, {"level": current_level+1}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"])


    def _predict_node(self, x, node):

        # Если лист, выводим класс
        if node["type"] == "terminal":
            return node["class"]

        # Если не лист получаем признак, сравниваем с порогом и отправляем в дочернюю вершину
        elif node["type"] == "nonterminal":
            feature = node["feature_split"]
            feature_type = self._feature_types[feature]

            if feature_type == "real":
                threshold = node["threshold"]
                if x[feature] < threshold:
                    return self._predict_node(x, node["left_child"])
                else:
                    return self._predict_node(x, node["right_child"])

            elif feature_type == "categorical":
                categories_split = node["categories_split"]
                if x[feature] in categories_split:
                    return self._predict_node(x, node["left_child"])
                else:
                    return self._predict_node(x, node["right_child"])

            else:
                raise ValueError

        else:
            raise ValueError


    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
