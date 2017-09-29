import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from scores import maximum_mean_absolute_error
import sys
sys.setrecursionlimit(5000000)


def weighted_choice(choices):
    total = sum(w for c, w in choices)
    r = np.random.uniform(0, total)
    upto = 0
    for c, w in choices:
        if upto + w >= r:
            return c
        upto += w
    assert False, "Shouldn't get here"


def f(index, current_label, labels, num_labels, dp_matrix, class_weight):
    if index >= len(labels) or current_label >= num_labels:
        return 0

    if dp_matrix[index][current_label] != -1:
        return dp_matrix[index][current_label]

    error = class_weight[labels[index]][current_label]

    if current_label + 1 == num_labels:
        dp_matrix[index][current_label] = \
            error + \
            f(index + 1, current_label, labels, num_labels, dp_matrix, class_weight)
    else:
        dp_matrix[index][current_label] = \
            min(error +
                f(index + 1, current_label, labels, num_labels, dp_matrix, class_weight),
                f(index, current_label + 1, labels, num_labels, dp_matrix, class_weight))
    return dp_matrix[index][current_label]


def _decide_thresholds(scores, labels, num_labels, class_weight):
    def traverse_matrix(dp_matrix, class_weight):
        nscores, nlabels = dp_matrix.shape
        index, current_label = 0, 0
        ret = []
        while index+1 < nscores and current_label+1 < num_labels:
            current = dp_matrix[index][current_label]
            keep = dp_matrix[index + 1][current_label]
            error = class_weight[labels[index]][current_label]
            if abs((current - error) - keep) < 1e-5:
                index += 1
            else:
                ret.append(index)
                current_label += 1
        return ret

    dp_matrix = -np.ones((len(labels), num_labels), dtype=np.float32)
    f(0, 0, labels, num_labels, dp_matrix, class_weight)
    path = traverse_matrix(dp_matrix, class_weight)

    #return scores[path]  # old behavior: return midpoints
    ths = np.asarray([(scores[p]+scores[max(p-1, 0)])/2 for p in path])
    return ths


def gen_full_thresholds(scores, N, K):
    ths = np.arange(K-1)
    real_ths = np.arange(K-1)
    while True:
        real_ths = (scores[ths]+scores[ths+1])/2
        yield real_ths, ths

        i = K-2
        while i >= 0:
            if ths[i] >= N-(K-i):
                i -= 1
            else:
                break
        if i >= 0:
            ths[i] += 1
            ths[i+1:] = np.arange(ths[i]+1, ths[i]+(K-i-1))
        else:
            break

def _decide_thresholds2(scores, y, K, class_weight):
    def cost_thresholds(scores, y, ths, class_weight):
        yp = np.sum(scores >= ths[:, np.newaxis], 0, int)
        cost = [class_weight[_y][_yp] for _y, _yp in zip(y, yp)]
        return np.sum(cost)

    min_cost = np.inf
    min_ths = None
    total = 0
    import sys
    for ths, _ in gen_full_thresholds(scores, len(scores), K):
        sys.stdout.write('\r%.6f' % (total/2573031125))
        sys.stdout.flush()
        cost = cost_thresholds(scores, y, ths, class_weight)
        if cost < min_cost:
            min_cost = cost
            min_ths = ths[:]
        total += 1
    sys.stdout.write('\r          \r')
    #print('tested %d thresholds' % total)
    return min_ths


def decide_thresholds(scores, y, k, strategy, full=False):
    if strategy == 'uniform':
        w = 1-np.eye(k)
    elif strategy == 'inverse':
        w = np.repeat(len(y) / (k*(np.bincount(y)+1)), k).reshape((k, k)) * (1-np.eye(k))
    elif strategy == 'absolute':
        w = [[np.abs(i-j) for i in range(k)] for j in range(k)]
    else:
        raise 'No such threshold strategy: %s' % strategy

    if full:
        return _decide_thresholds2(scores, y, k, w)
    return _decide_thresholds(scores, y, k, w)


if __name__ == '__main__':
    if False:
        # threshold total combinations
        for _, th in gen_full_thresholds(np.sort(np.random.random(10)), 10, 3):
            print(th)
    else:
        import time
        n = 5000
        k = 5
        noise = 0.30
        choices = [(-v, 1. / (2 ** v)) for v in range(1, k)] + \
                  [(v, 1. / (2 ** v)) for v in range(1, k)]

        scores = np.sort(np.random.random(n))
        thresholds = scores[np.sort(np.random.choice(np.arange(n), k - 1))]
        gt = [k - 1 - sum(s < thresholds) for s in scores]
        gt = [min(k - 1, max(0, s + (weighted_choice(choices)
              if np.random.random() < noise else 0))) for s in gt]
        print('thresholds:', thresholds)
    
        strategies = ['uniform', 'inverse', 'absolute']
        # n_samples / (n_classes * np.bincount(y))
        for strategy in strategies:
            print()
            print('strategy: %s' % strategy)
            for full in (False,):
                print('full' if full else 'kelwin')
                tic = time.time()
                learned_thresholds = decide_thresholds(
                    scores, gt, k, strategy, full)
                toc = time.time()
                print('\tlearned_threshold:', learned_thresholds)
                yp = [k - 1 - sum(s < learned_thresholds) for s in scores]
                print('\ttime:', toc-tic)
                print('\taccuracy:', accuracy_score(gt, yp))
                print('\tf1 score:', f1_score(gt, yp, average='weighted'))
                print('\tmae:', mean_absolute_error(gt, yp))
                print('\tmmae:', maximum_mean_absolute_error(gt, yp))
