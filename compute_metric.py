from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--score-filepath', dest = 'score_filepath', default = '', type = str)
    return parser.parse_args()

def read_score_file(path):
    y_pred = []
    y_true = []
    with open(path, 'r') as f:
        for line in f:
            line = line.rstrip()
            score, label, genre = line.split(' ')
            y_pred.append(eval(score))
            y_true.append(eval(label))
    return y_pred, y_true

def compute_eer(y_true, y_pred):
    fpr, tpr, threshold = roc_curve(y_true, y_pred, pos_label = 1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    threshold = interp1d(fpr, threshold)(eer)
    return eer, threshold

if __name__ == '__main__':
    args = parse_args()
    y_pred, y_true = read_score_file(args.score_filepath)
    eer, threshold = compute_eer(y_true, y_pred)
    print("EER: {:3.3f}%".format(100 * eer))
    print("Threshold: {:.4f}".format(threshold))
