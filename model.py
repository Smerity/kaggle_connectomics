'''Author: 		mlwave.com'''
'''Description:	Python benchmark code for Pearson Correlation with Descretization, in use for Kaggle Connectomics Contest'''
import brainparse as bp
import scipy.stats as stats
from sklearn import metrics
from datetime import datetime
import numpy as np
import sys
start = datetime.now()
last = datetime.now()

JUST_SCORE = False

predictions_loc = "kaggle_preds.csv"
flurofn, posfn = sys.argv[1:]
#flurofn, posfn, networkfn = sys.argv[1:]
if not JUST_SCORE:
  neuron_activities = bp.parse_time_series(flurofn)
  neuron_activities = bp.discretize_time_series(neuron_activities, threshold=0.12)
  #neuron_activities = bp.blur_time_series(neuron_activities)
  #neuron_activities = bp.normalize_time_series(neuron_activities)
  #neuron_activities = bp.discretize_time_series(neuron_activities, threshold=np.max(neuron_activities[0]) * 0.12)
  neuron_positions = bp.parse_neuron_positions(posfn)
  avg_spikes = []
  for k, v in neuron_activities.items():
    avg_spikes.append(np.sum(v))
  avg_spikes = np.mean(avg_spikes)
  print 'Average spikes:', avg_spikes


def create_predictions(predictions_loc, neuron_activities, neuron_positions, start=datetime.now(), last=datetime.now()):
  print "\nWriting:", predictions_loc
  cache = {}
  from collections import defaultdict
  scores = defaultdict(list)
  #
  for e, neuron_i_id in enumerate(xrange(len(neuron_positions))):
    computed = 0
    for neuron_j_id, neuron_position in neuron_positions.items():
      if (neuron_i_id, neuron_j_id) in cache:
        continue
      corr = stats.pearsonr(neuron_activities[neuron_i_id], neuron_activities[neuron_j_id])[0]
      cache[(neuron_i_id, neuron_j_id)] = cache[(neuron_j_id, neuron_i_id)] = corr
      scores[neuron_i_id].append(corr)
      scores[neuron_j_id].append(corr)
      computed += 1
    print e + 1, "/", len(neuron_positions), "\t", datetime.now() - start, "\t", datetime.now() - last
    print 'Computed {} Pearson values'.format(computed)
    last = datetime.now()
  #
  with open(predictions_loc, "wb") as outfile:
    outfile.write("NET_neuronI_neuronJ,Strength\n")
    for e, neuron_i_id in enumerate(xrange(len(neuron_positions))):
      for neuron_j_id, neuron_position in neuron_positions.items():
        if neuron_i_id == neuron_j_id:
          outfile.write("valid_" + str(neuron_i_id + 1) + "_" + str(neuron_j_id + 1) + ",0\n")
        elif (neuron_i_id, neuron_j_id) in cache:
          c = cache[(neuron_i_id, neuron_j_id)]
          s = scores[neuron_i_id]
          c = ((c - np.min(s)) / float(np.max(s)))
          outfile.write("valid_" + str(neuron_i_id + 1) + "_" + str(neuron_j_id + 1) + "," + str(c) + "\n")


def get_auc(goldfn, kaggle, n=1000):
  gold = np.zeros(n * n, dtype=np.int)
  for i, line in enumerate(open(goldfn)):
    nA, nB, strength = map(int, line.split(','))
    gold[(nA - 1) * n + (nB - 1)] = strength
  #
  pred = np.zeros(n * n, dtype=np.float)
  for i, line in enumerate(open(kaggle)):
    if i == 0:
      continue
    prefix, strength = line.split(',')
    _, nA, nB = prefix.split('_')
    nA, nB, strength = int(nA), int(nB), float(strength)
    pred[(nA - 1) * n + (nB - 1)] = strength
  #
  fpr, tpr, thresholds = metrics.roc_curve(gold, pred, pos_label=1)
  auc = metrics.auc(fpr, tpr)
  return auc

if not JUST_SCORE:
  create_predictions(predictions_loc, neuron_activities, neuron_positions, start, last)
#print 'AUC={}'.format(get_auc(networkfn, 'kaggle_preds.csv', n=100))
