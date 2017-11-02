import postprogress
from collections import Counter



eval_count = Counter(postprogress.eval)
eval_percent = {key: 100 * eval_count[key] / sum(eval_count.values()) for key in eval_count.keys()}

for key, percent in eval_percent:
    if key == '1':
        print("The percent of A & B are equally good: %.3f" %percent)
    elif key == '0':
        print("The percent of A & B are equally bad: %.3f" %percent)
    elif key == 'A':
        print("The percent of system A is better: %.3f" %percent)
    elif key == 'B':
        print("The percent of system B is better: %.3f" % percent)

