import tensorflow as tf
import json

_dir='./log/exp.txt'

def log_parse(_dir):
    json_stats=[]
    with open(_dir) as f:
        lines = f.readlines()
        for line in lines:
            if line[0:10] != 'json_stats':
                continue
            json_stat = json.loads(line.lstrip('json_stats: '))
            json_stats.append(json_stat)
    return json_stats

def summary(json_stats):
    varDict = {}
    for k in json_stats[0]:
        if type(json_stats[0][k]) not in [float, int]:
            continue
        if k == 'iter':
            continue
        varDict[k] = tf.get_variable(k, shape=(),
            dtype=tf.float32, initializer=tf.zeros_initializer)
        tf.summary.scalar(k, varDict[k])
    merged = tf.summary.merge_all()
    return merged,varDict


def start(json_stats,merged,varDict,_tfb_dir='./tflog/'):
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        writer = tf.summary.FileWriter(_tfb_dir, sess.graph)
        for j in json_stats:
            assignOps = []
            for k in varDict:
                assignOps.append(tf.assign(varDict[k], j[k]))

            _, summary = sess.run([assignOps, merged])
            writer.add_summary(summary, j['iter'])
            print(j['iter'])
            writer.flush()

def main():
    _log_dir = ''
    _tfb_dir = ''
    json_stats = log_parse(_log_dir)
    merged,varDict = summary(json_stats)
    start(json_stats,merged,varDict,_tfb_dir)


if __name__ == '__main__':
    main()
