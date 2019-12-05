from collections import Counter
from scipy import interpolate
from scipy.interpolate import spline
from sklearn.model_selection import KFold
import math
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tf_utils import load_dataset, random_mini_batches, \
    convert_to_one_hot, predict

from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import SMOTE

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import normalize
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt

from tf_utils import convert_to_one_hot


def over_sample(X, y_orig, ratio=0.25, n_class=0):
    # mu = np.hstack((np.mean(X, axis=0), 0))
    # var = var.reshape(1, -1)
    counter = Counter(y_orig.reshape(-1))
    if(n_class == 0):
        n_class = len(counter)
    mu = np.zeros((n_class, X.shape[1]))
    var = np.zeros((n_class, X.shape[1]))
    for c in range(n_class):
        idx = np.argwhere(y_orig.astype(np.int) == c+1)
        if(idx.shape[0] == 0):
            continue
        x = X[(np.argwhere(y_orig.astype(np.int) == c+1)[0])]
        var[c, :] = np.var(x, axis=0)
        mu[c, :] = np.mean(x, axis=0)

    dataset = np.hstack((X, y_orig))
    idx = np.random.randint(1, dataset.shape[0], size=(
        round(ratio*dataset.shape[0])))
    dataset = dataset[idx]
    for data in dataset:
        # data[:-1] = mu[data[-1].astype(np.int)-1, :] + \
        #     np.random.randn(1, data.shape[0]-1) * \
        #     var[data[-1].astype(np.int)-1, :]
        data[:-1] = data[:-1] + \
            np.random.randn(1, data.shape[0]-1) * \
            var[data[-1].astype(np.int)-1, :]
    idx = np.random.permutation([i for i in range(dataset.shape[0])])
    dataset = dataset[idx]
    X_over = dataset[:, :-1]
    X = np.vstack((X, X_over))
    y_over = dataset[:, -1].reshape(-1, y_orig.shape[1])
    y_orig = np.vstack((y_orig, y_over))
    idx = np.random.permutation([i for i in range(X_over.shape[0])])
    X_over = X_over[idx]
    y_over = y_over[idx].astype(np.int)
    idx = np.random.permutation([i for i in range(X.shape[0])])
    X = X[idx]
    y_orig = y_orig[idx]
    y_orig = y_orig.astype(np.int)

    return X, y_orig, X_over, y_over


def cost(logits, labels):
    z = tf.placeholder(tf.float32, name="z")
    y = tf.placeholder(tf.float32, name="y")

    cost = tf.nn.softmax_cross_entropy_with_logits_v2(logits=z, labels=y)
    sess = tf.Session()
    cost = sess.run(cost, feed_dict={z: logits, y: labels})
    sess.close()
    return cost


def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, [n_x, None], name='X')
    Y = tf.placeholder(tf.float32, [n_y, None], name='Y')
    return X, Y


def get_node_num(n_i, n_o, n_sample):
    alpha = 4
    # return n_i
    return round(math.log(n_i+n_o, 2))+alpha
    # return round(math.sqrt(n_i*n_o))
    # return 10


def initialize_parameters(n_features, n_class, m, layer_num=3):
    tf.set_random_seed(1)

    parameters = dict()
    n_in = n_features
    n_out = get_node_num(n_in, n_class, m)

    for i in range(layer_num):
        if (i == layer_num - 1):
            n_out = n_class
        W = tf.get_variable(
            "W"+str(i), [n_out, n_in],
            initializer=tf.contrib.layers.xavier_initializer(seed=1))
        b = tf.get_variable(
            "b"+str(i), [n_out, 1], initializer=tf.zeros_initializer())
        n_in = n_out
        n_out = round(n_in*2/3+0.5)
        parameters.update({'W'+str(i): W})
        parameters.update({'b'+str(i): b})

    return parameters


def forward_propagation(X, parameters, layer_num=3):

    A = X
    for i in range(layer_num):
        Z = tf.add(
            tf.matmul(parameters['W'+str(i)], A), parameters['b'+str(i)])
        A = tf.nn.relu(Z)

    return Z


def compute_cost(Z, Y, parameters, layer_num, lambda_=0.00005):
    logits = tf.transpose(Z)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=labels
    ))
    if(lambda_ != 0):
        regularizer = tf.contrib.layers.l2_regularizer(scale=lambda_)
        w_list = []
        for i in range(layer_num):
            w_list.append(parameters['W'+str(i)])
        cost += tf.contrib.layers.apply_regularization(regularizer, w_list)

    return cost


def k_fold_test(X, y, learning_rate, lambda_, layer_num, k=10, print_cost=False, print_curve=False):
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    test_cost = 0
    accu_sum = 0
    accu = 0
    accu_each = []
    cnt = 0
    for train, test in kfold.split(X, y):
        x_train = X[train]
        y_train = y[train]
        x_test = X[test]
        y_test = y[test]
        print("test size: %d" % (x_test.shape[0]))
        accu_, test_cost_, _ = model(x_train.T, y_train.T, x_test.T,
                                     y_test.T,
                                     learning_rate=learning_rate,
                                     lambda_=lambda_,
                                     print_cost=print_cost,
                                     print_curve=print_curve,
                                     num_epochs=0,
                                     threadshold=0.0001,
                                     layer_num=layer_num)
        accu_sum += accu_
        accu_each.append(accu_)
        test_cost += test_cost_
        cnt += 1
        print(cnt)
        print("current accu_avg: %f" % (accu_sum/cnt))
    accu = accu_sum/k
    test_cost /= k
    print("accu_avg: %f  test_cost_avg: %f" % (accu, test_cost))
    print("learning_rate: %f  lambda_: %f  layer_num: %f" %
          (learning_rate, lambda_, layer_num))
    print("accu: %f  test_cost: %f var: %f" %
          (accu, test_cost, np.var(accu_each)))


def k_fold_crossvalidation(X, y, k=10, print_cost=False,
                           print_curve=False):
    """
        x in shape [n_samples, n_features]
        y in shape [n_samples, ]
    """
    parameters = {
        "learning_rate": [1e-1, 1e-2, 1e-3, 1e-4],
        "lambda_": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 5e-1],
        "layer_num": [2, 3, 4, 5, 6]}
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)

    max_accu_learning_rate = -1
    max_accu_lambda_ = -1
    max_accu_layer_num = -1
    max_accu_cost = -1
    max_accu = -1
    max_accu_var = -1
    for learning_rate in parameters['learning_rate']:
        for lambda_ in parameters['lambda_']:
            for layer_num in parameters['layer_num']:
                test_cost = 0
                accu_sum = 0
                accu = 0
                accu_each = []
                print("learning_rate: %f  lambda_: %f  layer_num: %f" %
                      (learning_rate, lambda_, layer_num))
                cnt = 0
                for train, test in kfold.split(X, y):
                    x_train = X[train]
                    y_train = y[train]
                    x_test = X[test]
                    y_test = y[test]
                    accu_, test_cost_, _ = model(x_train.T, y_train.T, x_test.T,
                                                 y_test.T,
                                                 learning_rate=learning_rate,
                                                 lambda_=lambda_,
                                                 print_cost=print_cost,
                                                 print_curve=print_curve,
                                                 num_epochs=7400,
                                                 layer_num=layer_num)
                    accu_sum += accu_
                    accu_each.append(accu_)
                    test_cost += test_cost_
                    cnt += 1
                    print(cnt)
                accu = accu_sum/k
                test_cost /= k
                print("learning_rate: %f  lambda_: %f  layer_num: %f" %
                      (learning_rate, lambda_, layer_num))
                print("accu_avg: %f  test_cost_avg: %f" % (accu, test_cost))
                if (max_accu < accu):
                    max_accu = accu
                    max_accu_cost = test_cost
                    max_accu_var = np.var(accu_each)
                    max_accu_learning_rate = learning_rate
                    max_accu_lambda_ = lambda_
                    max_accu_layer_num = layer_num
                print("best parameters so far:")
                print("learning_rate: %f  lambda_: %f  layer_num: %f" %
                      (max_accu_learning_rate, max_accu_lambda_, max_accu_layer_num))
                print("accu: %f  test_cost: %f var: %f" %
                      (max_accu, max_accu_cost, max_accu_var))
    print("best parameters:")
    print("learning_rate: %f  lambda_: %f  layer_num: %f" %
          (max_accu_learning_rate, max_accu_lambda_, max_accu_layer_num))
    print("accu: %f  test_cost: %f var: %f" %
          (max_accu, max_accu_cost, max_accu_var))
    return


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001,
          lambda_=0.00005, num_epochs=0, print_cost=True, print_curve=True, layer_num=4, threadshold=0.001):
    """
    return test_accuracy, test_cost, epoch_cost
    """
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []

    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters(n_x, n_y, m, layer_num)
    Z3 = forward_propagation(X, parameters, layer_num)
    cost = compute_cost(Z3, Y, parameters, layer_num, lambda_)
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()
    print("layer_num: %d\nnode_num: %d" %
          (layer_num, get_node_num(n_x, n_y, m)))
    cnt = 0
    last_epoch = 0

    with tf.Session() as sess:
        sess.run(init)
        epoch_cost = 1e10
        last_cost = 0
        epoch = 0
        while True:
            # for epoch in range(num_epochs):
            if (num_epochs != 0):
                if (num_epochs <= epoch):
                    break
            else:
                if (num_epochs >= 30000):
                    break
            # num_minibatches = int(m/minibatch_size)
            seed = seed+1
            _, t_cost = sess.run([optimizer, cost],
                                 feed_dict={X: X_train, Y: Y_train})
            epoch_cost = t_cost
            if(epoch % 100 == 0 and np.abs((epoch_cost-last_cost)/epoch_cost) <= threadshold):
                cnt += 1
                if(print_cost):
                    print(last_cost)
                    print(epoch_cost)
                    print("div: "+str(np.abs(epoch_cost-last_cost)))
                if(cnt > 8):
                    break

            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f %f" %
                      (epoch, epoch_cost, (np.abs(epoch_cost-last_cost))))
                last_cost = epoch_cost
                # print(str(sess.run(Z3, feed_dict={X: X_train})[:, 0]))
            if (print_cost == True and epoch % 10 == 0):
                costs.append(epoch_cost)
            epoch += 1
        if(print_curve):
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("learning rate =" + str(learning_rate))
            plt.show()

        parameters = sess.run(parameters)
        print("Parameters have been trained!")
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Test Accuracy:", test_accuracy)

        test_cost = sess.run(cost, feed_dict={X: X_test, Y: Y_test})

    return test_accuracy, test_cost, epoch_cost


def find_lambda(x_train, y_train, x_test, y_test, layer_num, learning_rate, num_epochs=0):
    test_costs = []
    train_costs = []
    lambda_set = [i*0.00005+0.00005 for i in range(0, 20)]

    cnt = 0
    for lambda_ in lambda_set:
        _, test_cost, train_cost = model(x_train[:, :2000], y_train[:, :2000],
                                         x_test, y_test,
                                         layer_num=layer_num,
                                         learning_rate=learning_rate, num_epochs=num_epochs, lambda_=lambda_,
                                         print_cost=False, print_curve=False)
        test_costs.append(test_cost)
        train_costs.append(train_cost)
        cnt += 1
        print(cnt)

    plt.plot(lambda_set, test_costs, label='test')
    plt.plot(lambda_set, train_costs, label='train')
    plt.legend(ncol=2)
    plt.ylabel('cost')
    plt.xlabel('lambda')
    plt.title("cost-lambda")
    plt.show()

    return


def process_training_data(
        X, y_orig, do_simple_duplicate=False, do_smote=True, max_first_feature=0, do_one_hot=True, with_category=False):
    if(do_smote):
        if(with_category):
            sm = SMOTENC(categorical_features=[0], random_state=42)
        else:
            sm = SMOTE(random_state=42)
        X, y_orig = sm.fit_resample(X, y_orig.reshape(-1))
    elif (do_simple_duplicate):
        c = Counter(y_orig[:, -1])
        mc = c.most_common()[0]
        dup_num = []
        for cc in c.most_common():
            dup_num.append(mc[1]-cc[1])
        dup_x = np.zeros((0, X.shape[1]))
        dup_y = np.zeros((0, y_orig.shape[1]))
        for i in range(len(c)):
            class_ = c.most_common()[i][0]
            idx_c = np.argwhere(y_orig == class_)[:, 0].reshape(-1)
            idx_c = np.random.permutation(idx_c)
            if(idx_c.shape[0] >= dup_num[i]):
                idx_c = idx_c[:dup_num[i]]
            elif(idx_c.shape[0] < dup_num[i]):
                idx_c_ = idx_c[:(dup_num[i]-idx_c.shape[0])]
                dup_x = np.vstack((dup_x, X[idx_c_, :]))
                dup_y = np.vstack((dup_y, y_orig[idx_c_, :]))
            dup_x = np.vstack((dup_x, X[idx_c, :]))
            dup_y = np.vstack((dup_y, y_orig[idx_c, :]))
        X = np.vstack((X, dup_x))
        y_orig = np.vstack((y_orig, dup_y))
        idx = [i for i in range(X.shape[0])]
        idx = np.random.permutation(idx)
        X = X[idx]
        y_orig = y_orig[idx].astype(np.int)

    if (do_one_hot):
        if(max_first_feature == 0):
            max_first_feature = np.max(X[:, 0]).astype(np.int)
        one_hot_first_feature = np.eye(max_first_feature)[
            X[:, 0].reshape(-1).astype(np.int)-1]
        X = np.hstack((one_hot_first_feature, X[:, 1:]))
    max_minus_min = np.max(X, axis=0)-np.min(X, axis=0)
    idx = np.argwhere(max_minus_min.astype(np.int) == 0)
    if(idx.shape[0] != 0):
        max_minus_min[idx[:, 0]] = 1
    X = (X-np.mean(X, axis=0)) / max_minus_min
    # X = normalize(X, axis=0)
    # X[:, -1] = normalize(X[:, -1].reshape(-1, 1), axis=0)

    if(do_one_hot):
        new_x = np.hstack((one_hot_first_feature, X[:, 1:]))
    new_x = X

    return new_x, y_orig.reshape(-1, 1)


def expand_dataset(x_accord, y_accord, exp=7):
    _, _, X, y = over_sample(
        x_accord, y_accord, ratio=1.0, n_class=n_class)
    for i in range(round((2**exp)/2)):
        _, _, X_, y_ = over_sample(
            x_accord, y_accord, ratio=1.0, n_class=n_class)
        X = np.vstack((X, X_))
        y = np.vstack((y, y_))
    return X, y


x_raw, y_raw = np.loadtxt('4.1.csv', dtype=np.str, delimiter=',', unpack=True)
features = x_raw[0].split(';')
features = np.array(features)

n_features = features.shape[0]

n_x = [x_raw.shape[0] - 1, n_features]

n_y = y_raw.shape[0] - 1

x_orig = np.zeros(n_x)

for i in range(n_x[0]):
    x_orig[i] = np.array(x_raw[1+i].split(';'))

x_orig = x_orig.astype(np.int)

# print(x_orig)

y_orig = np.array(y_raw[1:]).reshape(n_y, 1).astype(np.int)

# print(y_orig)

print("n_x: "+str(n_x)+"\nn_y: %d" % n_y)

n_class = 0
counter = Counter(y_orig.reshape(-1))
n_class = len(counter)
print("n_class = "+str(n_class))
n_per_class = np.zeros(n_class).astype(np.int)
for i in range(n_class):
    n_per_class[i] = counter[i+1]

print(n_per_class)

y_orig = y_orig.astype(np.int)
print("y orig before split: "+str(Counter(y_orig.reshape(-1))))
print("x features num: ", end='')
print(np.count_nonzero(x_orig, axis=0))
for i in features:
    print(i)

X_smoted, y_orig_smoted = process_training_data(x_orig, y_orig)
X, y_orig = process_training_data(x_orig, y_orig, do_smote=False,
                                  )

x_train, x_test, y_train, y_test = train_test_split(
    X, y_orig, test_size=0.1, random_state=42
)

y_train = y_train.astype(np.int)

x_train, y_train = process_training_data(
    x_train, y_train, do_one_hot=False, do_smote=True)

print("y train: "+str(Counter(y_train.reshape(-1))))


exp = 5
x_test_over, y_test_over = expand_dataset(x_test, y_test, exp)
x_train_, y_train_ = expand_dataset(x_train, y_train, exp)
x_train = np.vstack((x_train, x_train_))
y_train = np.vstack((y_train, y_train_))
x_train = np.vstack((x_train, x_test_over))
y_train = np.vstack((y_train, y_test_over))
idx = [i for i in range(x_train.shape[0])]
idx = np.random.permutation(idx)
x_train = x_train[idx]
y_train = y_train[idx]

y_train = convert_to_one_hot(y_train-1, n_class)
y_test = convert_to_one_hot(y_test-1, n_class)


print("x_train_shape: "+str(x_train.shape))
print("y_train_shape: "+str(y_train.shape))
print("x_test_shape: "+str(x_test.shape))
print("y_test_shape: "+str(y_test.shape))
max_accu_echo = 0
max_accu_i = 0
t, test_cost, train_cost = model(x_train.T, y_train, x_test.T, y_test,
                                 learning_rate=0.0001, lambda_=0.014,
                                 num_epochs=0, layer_num=3,
                                 threadshold=0.0001)
print("max_accu_echo: %f\nmax_accu_i: %d" % (max_accu_echo, max_accu_i))
print("test cost: "+str(test_cost))

