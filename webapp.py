"""

    this code implements a simple web app for running active learning experiments
    algorithms, such as InfoTuple, are implemented in their respective repository on GitHub
    as https://github.com/Sensory-Information-Processing-Lab/infotuple

"""

import copy
import csv
import datetime
import json
import numpy as np
import os
import random
import time
from flask import Flask, request, url_for, flash, redirect, render_template
from markupsafe import escape
from scipy.spatial.distance import pdist
from torch.nn import TripletMarginLoss
from torch.utils.data import Dataset
# from infotuple import body_metrics
# global static variables
app = Flask(__name__)
app.secret_key = ''

# paths
path_abs = os.getcwd()
path_base = f"{path_abs}/annotations/"
base_url = "http://localhost:8000/"
path_data = f"{path_abs}/dataset/scenes.pt"
path_fixed_query_order = f"{path_abs}/dataset/query_order.npy"
# embeddings
path_embedding = f"{path_abs}/dataset/embeddings.npy"

# collect 100 InfoTuples
n_warmup = 2
n_fixed = 2
n_nn_infotuples = 2
n_halfnn_infotuples = 2
n_rnd = 2
n_al = 2
n_info = 2
# if people annotated or skipped this many, move on to next type.
n_offset = 1000

# test dataset settings
n_subset = 10

# TSTE params
no_dims = 2
max_iter = 100

# infotuple parameters
n_body = 8

# InfoTuple Algorithm params
n_samples = 10  # runs to determine mutual information
mu = 0.05
f_downsample = 0.1  # 10% of possible permutations
top_n = 100  # 10% of dataset: nearest neighbors of current embedding
n_permutations = 10  # 10 randomly selected infotuples

# ANN training
device = 'cpu'
sum_nn = 5
n_epochs = 5
loss_function = TripletMarginLoss()


#
# helper functions
#


def finetune_model(data, pool_data: Dataset):
    """
        here goes your fine-tuning code
    """
    return model, params


def get_type_of_query(path_csv):
    """
        This divides a study into phases.

        Future work may include interspersed query types or mixing to break participants heuristics.
    """
    if n_warmup > count_rows_in_csv_by_type(type_of_reply="warmup", path_csv=path_csv):
        return "warmup"
    # if no more left to process, skip this phase
    if n_fixed > count_rows_in_csv_by_type(type_of_reply="fix_1", path_csv=path_csv):
        if n_offset > count_rows_in_csv_by_type(type_of_reply="fix_1", path_csv=path_csv, count_skipped=True):
            return "fix_1"
    if n_rnd > count_rows_in_csv_by_type(type_of_reply="rnd", path_csv=path_csv):
        if n_offset > count_rows_in_csv_by_type(type_of_reply="rnd", path_csv=path_csv, count_skipped=True):
            return "rnd"
    if n_halfnn_infotuples > count_rows_in_csv_by_type(type_of_reply="halfnn", path_csv=path_csv):
        if n_offset > count_rows_in_csv_by_type(type_of_reply="halfnn", path_csv=path_csv, count_skipped=True):
            return "halfnn"
    if n_nn_infotuples > count_rows_in_csv_by_type(type_of_reply="nn", path_csv=path_csv):
        if n_offset > count_rows_in_csv_by_type(type_of_reply="nn", path_csv=path_csv, count_skipped=True):
            return "nn"
    if count_rows_in_csv_by_type(type_of_reply="fix_1", path_csv=path_csv, count_skipped=True) \
            > count_rows_in_csv_by_type(type_of_reply="fix_2", path_csv=path_csv, count_skipped=True):
        return "fix_2"
    # active sampling
    if n_al > count_rows_in_csv_by_type(type_of_reply="active", path_csv=path_csv):
        return "active"
    if n_info > count_rows_in_csv_by_type(type_of_reply="infotuple", path_csv=path_csv):
        return "infotuple"
    # default is done
    return "done"


def count_rows_in_csv_by_type(type_of_reply, path_csv, count_skipped=False):
    n = 0
    try:
        with open(path_csv, 'r') as csvfile:
            for line in csvfile:
                print()
                elements = line.split()
                if elements[0] == type_of_reply:
                    if elements[2] != '-1':
                        n = n + 1
                    if elements[2] == '-1' and count_skipped:
                        n = n + 1
    except Exception as e:
        pass
    return n


def create_user_folder_if_not_exist(username: str):
    if not os.path.exists(path_base + username):
        os.makedirs(path_base + username)


def load_last_query_id(username):
    path_last_query = path_base + username + "/last_query_id"
    if not os.path.exists(path_last_query):
        with open(path_last_query, 'w') as f:
            f.write(str(1))
        return 1
    else:
        with open(path_last_query, 'r') as f:
            query_id = int(f.read())
            return query_id


def write_query_id(username, query_id):
    path_last_query = path_base + username + "/last_query_id"
    with open(path_last_query, 'w') as f:
        f.write(query_id)


def initialize_csv(username: str):
    # write triplet accuracy
    path_timestamps = path_base + escape(username) + "/accuracy.csv"
    if not os.path.exists(path_timestamps):
        # done: create csv with header
        header = ['type_of_test', 'query_type', 'id', 'accuracy']
        with open(path_timestamps, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, delimiter=' ', fieldnames=header)
            writer.writeheader()
    # write timestamps
    path_timestamps = path_base + escape(username) + "/timestamps.csv"
    if not os.path.exists(path_timestamps):
        # done: create csv with header
        header = ['type_of_reply', 'id', 'time', 'type']
        with open(path_timestamps, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, delimiter=' ', fieldnames=header)
            writer.writeheader()
    path_timestamps = path_base + escape(username) + "/active_timestamps.csv"
    if not os.path.exists(path_timestamps):
        # done: create csv with header
        header = ['type', 'id', 'seconds']
        with open(path_timestamps, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, delimiter=' ', fieldnames=header)
            writer.writeheader()
    # write annotations with id
    path_csv = path_base + escape(username) + "/id_annotations.csv"
    if not os.path.exists(path_csv):
        # done: create csv with header
        header = ['type_of_reply', 'id', 'head', 'best', 'rest0', 'rest1', 'rest2', 'rest3', 'rest4', 'rest5',
                  'rest6']  # , 'rest7']
        with open(path_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, delimiter=' ', fieldnames=header)
            writer.writeheader()
    # write annotations
    path_csv = path_base + escape(username) + "/annotations.csv"
    if not os.path.exists(path_csv):
        # done: create csv with header
        header = ['type_of_reply', 'head', 'best', 'rest0', 'rest1', 'rest2', 'rest3', 'rest4', 'rest5',
                  'rest6']  # , 'rest7']
        with open(path_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, delimiter=' ', fieldnames=header)
            writer.writeheader()


def count_rows_in_csv(path_csv):
    n = 0
    try:
        with open(path_csv, 'r') as csvfile:
            for _ in csvfile:
                n = n + 1
    except Exception as e:
        pass
    return n


def write_time_to_csv(username, id, time, type_of_timestamp, type_of_reply):
    path_timestamps = path_base + escape(username) + "/timestamps.csv"
    with open(path_timestamps, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        writer.writerow([str(type_of_reply), str(id), str(time), str(type_of_timestamp)])


def write_al_time_to_csv(username, id, type_of_al, seconds):
    path_timestamps = path_base + escape(username) + "/active_timestamps.csv"
    with open(path_timestamps, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        writer.writerow([str(type_of_al), str(id), str(seconds)])


def write_acc_to_csv(username, query_type, id, type_of_test, acc):
    path_timestamps = path_base + escape(username) + "/accuracy.csv"
    with open(path_timestamps, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        writer.writerow([str(type_of_test), str(query_type), str(id), str(acc)])


def store_id_reply_in_csv(path_csv, id, response_to_store, type_of_reply):
    with open(path_csv, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        # 'head', 'best', 'rest0', 'rest1', 'rest2', 'rest3', 'rest4'
        row = [str(type_of_reply), str(id), response_to_store['head'], response_to_store['best']]
        for rest in response_to_store['rest']:
            row.append(rest)
        writer.writerow(row)


def store_reply_in_csv(path_csv, response_to_store, type_of_reply):
    with open(path_csv, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        # 'head', 'best', 'rest0', 'rest1', 'rest2', 'rest3', 'rest4'
        row = [str(type_of_reply), response_to_store['head'], response_to_store['best']]
        for rest in response_to_store['rest']:
            row.append(rest)
        writer.writerow(row)


def count_replies(path_csv):
    n_replies = count_rows_in_csv_by_type(type_of_reply="nn", path_csv=path_csv) \
                + count_rows_in_csv_by_type(type_of_reply="halfnn", path_csv=path_csv) \
                + count_rows_in_csv_by_type(type_of_reply="fix_1", path_csv=path_csv) \
                + count_rows_in_csv_by_type(type_of_reply="fix_2", path_csv=path_csv) \
                + count_rows_in_csv_by_type(type_of_reply="rnd", path_csv=path_csv) \
                + count_rows_in_csv_by_type(type_of_reply="infotuple", path_csv=path_csv) \
                + count_rows_in_csv_by_type(type_of_reply="active", path_csv=path_csv)
    return n_replies


#
# web app functions
#

@app.route("/", methods=['GET', 'POST'])
def hello_world():
    """
        init experiment and parse username
    """
    print(f"request method {request.method}")
    if request.method == 'POST':
        username = request.form['username']
        if not username:
            flash('Username is required')
        else:
            print(f"hello {escape(username)}")

            create_user_folder_if_not_exist(escape(username))
            query_id = load_last_query_id(username=escape(username))
            if query_id > n_warmup:
                return redirect(url_for(f'query', username=escape(username), query_id=query_id))
            else:
                return redirect(url_for(f'warmup_query', username=escape(username), query_id=query_id))
    return render_template('webapp.html', base_url=base_url)


@app.route('/warmup_query/<username>/<int:query_id>')
def warmup_query(username, query_id):
    print(f"Warmup {escape(username)} with  query #{query_id}")
    # retrieve number of answered queries:
    initialize_csv(escape(username))

    # warmup with random samples
    choices = [i for i in range(0, 1005) if i != query_id]
    head_id = query_id
    body_ids = np.random.choice(choices, size=n_body, replace=False)

    # measure time taken to reply
    type_of_reply = "warmup"
    write_time_to_csv(username=username, id=query_id, time=datetime.datetime.now(),
                      type_of_timestamp="query", type_of_reply=type_of_reply)

    # render page
    return render_template('query_warmup.html', username=escape(username),
                           query_id=query_id, base_url=base_url,
                           plots_body=[str(idx) for idx in body_ids.tolist()],
                           plot_head=str(head_id), n_replies=0,
                           query_type="warmup")


@app.route('/warmup_reply/<username>/<int:query_id>/', methods=['GET'])
@app.route('/warmup_reply/<username>/<int:query_id>/<string:jsonreply>', methods=['GET'])
def warmup_reply(username, query_id, jsonreply):
    # process json reply
    print(f"Warmup reply of {escape(username)} to query #{query_id} was {jsonreply}")
    response = json.loads(jsonreply)
    initialize_csv(escape(username))
    # store reply on disk
    type_of_reply = response["qtype"]
    # write reply to ID file
    path_csv_id = path_base + escape(username) + "/id_annotations.csv"
    store_id_reply_in_csv(path_csv=path_csv_id, id=query_id, response_to_store=response, type_of_reply=type_of_reply)
    path_csv = path_base + escape(username) + "/annotations.csv"
    store_reply_in_csv(path_csv=path_csv, response_to_store=response, type_of_reply=type_of_reply)
    next_query_id = query_id + 1
    write_query_id(username=escape(username), query_id=str(next_query_id))

    # measure time taken to reply
    write_time_to_csv(username=username,
                      id=query_id,
                      time=datetime.datetime.now(),
                      type_of_timestamp="reply",
                      type_of_reply=type_of_reply)

    # redirect to next query
    print(f"query id {query_id} and warmup {n_warmup}")
    if query_id >= n_warmup:
        return redirect(url_for(f'query', username=escape(username), query_id=next_query_id))
    else:
        return redirect(url_for(f'warmup_query', username=escape(username), query_id=next_query_id))


@app.route('/query/<username>/<int:query_id>')
def query(username, query_id):
    print(f"Query {escape(username)} with  query #{query_id}")

    # retrieve number of answered queries:
    initialize_csv(escape(username))
    path_csv = path_base + escape(username) + "/annotations.csv"
    n_replies = count_replies(path_csv=path_csv)
    query_type = get_type_of_query(path_csv=path_csv)

    # initialize experiment, with first fixed sample
    if query_id == n_warmup + 1:
        # initialize experiment, e.g., copy embeddings for participant
        pass

    # you should define a fixed order for every participant
    # fixed_query_order = np.load(path_fixed_query_order)

    # load pre-defined InfoTuple at beginning and end of annotations
    # create choices for InfoTuple, for random sampling, anything but anchor.
    choices = [i for i in range(0, 1005) if i != query_id]
    head_id = query_id
    plots_body = np.random.choice(choices, size=n_body, replace=False)
    # implement each phase's logic in the following if-else cascade:
    if query_type == "done":
        return render_template('done.html', username=escape(username), n_replies=n_replies)
    elif query_type == "active" or query_type == "infotuple":
        return redirect(url_for(f'active_query', username=escape(username), query_id=query_id))
    elif query_type == "fix_1":
        # load first fixed set of InfoTuples for intra/inter rater metrics.
        # if available, use same queries for everyone:
        # head_id, plots_body = load_fixed_infotuple(query_id=idx, path_to_load=path_fixed_infotuples)
        print(f"Fixed 1 infotuple = {head_id} {plots_body}")
    elif query_type == "fix_2":
        # the second set of queries just repeat the first ones.
        # idx = count_rows_in_csv_by_type(path_csv=path_csv, type_of_reply=query_type, count_skipped=True)
        # head_id, plots_body = load_fixed_infotuple(query_id=idx, path_to_load=path_fixed_infotuples)
        print(f"Fixed 2 infotuple = {head_id} {plots_body}")
    elif query_type == "nn":
        # the fully nearest neighbor infotuples
        # idx = count_rows_in_csv_by_type(path_csv=path_csv, type_of_reply=query_type, count_skipped=True)
        # head_id, plots_body = load_fixed_infotuple(query_id=idx, path_to_load=path_nn_infotuples)
        print(f"nn infotuple = {head_id} {plots_body}")
    elif query_type == "halfnn":
        # the half nearest neighbor infotuples
        # idx = count_rows_in_csv_by_type(path_csv=path_csv, type_of_reply=query_type, count_skipped=True)
        # head_id, plots_body = load_fixed_infotuple(query_id=idx, path_to_load=path_halfnn_infotuples)
        print(f"half nn infotuple = {head_id} {plots_body}")
    else:
        print(f"random sampling")
        # you may replace this with a pre-defined query order stored for all participants
        head_id = query_id
        np.random.seed(query_id)
        selected_tuple = np.random.choice(choices,
                                          size=n_body,
                                          replace=False)
        print(f"Remap head from {query_id} to {head_id}")
        plots_body = [str(idx) for idx in selected_tuple.tolist()]

    # measure time taken to reply
    # type of reply as parameter
    write_time_to_csv(username=username,
                      id=query_id,
                      time=datetime.datetime.now(),
                      type_of_timestamp="query",
                      type_of_reply=query_type)

    # shuffle the plots so that they don't repeat exactly, and so that rnd/nn are mixed
    random.shuffle(plots_body)
    # render page
    plots_body = [str(tmp) for tmp in plots_body]
    return render_template('query_central.html', username=escape(username),
                           query_id=query_id, base_url=base_url,
                           plots_body=plots_body, plot_head=str(head_id),
                           n_replies=n_replies, query_type=query_type)


@app.route('/reply/<username>/<int:query_id>/', methods=['GET'])
@app.route('/reply/<username>/<int:query_id>/<string:jsonreply>', methods=['GET'])
def reply(username, query_id, jsonreply):
    time_of_reply = datetime.datetime.now()
    # process json reply
    print(f"reply of {escape(username)} to query #{query_id} was {jsonreply}")
    response = json.loads(jsonreply)
    initialize_csv(escape(username))
    # store reply on disk
    type_of_reply = response["qtype"]
    #  write reply to ID file
    path_csv_id = path_base + escape(username) + "/id_annotations.csv"
    store_id_reply_in_csv(path_csv=path_csv_id, id=query_id, response_to_store=response, type_of_reply=type_of_reply)
    path_csv = path_base + escape(username) + "/annotations.csv"
    store_reply_in_csv(path_csv=path_csv, response_to_store=response, type_of_reply=type_of_reply)
    next_query_id = query_id + 1
    write_query_id(username=escape(username), query_id=str(next_query_id))
    # measure time taken to reply
    write_time_to_csv(username=username,
                      id=query_id,
                      time=time_of_reply,
                      type_of_timestamp="reply",
                      type_of_reply=type_of_reply)
    # redirect to next query
    return redirect(url_for(f'query', username=escape(username), query_id=next_query_id))


#
# Active Learning
#

def nearest_neighbor_selection(data, pool_data, head_id, n_nearest_neighbors):
    # load model
    print("loading model")
    model, params = finetune_model(data=data, pool_data=pool_data)
    # create embedding
    embedding = None  # implement your method to generate the ANN's embedding
    # sample nearest neighbors of head_id
    selected_body = get_neighbors(idx_head=head_id, n_body=n_nearest_neighbors, summarize_neighbors=sum_nn,
                                  embedding=embedding)
    return selected_body, embedding, model


def infotuple_selection(username, data, pool_data, head_id):
    # the embedding learned via InfoTuple
    # metric_learner = tste

    """
        Here, you can use the InfoTuple code from https://github.com/Sensory-Information-Processing-Lab/infotuple
        Also compare the pre-selection of the search space as described in the publication
    """

    # get the head and prepare choices
    # cluster center instead of random choices, instead of len(M_prime)
    choices, embedding, model = nearest_neighbor_selection(data, pool_data, head_id, n_nearest_neighbors=top_n)
    tuples = generate_infotuple_choices(choices=choices, n_permutations=n_permutations, len=n_body)

    # select using InfoTuple
    selected_body, tuple_qualities, tuple_probabilities, intermediate_params \
        = body_metrics.primal_body_selector(head_id,
                                            M_prime,
                                            tuples,
                                            n_samples=n_samples,
                                            mu=mu,
                                            verbose=False,
                                            dist_std=dists,
                                            f_downsample=f_downsample)
    print(
        f"selected tuple {selected_body} with qualities {tuple_qualities}, probs {tuple_probabilities} and interm. "
        f"params {intermediate_params}")
    return selected_body, embedding, model


@app.route('/active_query/<username>/<int:query_id>')
def active_query(username, query_id):
    print(f"Query {escape(username)} with active_query #{query_id}")

    # retrieve number of answered queries:
    initialize_csv(escape(username))
    path_csv = path_base + escape(username) + "/annotations.csv"
    n_replies = count_replies(path_csv=path_csv)
    query_type = get_type_of_query(path_csv=path_csv)
    # fixed_query_order = np.load(path_fixed_query_order)

    # load data
    # load annotations
    # measure time of AL methods

    head_id = query_id  # fixed_query_order[query_id]
    # print(f"Remap head from {query_id} to {head_id}")

    t0 = time.time()
    selected_body = []
    if query_type == "active" or query_type == "infotuple":
        # e.g., nearest neighbor selection or InfoTuple
        choices = [i for i in range(0, 1005) if i != query_id]
        head_id = query_id
        selected_body = np.random.choice(choices, size=n_body, replace=False)
    # store model and annotations
    n_of_annotations = count_rows_in_csv_by_type(type_of_reply=query_type, path_csv=path_csv, count_skipped=False)
    print(f"saved checkpoint {query_type} number {n_of_annotations}")

    def eval_triplet_accuracy(data, embedding, query_type_to_eval, eval_for_username):
        # implement triplet accuracy for your own dataset here.
        return 0.0

    mean_triplet_acc_rnd = eval_triplet_accuracy(data=None, embedding=None, query_type_to_eval="rnd",
                                                 eval_for_username=escape(username))
    write_acc_to_csv(username=username, query_type=query_type, id=n_of_annotations, type_of_test="rnd",
                     acc=mean_triplet_acc_rnd)
    print(f"{query_type}: mean_triplet_acc = {mean_triplet_acc_rnd} on rnd")

    ts_diff = time.time() - t0
    print("selection finished after", ts_diff)
    write_al_time_to_csv(username, id=query_id, type_of_al=query_type, seconds=ts_diff)

    head_id = str(head_id)
    plots_body = [str(idx) for idx in selected_body]
    print(f"Selected: {head_id} with body {plots_body}")
    # measure time taken to reply
    # type of reply as parameter
    write_time_to_csv(username=username,
                      id=query_id,
                      time=datetime.datetime.now(),
                      type_of_timestamp="query",
                      type_of_reply=query_type)

    # shuffle the plots so that they don't repeat exactly, and so that rnd/nn are mixed
    random.shuffle(plots_body)
    # render page
    return render_template('query_finetune.html', username=escape(username),
                           query_id=query_id, base_url=base_url,
                           plots_body=plots_body, plot_head=str(head_id),
                           n_replies=n_replies, query_type=query_type)
