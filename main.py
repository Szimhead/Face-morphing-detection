import csv
import json
import operator
import os
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import progressbar
import seaborn as sns
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from numpy.linalg import norm
from sklearn import cluster, metrics
from sklearn.manifold import MDS
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import configparser
from DET import DET

config = configparser.ConfigParser()
config.read('config.ini')

using_distance_vector = config['DEFAULT']['using_distance_vector'] == 'True'
db_dir = config['DEFAULT']['db_dir']

# probe_id ->
#   [
#   "probe" ->
#       [image_dir_filename -> feature vectors]
#   "reference" ->
#       [image_dir_filename -> feature vectors]
#   "facefusion" ->
#       [image_dir_filename -> feature vectors]
#   "facemorpher" ->
#       [image_dir_filename -> feature vectors]
#   "opencv" ->
#       [image_dir_filename -> feature vectors]
#   "ubo" ->
#       [image_dir_filename -> feature vectors]
#   ]

invalid = 0


def extract_face_features(directory, category, separator, uber_map, db):
    global uber_map_feret_changed, uber_map_frgc_changed, invalid
    bar = progressbar.ProgressBar(maxval=len(os.listdir(directory)),
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()],
                                  redirect_stdout=True)
    i = 0
    sleep(0.02)
    bar.start()
    sleep(0.05)
    for filename in os.listdir(directory):
        i += 1
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            probes = [p_id[0:p_id.find(separator)] for p_id in filename.split('vs_', 1)]
            for p_id in probes:
                if p_id not in uber_map:
                    uber_map[p_id] = {}
                if category not in uber_map[p_id]:
                    uber_map[p_id][category] = {}
                if f not in uber_map[p_id][category]:
                    image_name = f[0:f.rfind('.')]
                    # print(image_name)
                    img = ins_get_image(image_name)
                    if img is not None:
                        if db == "FERET":
                            uber_map_feret_changed = True
                        elif db == "FRGC":
                            uber_map_frgc_changed = True
                        face = app.get(img)
                        uber_map[p_id][category][f] = face[0].embedding.tolist()
                    else:
                        invalid += 1
        bar.update(i)
    sleep(0.05)
    bar.finish()


def create_uber_map_feret():
    filename = 'features_feret.json'
    with open(filename) as json_file:
        uber_map = json.load(json_file)
    json_file.close()
    print("Extracting face features for feret")
    print("Processing probe images")
    bona_fide_directory = db_dir + 'FERET\\bonafide_probe'
    extract_face_features(bona_fide_directory, "probe", '_', uber_map, "FERET")
    print("Processing reference images")
    ref_directory = db_dir + 'FERET\\bonafide_reference'
    extract_face_features(ref_directory, "reference", '_', uber_map, "FERET")
    print("Processing facefusion images")
    morph_directory = db_dir + 'FERET\\morphs_facefusion'
    extract_face_features(morph_directory, "facefusion", '_', uber_map, "FERET")
    print("Processing facemorpher images")
    morph_directory = db_dir + 'FERET\\morphs_facemorpher'
    extract_face_features(morph_directory, "facemorpher", '_', uber_map, "FERET")
    print("Processing opencv images")
    morph_directory = db_dir + 'FERET\\morphs_opencv'
    extract_face_features(morph_directory, "opencv", '_', uber_map, "FERET")
    print("Processing ubo images")
    morph_directory = db_dir + 'FERET\\morphs_ubo'
    extract_face_features(morph_directory, "ubo", '_', uber_map, "FERET")

    serialize_uber_map(filename, uber_map, uber_map_feret_changed)

    return uber_map


def create_uber_map_feret_one_tool(filename, directory):
    with open(filename) as json_file:
        uber_map = json.load(json_file)
    json_file.close()
    print("Extracting face features for feret,", filename)
    print("Processing probe images")
    bona_fide_directory = db_dir + 'FERET\\bonafide_probe'
    extract_face_features(bona_fide_directory, "probe", '_', uber_map, "FERET")
    print("Processing reference images")
    ref_directory = db_dir + 'FERET\\bonafide_reference'
    extract_face_features(ref_directory, "reference", '_', uber_map, "FERET")
    print("Processing {} images".format(filename))
    morph_directory = directory
    extract_face_features(morph_directory, "facefusion", '_', uber_map, "FERET")

    serialize_uber_map(filename, uber_map, uber_map_feret_changed)

    return uber_map


def create_uber_map_frgc():
    global uber_map_frgc
    with open('features_frgc.json') as json_file:
        uber_map_frgc = json.load(json_file)
    json_file.close()
    print("Extracting face features for frgc")
    print("Processing probe images")
    bona_fide_directory = db_dir + 'FRGC\\bonafide_probe'
    extract_face_features(bona_fide_directory, "probe", 'd', uber_map_frgc, "FRGC")
    print("Processing reference images")
    ref_directory = db_dir + 'FRGC\\bonafide_reference'
    extract_face_features(ref_directory, "reference", 'd', uber_map_frgc, "FRGC")
    print("Processing facefusion images")
    morph_directory = db_dir + 'FRGC\\morphs_facefusion'
    extract_face_features(morph_directory, "facefusion", 'd', uber_map_frgc, "FRGC")
    print("Processing facemorpher images")
    morph_directory = db_dir + 'FRGC\\morphs_facemorpher'
    extract_face_features(morph_directory, "facemorpher", 'd', uber_map_frgc, "FRGC")
    print("Processing opencv images")
    morph_directory = db_dir + 'FRGC\\morphs_opencv'
    extract_face_features(morph_directory, "opencv", 'd', uber_map_frgc, "FRGC")
    print("Processing ubo images")
    morph_directory = db_dir + 'FRGC\\morphs_ubo'
    extract_face_features(morph_directory, "ubo", 'd', uber_map_frgc, "FRGC")


def serialize_uber_map(json_filename, uber_map, uber_map_changed):
    if uber_map_changed:
        print("Serializing the map")
        with open(json_filename, "w") as outfile:
            json.dump(uber_map, outfile)
        outfile.close()
    else:
        print("Feature map not changed, skipping serializing")


def calculate_difference_vectors(uber_map, vec_diff, cos_vec_diff):
    for probe_id in uber_map:
        morph = []
        probe = []
        ref = []
        for category in uber_map[probe_id]:
            if category == 'probe':
                for image in uber_map[probe_id][category]:
                    probe.append(uber_map[probe_id][category][image])
            elif category == 'reference':
                for image in uber_map[probe_id][category]:
                    ref.append(uber_map[probe_id][category][image])
            else:
                for image in uber_map[probe_id][category]:
                    morph.append(uber_map[probe_id][category][image])

        for p in probe:
            for r in ref:
                vec_diff["bf"].append(np.subtract(r, p))
                cos_vec_diff["bf"].append(np.dot(r, p) / (norm(r) * norm(p)))
            for m in morph:
                vec_diff["m"].append(np.subtract(m, p))
                np.dot(m, p) / (norm(m) * norm(p))
                cos_vec_diff["m"].append(np.dot(m, p) / (norm(m) * norm(p)))


def merge_difference_vectors(vec_diff, all_faces):
    for vectors in vec_diff.values():
        all_faces.extend(vectors)


def cluster_faces(all_faces, vec_diff, mds_filename):
    mds = MDS(n_components=2, random_state=0)

    if len(all_faces) > 10000:
        cluster_sizes = [10, 50, 100, 1000, 2000, 5000, 8000, 10000]
        print("Clustering difference vectors")
        clustered_bona_fide, clustered_morphs = None, None
        for size in cluster_sizes:
            print("Clustering using", size, "clusters")
            if len(vec_diff["bf"]) > size:
                print("Clustering bona fide subset")
                clustered_bona_fide = cluster.KMeans(size).fit(vec_diff["bf"]).cluster_centers_
            else:
                print("Bona fide subset smaller than cluster size - using full subset")
                clustered_bona_fide = vec_diff["bf"]
            if len(vec_diff["m"]) > size:
                print("Clustering morph subset")
                clustered_morphs = cluster.KMeans(size).fit(vec_diff["m"]).cluster_centers_
            else:
                print("Morph subset smaller than cluster size - using full subset")
                clustered_morphs = vec_diff["m"]

            merged = np.concatenate((clustered_bona_fide, clustered_morphs))

            print("MDS transformation")
            mds_vectors = mds.fit_transform(merged)

            print("Transformation done, saving results to a file\n")
            with open(mds_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(mds_vectors)
            f.close()

            plt.ion()
            plt.show()
            plt.scatter(mds_vectors[0:size, 0], mds_vectors[0:size, 1], color='g', s=5)
            plt.scatter(mds_vectors[size:, 0], mds_vectors[size:, 1], color='purple', s=5)

            plt.draw()
            plt.pause(0.01)
        return clustered_bona_fide, len(clustered_bona_fide)
    return mds.fit_transform(all_faces), len(vec_diff["bf"])


def handle_mds_vector(mds_filename, all_faces, mds_vectors, vec_diff):
    with open(mds_filename, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            mds_vectors.append([float(x) for x in row])
    mds_vectors = np.array(mds_vectors)
    bf_card = len(vec_diff["bf"])
    if len(mds_vectors) == 0:
        mds_vectors, bf_card = cluster_faces(all_faces, vec_diff, mds_filename)
        with open(mds_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(mds_vectors)
        f.close()
    return mds_vectors, bf_card


def display_difference_vectors(mds_vectors, bona_fide_cardinality, title):
    bf_points_0 = mds_vectors[0:bona_fide_cardinality, 0]
    bf_points_1 = mds_vectors[0:bona_fide_cardinality, 1]
    plt.scatter(bf_points_0, bf_points_1, color='g', s=5, alpha=0.3, label="Bona fide")
    plt.title(title, size=40)
    m_points_0 = mds_vectors[bona_fide_cardinality:, 0]
    m_points_1 = mds_vectors[bona_fide_cardinality:, 1]
    plt.scatter(m_points_0, m_points_1, color='purple', s=5, alpha=0.3, label="Morph")
    plt.xlabel("Dim0", size=35)
    plt.ylabel("Dim1", size=35)
    plt.legend(fontsize=40)
    plt.show()


def plot_histogram(mated_scores, nonmated_scores, normalise=True, savename=None, title=None):
    def normalise_scores(distribution):
        return np.ones_like(distribution) / len(distribution)

    plt.figure(figsize=figure_size)
    if normalise:
        plt.hist(mated_scores, bins=50, weights=normalise_scores(mated_scores), color=mated_colour, alpha=0.5,
                 label=mated_label)
        plt.hist(nonmated_scores, bins=50, weights=normalise_scores(nonmated_scores), color=nonmated_colour, alpha=0.5,
                 label=nonmated_label)
        xlabel = "Probability Density"
    else:
        plt.hist(mated_scores, bins=50, color=mated_colour, alpha=0.5, label=mated_label)
        plt.hist(nonmated_scores, bins=50, color=nonmated_colour, alpha=0.5, label=nonmated_label)
        xlabel = "Count"
    plt.xlabel("Comparison Score", size=label_fontsize)
    plt.ylabel(xlabel, size=label_fontsize)
    plt.title(title, size=40)
    plt.grid(True)
    plt.legend(fontsize=legend_fontsize)

    if savename is not None:
        plt.savefig(savename, bbox_inches="tight")
        plt.cla()
        plt.clf()
        plt.close()
    else:
        plt.show()


def calculate_roc(gscores, iscores, ds_scores=False, rates=True):
    if isinstance(gscores, list):
        gscores = np.array(gscores, dtype=np.float64)

    if isinstance(iscores, list):
        iscores = np.array(iscores, dtype=np.float64)

    if gscores.dtype == np.int:
        gscores = np.float64(gscores)

    if iscores.dtype == np.int:
        iscores = np.float64(iscores)

    if ds_scores:
        gscores = gscores * -1
        iscores = iscores * -1

    gscores_number = len(gscores)
    iscores_number = len(iscores)

    gscores = zip(gscores, [1] * gscores_number)
    iscores = zip(iscores, [0] * iscores_number)

    gscores = list(gscores)
    iscores = list(iscores)

    scores = np.array(sorted(gscores + iscores, key=operator.itemgetter(0)))
    cumul = np.cumsum(scores[:, 1])

    thresholds, u_indices = np.unique(scores[:, 0], return_index=True)

    fnm = cumul[u_indices] - scores[u_indices][:, 1]
    fm = iscores_number - (u_indices - fnm)

    if rates:
        fnm_rates = fnm / gscores_number
        fm_rates = fm / iscores_number
    else:
        fnm_rates = fnm
        fm_rates = fm

    if ds_scores:
        return thresholds * -1, fm_rates, fnm_rates

    return thresholds, fm_rates, fnm_rates


def get_fmr_op(fmr, fnmr, op):
    index = np.argmin(abs(fmr - op))
    return fnmr[index]


def get_eer(fmr, fnmr):
    diff = fmr - fnmr
    t2 = np.where(diff <= 0)[0]

    if len(t2) > 0:
        t2 = t2[0]
    else:
        return 0, 1, 1, 1

    return (fnmr[t2] + fmr[t2]) / 2


app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# FRGC - testing dataset

print("Processing FRGC dataset")
uber_map_frgc = {}
uber_map_frgc_changed = False
create_uber_map_frgc()

print("Invalid images: ", invalid)

serialize_uber_map("features_frgc.json", uber_map_frgc, uber_map_frgc_changed)

print("Combining features for FRGC")

vec_diff_frgc = {"bf": [], "m": []}
cos_vec_diff_frgc = {"bf": [], "m": []}
euc_vec_diff_frgc = {"bf": [], "m": []}

print("Calculating difference vectors")
calculate_difference_vectors(uber_map_frgc, vec_diff_frgc, cos_vec_diff_frgc)

all_faces_frgc = list()
similarities_merged = list()
print("Merging difference vectors")
if using_distance_vector:
    print("Using distance vector")
    merge_difference_vectors(vec_diff_frgc, all_faces_frgc)
else:
    print("Using cosine distance")
    merge_difference_vectors(cos_vec_diff_frgc, all_faces_frgc)
    all_faces_frgc = np.array(all_faces_frgc).reshape(-1, 1)

mds_vectors_frgc = []

print("Multidimensional scaling")
mds_vectors_frgc, bona_fide_cardinality_frgc = handle_mds_vector('mds_frgc.csv', all_faces_frgc, mds_vectors_frgc,
                                                                 vec_diff_frgc)

print("Displaying difference vectors")
display_difference_vectors(mds_vectors_frgc, bona_fide_cardinality_frgc, "FRGC")

bona_fide_cardinality_frgc = len(vec_diff_frgc["bf"])
morph_cardinality_frgc = len(vec_diff_frgc["m"])

print()

# FERET - training dataset

print("Processing FERET dataset")
tars = []
nons = []

uber_map_feret_changed = False
uber_maps_feret = {"all_feret": create_uber_map_feret(),
                   "facefusion": create_uber_map_feret_one_tool('features_feret_facefusion.json',
                                                                db_dir + 'FERET\\morphs_facefusion'),
                   "facemorpher": create_uber_map_feret_one_tool('features_feret_facemorpher.json',
                                                                 db_dir + 'FERET\\morphs_facemorpher'),
                   "opencv": create_uber_map_feret_one_tool('features_feret_opencv.json',
                                                            db_dir + 'FERET\\morphs_opencv'),
                   "ubo": create_uber_map_feret_one_tool('features_feret_ubo.json',
                                                         db_dir + 'FERET\\morphs_ubo')}

svm_params = {"all_feret": {'C': 1, 'gamma': 0.001},
              "facefusion": {'C': 1, 'gamma': 0.001},
              "facemorpher": {'C': 0.1, 'gamma': 0.001},
              "opencv": {'C': 10, 'gamma': 0.001},
              "ubo": {'C': 1, 'gamma': 0.001}}

for uber_map_name in uber_maps_feret:
    print("Processing", uber_map_name)
    uber_map_feret = uber_maps_feret[uber_map_name]

    # rimg = app.draw_on(img, faces)
    print("Combining features for FERET")

    vec_diff_feret = {"bf": [], "m": []}
    cos_vec_diff_feret = {"bf": [], "m": []}
    print("Calculating difference vectors")
    calculate_difference_vectors(uber_map_feret, vec_diff_feret, cos_vec_diff_feret)

    all_faces_feret = list()
    if using_distance_vector:
        print("Using distance vector")
        merge_difference_vectors(vec_diff_feret, all_faces_feret)
    else:
        print("Using cosine distance")
        merge_difference_vectors(cos_vec_diff_feret, all_faces_feret)
        all_faces_feret = np.array(all_faces_feret).reshape(-1, 1)

    mds_vectors_feret = []

    print("Multidimensional scaling")
    mds_vectors_feret, bona_fide_cardinality_feret = handle_mds_vector('mds_feret_' + uber_map_name + '.csv',
                                                                       all_faces_feret, mds_vectors_feret,
                                                                       vec_diff_feret)

    morph_cardinality_feret = len(mds_vectors_feret) - bona_fide_cardinality_feret

    display_difference_vectors(mds_vectors_feret, bona_fide_cardinality_feret, uber_map_name)

    # SVM

    labels_feret = np.full(bona_fide_cardinality_feret, "bona fide")
    labels_feret = np.append(labels_feret, np.full(morph_cardinality_feret, "morph"))

    labels_frgc = np.full(bona_fide_cardinality_frgc, "bona fide")
    labels_frgc = np.append(labels_frgc, np.full(morph_cardinality_frgc, "morph"))

    if using_distance_vector:
        print(svm_params[uber_map_name])
        clf = SVC(kernel='rbf', C=svm_params[uber_map_name]['C'], gamma=svm_params[uber_map_name]['gamma'])
        # training non-linear model
        clf.fit(all_faces_feret, labels_feret)
    else:
        params = {
            'C': [0.1, 1, 10, 100, 1000],
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
            'kernel': ['rbf']
        }

        clf = GridSearchCV(
            estimator=SVC(),
            param_grid=params,
            cv=5,
            n_jobs=5,
            verbose=1
        )
        # training non-linear model
        clf.fit(all_faces_feret, labels_feret)
        print(clf.best_params_)

    y_pred = clf.predict(all_faces_frgc)
    print("Accuracy:", metrics.accuracy_score(labels_frgc, y_pred))

    tar = []
    non = []
    result_distances = clf.decision_function(all_faces_frgc)
    max_dist = max(result_distances)
    res_dist_min = np.min(result_distances)
    res_dist_max = np.max(result_distances)
    result_distances = (result_distances - res_dist_min) / (res_dist_max - res_dist_min)

    for value, res, label in zip(all_faces_frgc, result_distances, labels_frgc):
        if label == "morph":
            non.append(res)
        else:
            tar.append(res)


    def adjust_scores_for_DET(scores_array, scores_type):
        scores_array = np.asarray(scores_array)
        if scores_type == "similarity":
            return scores_array
        elif scores_type == "dissimilarity":
            return -scores_array + 1
        else:
            raise ValueError(f"Unknown type of comparison scores: {scores_type}")


    if using_distance_vector:
        non = adjust_scores_for_DET(non, "dissimilarity")
        tar = adjust_scores_for_DET(tar, "dissimilarity")
    else:
        non = adjust_scores_for_DET(non, "dissimilarity")
        tar = adjust_scores_for_DET(tar, "dissimilarity")

    # Plotting

    mated_colour = "green"
    mated_label = "Mated scores"
    nonmated_colour = "red"
    nonmated_label = "Non-mated scores"

    figure_size = (12, 6)
    alpha_shade = 0.25
    alpha_fill = 1.0
    linewidth = 2
    legend_loc = "upper left"
    legend_anchor = (1.0, 1.02)
    legend_cols = 1
    legend_fontsize = 35
    label_fontsize = 35

    threshold_colour = "black"
    threshold_style = "--"
    round_digits = 5
    sns.set(style="white", palette="muted", color_codes=True)
    plt.rc("axes", axisbelow=True)

    plot_histogram(tar, non, normalise=True, title=uber_map_name)

    tars.append(tar)
    nons.append(non)
    print()

print("Creating DET")

det = DET(biometric_evaluation_type='PAD', abbreviate_axes=True, plot_eer_line=True, plot_title="D-MAD")
det.x_limits = np.array([1e-4, .7])
det.y_limits = np.array([1e-4, .7])
det.x_ticks = np.array([1e-3, 1e-2, 5e-2, 20e-2, 40e-2, 70e-2])
det.x_ticklabels = np.array(['0.1', '1', '5', '20', '40', '70'])
det.y_ticks = np.array([1e-3, 1e-2, 5e-2, 20e-2, 40e-2, 70e-2])
det.y_ticklabels = np.array(['0.1', '1', '5', '20', '40', '70'])
det.create_figure()
for t, n, tool in zip(tars, nons, uber_maps_feret):
    thresholds_system4, fmrs_system4, fnmrs_system4 = calculate_roc(t, n, ds_scores=False)
    print("System", tool, "EER:", round(get_eer(fmrs_system4, fnmrs_system4) * 100, round_digits))
    det.plot(tar=t, non=n, label=tool)
det.legend_on(loc="upper right", fontsize=15)
det.save('DET_produced', 'png')
det.show()
