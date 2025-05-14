import numpy as np
from easydl import *
from torch import nn
import torch
import torch.nn.functional as F
from scipy import linalg
from sklearn.mixture import BayesianGaussianMixture
import pickle, os
from mixture import nmi
from datetime import datetime
from sklearn.manifold import TSNE
import plotly.express as px
from function import BetaMixture1D
import torch.nn.functional as F

def seed_everything(seed=1234):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)


def entropy_numpy(logits):
    K = logits.shape[-1]  # Number of classes
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))  # Stable softmax
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)  # Compute probabilities
    log_probs = np.log(probs + 1e-9)  # Avoid log(0)
    entropy = -np.sum(probs * log_probs, axis=-1)  # Compute entropy per sample
    max_entropy = np.log(K)  # Max entropy log(K)
    normalized_entropy = entropy / max_entropy  # Normalize to [0, 1]
    return normalized_entropy


def report(predict_id, label, source_classes):
    counters = [AccuracyCounter() for x in range(len(source_classes) + 1)]
    unknown_test_truth, unknown_test_pred = [], []
    known_test_truth, known_test_pred = [], []

    for (each_pred_id, each_label) in zip(predict_id, label):
        if each_label in source_classes:
            counters[each_label].Ntotal += 1.0
            if each_pred_id == each_label:
                counters[each_label].Ncorrect += 1.0
            known_test_pred.append(each_pred_id)
            known_test_truth.append(each_label)
        else:
            unknown_test_pred.append(each_pred_id)
            unknown_test_truth.append(each_label)
            counters[-1].Ntotal += 1.0
            if each_pred_id >= len(source_classes):
                counters[-1].Ncorrect += 1.0
    acc_tests = {ii: x.reportAccuracy() for ii, x in enumerate(counters) if not np.isnan(x.reportAccuracy())}
    return counters, unknown_test_truth, unknown_test_pred, known_test_truth, known_test_pred, acc_tests


# class LogUpdater():
#     def __init__(self):
#         self.tgt_member_all, self.tgt_member_new_all, self.accs, self.unknown_test_truths, self.unknown_test_preds = [], [], [], [], []
#         self.counters_all1, self.accs1, self.unknown_test_truths1, self.unknown_test_preds1 = [], [], [], []
#         self.dp_preds_all1, self.dp_preds_all2, self.tgt_member_new2_all, self.counters_all = [], [], [], []
#         self.tgt_member_post_all, self.predict_idl_all, self.filter_out_idx_all = [], [], []

#     def update(self, unknown_test_truth=None, unknown_test_pred=None, unknown_test_truth1=None, unknown_test_pred1=None,
#                counters=None, counters1=None, preds=None, tgt_member=None,
#                tgt_member_post=None, predict_idl=None, filter_out_idx=None,
#                tgt_member_new=None, acc_tests=None, acc_tests1=None):
#         self.unknown_test_truths.append(unknown_test_truth)
#         self.unknown_test_preds.append(unknown_test_pred)
#         self.counters_all.append(counters)
#         self.counters_all1.append(counters1)
#         self.dp_preds_all1.append(preds)
#         self.tgt_member_all.append(tgt_member)
#         self.tgt_member_new_all.append(tgt_member_new)
#         self.tgt_member_post_all.append(tgt_member_post)
#         self.predict_idl_all.append(predict_idl)
#         self.filter_out_idx_all.append(filter_out_idx)
#         self.accs.append(acc_tests)
#         self.accs1.append(acc_tests1)
#         self.unknown_test_truths1.append(unknown_test_truth1)
#         self.unknown_test_preds1.append(unknown_test_pred1)

#     def save(self, log_dir, args):
#         with open(os.path.join(log_dir, 'tgt.pkl'), 'wb') as f:
#             pickle.dump([self.tgt_member_all, self.tgt_member_new_all, self.dp_preds_all1, self.accs, self.accs1,
#                          self.counters_all, self.counters_all1, self.unknown_test_preds, self.unknown_test_preds1,
#                          self.unknown_test_truths, self.unknown_test_truths1,
#                          self.tgt_member_post_all, self.predict_idl_all, self.filter_out_idx_all, args], f)



def plotpy_save(src_data, tar_data, src_label, tar_label, log_dir, extra=None):
    data = np.concatenate([src_data, tar_data], axis=0)
    y = np.concatenate([src_label, tar_label], axis=0)
    m = np.array([1 for _ in range(len(src_label))] + [5 for _ in range(len(tar_label))])
    s = np.array([2 for _ in range(len(src_label))] + [5 for _ in range(len(tar_label))])
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(data)
    fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], color=y, symbol=m, size=s)
    fig.update_layout(
        title="t-SNE visualization of Custom Classification dataset",
        xaxis_title="First t-SNE",
        yaxis_title="Second t-SNE",
    )
    #fig.show()
    import datetime
    now = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    if extra:
        now = "{}_{}".format(now, extra)
    fig.write_image(os.path.join(log_dir, "{}.png".format(now)))


def plotpy2(tar_data, tar_label, cap='truth'):
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(tar_data)
    fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], color=tar_label)
    fig.update_layout(
        title="t-SNE {}".format(cap),
        xaxis_title="First t-SNE",
        yaxis_title="Second t-SNE",
    )
    fig.show()



class Memory(nn.Module):
    def __init__(self, num_cls=10, feat_dim=256):
        super(Memory, self).__init__()
        self.num_cls = num_cls
        self.feat_dim = feat_dim
        self.memory = torch.zeros(self.num_cls, feat_dim, dtype=torch.float).cuda()

    def init(self, embeddings, labels, device):
        unique_classes, class_counts = np.unique(labels, return_counts=True)
        print('=== unique classes==============', unique_classes)
        euclidean_centers = np.zeros((len(unique_classes), embeddings.shape[1]))
        for i, class_label in enumerate(unique_classes):
            # Get indices of embeddings belonging to current class
            class_indices = np.where(labels == class_label)[0]
            # Compute mean of embeddings for current class
            class_embeddings = embeddings[class_indices]
            class_center = np.mean(class_embeddings, axis=0)
            # Store the Euclidean center for the current class
            euclidean_centers[i] = class_center
        euclidean_centers = torch.from_numpy(euclidean_centers).float().to(device)
        self.memory = F.normalize(euclidean_centers, p=2, dim=-1)

    def update_center_by_simi(self, batch_center, flags):
        old_center = self.memory
        update_wei = (old_center * batch_center).sum(dim=-1).squeeze()
        update_wei = update_wei.view(-1, 1).expand_as(old_center)
        flags = flags.expand_as(self.memory)
        update_wei = torch.ones_like(flags) - (1 - update_wei) * flags  # update_wei
        self.memory = update_wei * self.memory + (1 - update_wei) * batch_center
        self.memory = F.normalize(self.memory, p=2, dim=-1)

    def update(self, feat, label):
        feat = feat.detach()
        batch_center = []
        empty = torch.zeros((1, self.feat_dim), dtype=torch.float).cuda()
        flags = []
        for i in range(self.num_cls):
            mask = label == i
            if mask.sum() == 0:
                flags.append(torch.Tensor([.0]).cuda())
                batch_center.append(empty)
                continue
            index = mask.squeeze().nonzero(as_tuple=False)
            cur_feat = feat[index, :]
            cur_feat = cur_feat.sum(dim=0)
            cur_feat = F.normalize(cur_feat, p=2, dim=-1)
            cur_feat = cur_feat.view(1, -1)

            flags.append(torch.Tensor([1.0]).cuda())
            batch_center.append(cur_feat)
        batch_center = torch.cat(batch_center, dim=0)
        flags = torch.stack(flags).cuda()
        self.update_center_by_simi(batch_center, flags)

    def forward(self, feat, label, t=0.1): # t = 1.0 or 0.1
        feat = F.normalize(feat, p=2, dim=-1)
        self.update(feat, label.unsqueeze(0))
        simis = torch.matmul(feat, self.memory.transpose(0, 1))
        simis = simis / t
        loss = F.cross_entropy(simis, label.squeeze())
        return loss.mean()


def post_match(t_codes):
    # make label of t_codes to be 0-index (0,1,2,...)
    # then return the match relationship between src center and tar center
    # return reindex t_codes and the map (src_index, tar_index)
    unique_elements, indices = np.unique(t_codes, return_inverse=True)
    #match = dict(zip(unique_elements, range(len(unique_elements))))
    match = dict(zip(range(len(unique_elements)), unique_elements))
    return indices, match

# def post_match(t_codes, num_src_cls):
#     # make label of t_codes to be 0-index (0,1,2,...)
#     # then return the match relationship between src center and tar center
#     # return reindex t_codes and the map (src_index, tar_index)
#     t_codes_unk = t_codes[t_codes >= num_src_cls]
#     unique_elements_unk, indices_unk = np.unique(t_codes_unk, return_inverse=True)
#     #match = dict(zip(unique_elements, range(len(unique_elements))))
#     indices_unk += num_src_cls
#     indices = np.copy(t_codes)
#     indices[t_codes >= num_src_cls] = indices_unk
#     match = dict(zip(range(num_src_cls, len(unique_elements_unk)+num_src_cls), unique_elements_unk))
#     match.update(dict(zip(range(num_src_cls), range(num_src_cls))))
#     return indices, match


def cos_simi(x1, x2):
    simi = torch.matmul(x1, x2.transpose(0, 1))
    return simi


def map_values(x, mapping_dict):
    return mapping_dict.get(x, x)

# Vectorize the function
vectorized_map = np.vectorize(map_values)



def valid(t_centers , Net, target_test_dl, output_device, source_classes, tgt_match, ttype='OPDA'):
    tgt_member, tgt_predict = [], []
    Net.eval()
    for i, (im_target, label_target) in enumerate(target_test_dl):
        im_target = im_target.to(output_device)
        _, feature_target, _ = Net(im_target)
        tgt_member.append(label_target.detach().cpu().numpy())
        clus_index = cos_simi(F.normalize(feature_target, p=2, dim=-1), t_centers).argmax(dim=-1) # get argmax of similarity to cluster
        tgt_predict.append(clus_index.detach().cpu().numpy())

    tgt_member = np.concatenate(tgt_member, axis=0)
    tgt_predict = np.concatenate(tgt_predict, axis=0)
    tgt_predict = vectorized_map(tgt_predict, tgt_match)
    Net.train()
    if ttype == 'PDA':
        k_acc = merge_perf_pda(tgt_member, tgt_predict)
        print(
            '*********evaluate performance********** k_acc: {}'.format(k_acc))
        return k_acc, tgt_member, tgt_predict
    else:
        merge_perf(tgt_member, tgt_predict, ncls=len(source_classes))
        counters, unknown_test_truth, unknown_test_pred, known_test_truth, known_test_pred, acc_tests = report(tgt_predict, tgt_member, source_classes)
        acc_test = np.round(np.mean(list(acc_tests.values())), 3)
        acc_testv = np.array(list(acc_tests.values()))
        kn_acc = np.mean(acc_testv[:-1])
        unk_acc = acc_testv[-1]
        hos = 2 * (kn_acc * unk_acc) / (kn_acc + unk_acc)
        nmi_v = nmi(tgt_member,tgt_predict)
        unk_nmi = nmi(unknown_test_truth, unknown_test_pred)
        k_acc = np.sum(np.array(known_test_truth) == np.array(known_test_pred)) / len(known_test_truth)
        print(
            '*********evaluate performance********** acc_tests: {}, acc_test: {}, final_hos: {}, nmi: {}, unk_nmi: {}, k_acc: {}'.format(acc_tests, acc_test, hos, nmi_v, unk_nmi, k_acc))
        return counters, unknown_test_truth, unknown_test_pred, acc_tests, acc_test, hos, nmi_v, unk_nmi, k_acc, tgt_member, tgt_predict

def gen_cluster_input(Net, target_test_dl, output_device):
    tgt_embedding, tgt_member = [], []
    with TrainingModeManager([Net.feature_extractor, Net.bottle_neck, Net.classifier],
                             train=False) as mgr, torch.no_grad():
        for i, (im_target, label_target) in enumerate(target_test_dl):
            im_target = im_target.to(output_device)
            _, feature_target, _ = Net(im_target)
            tgt_embedding.append(feature_target.detach().cpu().numpy())
            tgt_member.append(label_target.detach().cpu().numpy())
        tgt_embedding = np.concatenate(tgt_embedding, axis=0)
        tgt_member = np.concatenate(tgt_member, axis=0)
    return tgt_embedding, tgt_member

def merge_cluster(Cluster, tgt_embedding, tgt_member, tgt_predict_src, plot=False, num_src_cls=20):
    tgt_embedding_k = tgt_embedding[tgt_predict_src < num_src_cls]
    tgt_embedding_unk = tgt_embedding[tgt_predict_src >= num_src_cls]
    tgt_predict_k = tgt_predict_src[tgt_predict_src < num_src_cls]
    unique_tgt_predict_k = np.unique(tgt_predict_k)
    tgt_predict = np.copy(tgt_predict_src)
    embedding = np.concatenate([tgt_embedding_k, tgt_embedding_unk], axis=0)
    # if plot:
    #     tgt_plot = np.copy(tgt_predict)
    #     tgt_plot[tgt_plot >= num_src_cls] = 30
    #     plotpy2(tgt_embedding, tgt_plot)
    #print('feature size', embedding.shape)
    tgt_member_new, _, _ = Cluster.fit_merge(embedding, tgt_predict_k, v1=False)
    mask = ~np.isin(tgt_member_new, unique_tgt_predict_k)
    tgt_member_new[mask] += num_src_cls
    tgt_predict[tgt_predict_src >= num_src_cls] = tgt_member_new
    if plot:
        plotpy2(tgt_embedding, tgt_member)
        plotpy2(tgt_embedding, tgt_predict)
    return tgt_predict


# def plot_cluster(Net, target_test_dl, output_device, tgt_predict, cap='step'):
#     tgt_embedding, tgt_member = [], []
#     with TrainingModeManager([Net.feature_extractor, Net.bottle_neck, Net.classifier],
#                              train=False) as mgr, torch.no_grad():
#         for i, (im_target, label_target) in enumerate(target_test_dl):
#             im_target = im_target.to(output_device)
#             _, feature_target, _ = Net(im_target)
#             tgt_embedding.append(feature_target.detach().cpu().numpy())
#             tgt_member.append(label_target.detach().cpu().numpy())

#     tgt_embedding = np.concatenate(tgt_embedding, axis=0)
#     tgt_member = np.concatenate(tgt_member, axis=0)
#     plotpy2(tgt_embedding, tgt_member, cap='{} truth'.format(cap))
#     plotpy2(tgt_embedding, tgt_predict, cap='{} estimate'.format(cap))


def ExpWeight(step, gamma=3, max_iter=5000):
    step = max_iter-step
    ans = 1.0 * (np.exp(- gamma * step * 1.0 / max_iter))
    return float(ans)



# def merge_perf(tgt_member, tgt_member_new, ncls):
#     tar_label_known = tgt_member[tgt_member < ncls]
#     tgt_member_new_known = tgt_member_new[tgt_member < ncls]
#     tar_label_unknown = tgt_member[tgt_member > ncls-1]
#     tgt_member_new_unknown = tgt_member_new[tgt_member > ncls-1]
#     print('=====nmi========', nmi(tgt_member_new,
#                                   tgt_member))#, tgt_member_new, tgt_member, tgt_member_new.shape, tgt_member.shape)
#     print('known acc', np.sum(tar_label_known == tgt_member_new_known) / np.sum(tgt_member < ncls),
#           len(tar_label_known))#, tar_label_known, tgt_member_new_known)
#     print('unknown nmi', nmi(tar_label_unknown, tgt_member_new_unknown))#, tar_label_unknown, tgt_member_new_unknown, len(tar_label_unknown))
#     TP = np.sum((tgt_member > ncls-1) & (tgt_member_new > ncls-1))
#     TN = np.sum((tgt_member < ncls) & (tgt_member_new < ncls))
#     FP = np.sum((tgt_member < ncls) & (tgt_member_new > ncls-1))
#     FN = np.sum((tgt_member > ncls-1) & (tgt_member_new < ncls))
#     print('recall is ', TP / (TP + FN), TP, TP + FN)
#     print('precision is', TP / (TP + FP), TP, TP + FP)


def merge_perf(tgt_member, tgt_member_new, ncls):
    tar_label_known = tgt_member[tgt_member < ncls]
    tgt_member_new_known = tgt_member_new[tgt_member < ncls]
    tar_label_unknown = tgt_member[tgt_member > ncls-1]
    tgt_member_new_unknown = tgt_member_new[tgt_member > ncls-1]
    nmi_v = nmi(tgt_member_new, tgt_member)
    #print('=====nmi========', nmi_v)#, tgt_member_new, tgt_member, tgt_member_new.shape, tgt_member.shape)
    k_acc = np.sum(tar_label_known == tgt_member_new_known) / np.sum(tgt_member < ncls)
    k_acc_total = {}
    k_num_total = {}
    k_pos_total = {}
    for c in np.unique(tar_label_known):
        tr = tar_label_known[tar_label_known==c]
        es = tgt_member_new_known[tar_label_known==c]
        pacc = np.sum(tr==es) / len(tr)
        k_acc_total[c] = pacc
        k_num_total[c] = len(tr)
        k_pos_total[c] = np.sum(tr==es)

    print(k_acc_total)
    print(k_num_total)
    print(k_pos_total)

    known_acc = np.array(list(k_acc_total.values())).mean()
    unknown_acc = np.sum(tgt_member_new_unknown > ncls-1) / len(tgt_member_new_unknown)
    h_score = 2 * known_acc * unknown_acc / (known_acc + unknown_acc + 1e-5)
    print('m_hos: {}; m_known:{}; m_unknown:{}'.format(h_score, known_acc, unknown_acc))
    #print('known acc', k_acc)#, tar_label_known, tgt_member_new_known)
    #len(tar_label_known), tar_label_known, tgt_member_new_known)
    uk_nmi = nmi(tar_label_unknown, tgt_member_new_unknown)
    print('m_nmi: {}; unknown m_nmi: {}; overall known m_acc: {}'.format(nmi_v, uk_nmi, k_acc))#, tar_label_unknown, tar_label_unknown, tgt_member_new_unknown)
    #tgt_member_new_unknown, len(tar_label_unknown))
    TP = np.sum((tgt_member > ncls-1) & (tgt_member_new > ncls-1))
    TN = np.sum((tgt_member < ncls) & (tgt_member_new < ncls))
    FP = np.sum((tgt_member < ncls) & (tgt_member_new > ncls-1))
    FN = np.sum((tgt_member > ncls-1) & (tgt_member_new < ncls))
    rec = TP / (TP + FN)
    print('recall is ', rec, TP, TP + FN)
    prec = TP / (TP + FP)
    print('precision is', prec, TP, TP + FP)
    return nmi_v, k_acc, uk_nmi, rec, prec

def merge_perf_only(tgt_member, tgt_member_new, ncls):
    tar_label_known = tgt_member[tgt_member < ncls]
    tgt_member_new_known = tgt_member_new[tgt_member < ncls]
    tar_label_unknown = tgt_member[tgt_member > ncls-1]
    tgt_member_new_unknown = tgt_member_new[tgt_member > ncls-1]
    nmi_v = nmi(tgt_member_new, tgt_member)
    #print('=====nmi========', nmi_v)#, tgt_member_new, tgt_member, tgt_member_new.shape, tgt_member.shape)
    k_acc = np.sum(tar_label_known == tgt_member_new_known) / np.sum(tgt_member < ncls)
    k_acc_total = {}
    k_num_total = {}
    k_pos_total = {}
    for c in np.unique(tar_label_known):
        tr = tar_label_known[tar_label_known==c]
        es = tgt_member_new_known[tar_label_known==c]
        pacc = np.sum(tr==es) / len(tr)
        k_acc_total[c] = pacc
        k_num_total[c] = len(tr)
        k_pos_total[c] = np.sum(tr==es)

    print(k_acc_total)
    print(k_num_total)
    print(k_pos_total)
    known_acc = np.array(list(k_acc_total.values())).mean()
    unknown_acc = np.sum(tgt_member_new_unknown > ncls-1) / len(tgt_member_new_unknown)
    acc_total = {**k_acc_total, ncls: unknown_acc}
    acc_test = np.round(np.mean(list(acc_total.values())), 3)
    h_score = 2 * known_acc * unknown_acc / (known_acc + unknown_acc + 1e-5)
    print('m_hos: {}; m_known:{}; m_unknown:{}'.format(h_score, known_acc, unknown_acc))
    #print('known acc', k_acc)#, tar_label_known, tgt_member_new_known)
    #len(tar_label_known), tar_label_known, tgt_member_new_known)
    uk_nmi = nmi(tar_label_unknown, tgt_member_new_unknown)
    print('m_nmi: {}; unknown m_nmi: {}; overall known m_acc: {}'.format(nmi_v, uk_nmi, k_acc))#, tar_label_unknown, tar_label_unknown, tgt_member_new_unknown)
    #tgt_member_new_unknown, len(tar_label_unknown))
    TP = np.sum((tgt_member > ncls-1) & (tgt_member_new > ncls-1))
    TN = np.sum((tgt_member < ncls) & (tgt_member_new < ncls))
    FP = np.sum((tgt_member < ncls) & (tgt_member_new > ncls-1))
    FN = np.sum((tgt_member > ncls-1) & (tgt_member_new < ncls))
    rec = TP / (TP + FN)
    print('recall is ', rec, TP, TP + FN)
    prec = TP / (TP + FP)
    print('precision is', prec, TP, TP + FP)
    return h_score, acc_test, nmi_v, k_acc, uk_nmi, rec, prec


def merge_perf_pda(tgt_member, tgt_member_new):
    return np.sum(tgt_member == tgt_member_new) / len(tgt_member)




class sim_bmm(object):
    def __init__(self, norm=False):
        self.bmm_model = BetaMixture1D()
        self.norm = norm
    
    def compute_probabilities_batch(self, sim_t, unk=1):
        sim_t[sim_t >= 1 - 1e-4] = 1 - 1e-4
        sim_t[sim_t <=  1e-4] = 1e-4
        B = self.bmm_model.posterior(sim_t, unk)
        return B

    def bmm_fit(self, sim_array):
        if self.norm:
            self.min = np.min(sim_array)
            self.max = np.max(sim_array)
            sim_array = (sim_array - self.min) / (self.max - self.min)

        sim_array[sim_array >= 1] = 1 - 10e-4
        sim_array[sim_array <= 0] = 10e-4
        self.bmm_model.fit(sim_array)
        self.bmm_model.create_lookup(1)

    def get_posterior(self, sim_t):
        '''
        out_t_free: detached tensor
        '''
        if self.norm:
            sim_t = (sim_t - self.min) / (self.max - self.min)
        w_unk_posterior = self.compute_probabilities_batch(sim_t, 1)
        w_k_posterior = 1 - w_unk_posterior
        return w_k_posterior, w_unk_posterior



# def kl_divergence(p, q):
#     """Compute KL divergence D(P || Q) element-wise."""
#     p = p + 1e-5  # To avoid log(0)
#     q = q + 1e-5
#     return torch.sum(p * torch.log(p / q), dim=-1)


# def pairwise_kl_divergence(batch):
#     """

#     # Take log of the batch to use in KL divergence calculation
#     """
#     kl = 0.
#     n = 0
#     for i, x in enumerate(batch):
#         for j, y in enumerate(batch):
#             if i != j:
#                 kl += F.kl_div(x.log(), y, reduction='none').sum(dim=-1)
#                 n += 1
#     return kl/n


