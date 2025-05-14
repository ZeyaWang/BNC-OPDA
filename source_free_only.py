import numpy as np
import pandas as pd
import torch
from data import *
from net import *
from lib import *
import datetime
from tqdm import tqdm
from torch import optim
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from easydl import *
from mixture import BayesianGaussianMixtureMerge, nmi
import sys
from sklearn.metrics import silhouette_samples, silhouette_score
import yaml
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import pickle as pk
import math

cudnn.benchmark = True
cudnn.deterministic = True
torch.multiprocessing.set_sharing_strategy('file_system')

seed_everything()

if args.gpus < 1:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    gpu_ids = []
    output_device = torch.device('cpu')
else:
    # gpu_ids = select_GPUs(args.gpus)
    # output_device = gpu_ids[0]
    #output_device = torch.device('cuda:{}'.format(args.gid))
    output_device = torch.device('cuda')
#print('==========, device, ', output_device)
Cluster = BayesianGaussianMixtureMerge(
    n_components=args.max_k,
    n_init=5,
    weight_concentration_prior=args.alpha / args.max_k,
    weight_concentration_prior_type='dirichlet_process',
    init_params='kmeans_merge',
    covariance_prior=args.covariance_prior * args.bottle_neck_dim * np.identity(
        args.bottle_neck_dim),
    covariance_type='full')



def detect(totalNet, thresh=0.5, score='cos'):
    ### score = 'cos' or 'entropy' respectively corresponding to cosine similarity and entropy score
    feature_list, label_list, pred_logits = [], [], []
    with TrainingModeManager(
            [totalNet.feature_extractor, totalNet.bottle_neck, totalNet.classifier], train=False) as mgr, \
            torch.no_grad():
        for i, (im, label) in enumerate(tqdm(target_test_dl)):
            im = im.to(output_device)
            _, feature, predict_logit = totalNet(im)
            feature_list.append(feature.detach().cpu().numpy())
            pred_logits.append(predict_logit.detach().cpu().numpy())
            label_list.append(label.numpy())
    #print(labels)
    feature = np.concatenate(feature_list, axis=0)
    pred_logits = np.concatenate(pred_logits, axis=0)
    preds = np.argmax(pred_logits, axis=1)
    labels = np.concatenate(label_list, axis=0)
    entropy_scores = entropy_numpy(pred_logits)
    sim_bmm_model = sim_bmm(norm=True)
    init_centers = totalNet.classifier.fc.weight.detach().cpu().numpy()
    #print('===========', init_centers.shape)
    #init_centers = init_centers[:num_src_cls]
    cos = cosine_similarity(feature, init_centers) #* np.linalg.norm(init_centers, axis=1)
    #print('=================cos================', cos.max(1))
    cos_max = cos.max(1)
    cos_argmax = cos.argmax(1)
    if score == 'cos':
        #ys = np.concatenate(label_list, axis=0)
        #plotpy2(feature, ys)
        sim_bmm_model.bmm_fit(1-cos_max)
        #sim_bmm_model.bmm_fit(cos_max[cos_argmax < num_src_cls])
        #w_k_posterior = np.zeros(cos_max.shape)
        #w_k_posterior_tmp, _ = sim_bmm_model.get_posterior(cos_max[cos_argmax < num_src_cls])
        # w_k_posterior_tmp, _ = sim_bmm_model.get_posterior(cos_max)
        # w_k_posterior[cos_argmax < num_src_cls] = w_k_posterior_tmp
        w_k_posterior, _ = sim_bmm_model.get_posterior(1-cos_max)
    else:
        sim_bmm_model.bmm_fit(entropy_scores)
        w_k_posterior, _ = sim_bmm_model.get_posterior(entropy_scores)
    # label_ = np.copy(preds)
    label_ = np.copy(cos_argmax)
    label_[w_k_posterior <= thresh] = 100
    return label_, cos_max, w_k_posterior, cos_argmax, feature, init_centers, sim_bmm_model



def clustering(tgt_embedding, tgt_member, predict_src, ttype='OPDA'):
    #report before perfomance
    print('========= before clustering ==========')
    merge_perf(tgt_member, predict_src, ncls=num_src_cls)
    tgt_predict = merge_cluster(Cluster, tgt_embedding, tgt_member, predict_src, plot=False, num_src_cls=num_src_cls)
    hos_v, acc_v, nmi_v, k_acc, uk_nmi, rec, prec = merge_perf_only(tgt_member, tgt_predict, ncls=num_src_cls)
    metrics = {'cl_hos': hos_v, 'cl_acc_test': acc_v, 'cl_nmi': nmi_v.item(), 'cl_k_acc': k_acc.item(), 'cl_uk_nmi': uk_nmi.item(), 'cl_rec': rec.item(), 'cl_prec': prec.item()}

    return tgt_predict, metrics

# def generate_memory(tgt_predict, embedding):
#     tgt_predict_post, tgt_match = post_match(tgt_predict)
#     memory = Memory(len(np.unique(tgt_predict_post)), feat_dim=args.bottle_neck_dim)
#     memory.init(embedding, tgt_predict_post, output_device)
#     return memory, tgt_predict_post, tgt_match




now = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
subdir = f'{args.balance}_{args.lr}_{args.lr_scale}_{args.iter_factor}_{args.lambdav}_{args.max_k}_{args.KK}_{args.covariance_prior}_{args.score}_{args.classifier}'
#log_dir = f'exp_{args.dataset}/{source_domain_name}_{target_domain_name}/{args.balance}_{args.interval}_{args.lambdav}_{args.lr}/{now}'
log_dir = f'exponly_{args.dataset}/{args.target_type}/{source_domain_name}_{target_domain_name}/{subdir}/{now}'
logger = SummaryWriter(log_dir)
old_stdout = sys.stdout
log_file = open(f'{log_dir}/message.log', 'w')
sys.stdout = log_file
with open(join(log_dir, 'config.yaml'), 'w') as f:
    f.write(yaml.dump(args))

rpath = '/home/zwa281/UDA'
#rpath = '/storage'



pretrain_file = os.path.join(rpath, 'UDA/pretrained_source_str/{}/{}_{}.pkl'.format(args.target_type, args.dataset, domain_map[args.dataset][args.source]))

totalNet = SimpleNet(num_cls=num_src_cls, output_device=output_device,
                    bottle_neck_dim=args.bottle_neck_dim, base_model=args.base_model)
totalNet.load_model(pretrain_file, load=('feature_extractor', 'bottleneck', 'classifier'))
#optSets = OptSets(totalNet, args.lr, args.total_epoch * len(target_train_dl) * args.interval, lr_scale=args.lr_scale)
global_step = 0

metrics_epoch = {}
best_hos, best_acc = 0., 0.

if args.thresh is not None:
    print('##########Threshold is set to {}##########'.format(args.thresh))
    threshs = [args.thresh]
else:
    if args.dataset != 'visda':
        if args.target_type == 'PDA':
            threshs = [0.5, 0.6, 0.7, 0.8, 0.9]
        else:
            threshs = [0.4, 0.45, 0.5, 0.55, 0.6]
    else:
        threshs = [0.5]

d_result = {}
dt = {}
for it, t in enumerate(threshs):
    predict_y, cos_max, w_k_posterior, arg_y, feature, init_centers, sim_bmm_model = detect(totalNet, thresh=t, score=args.score)
    if (len(predict_y[predict_y < 100]) == 0) or (len(predict_y[predict_y == 100]) == 0):
        continue
    if it == 0:
        tgt_embedding, tgt_member = gen_cluster_input(totalNet, target_test_dl, output_device)
    tgt_predict, metrics = clustering(tgt_embedding, tgt_member, predict_y, args.target_type)
    metrics['pre_cluster_tgt_predict'] = predict_y
    metrics['pre_cluster_embedding'] = tgt_embedding
    metrics['post_cluster_tgt_predict'] = tgt_predict
    metrics['post_cluster_tgt_member'] = tgt_member
    sil = silhouette_score(tgt_embedding, tgt_predict)
    d_result[sil] = (t, tgt_predict, metrics)
    dt[t] = sil
max_sil = max(d_result.keys())
t, tgt_predict, metrics = d_result[max_sil]
print('epoch_id {}:; best thresh: {}; silhouette scores: {}'.format(0, t, dt))

# memory, tgt_predict_post, tgt_match = generate_memory(tgt_predict, tgt_embedding)
# target_train_ds.labels = list(zip([i for i in range(len(target_train_ds.datas))], tgt_predict_post))
# target_train_dl = DataLoader(dataset=target_train_ds, batch_size=args.batch_size, shuffle=True,
#                              num_workers=data_workers, drop_last=True)

# (counters, unknown_test_truth, unknown_test_pred, acc_tests, acc_test, hos,
# nmi_v, unk_nmi, k_acc, tgt_member, tgt_predict) = valid(memory.memory,
#                                                           totalNet,
#                                                           target_test_dl,
#                                                           output_device,
#                                                           source_classes,
#                                                           tgt_match, args.target_type)
metrics['tgt_member'] = tgt_member
metrics['tgt_predict'] = tgt_predict

# metrics = metrics_epoch[4]
# final_metrics = metrics_epoch[9]

mvalue= [[0, metrics['cl_hos'], metrics['cl_acc_test'], metrics['cl_nmi'], metrics['cl_k_acc'], metrics['cl_uk_nmi']]]
best_df = pd.DataFrame(mvalue)


if args.alpha == 1.0:
    outcsv = 'exponly_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.csv'.format(args.target_type, source_domain_name, target_domain_name, args.balance, args.lr, args.lr_scale,
                                                                  args.iter_factor, 0.0, args.max_k, args.KK, args.covariance_prior, args.score, args.classifier, args.batch_size)
else:
    outcsv = 'exponly_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.csv'.format(args.target_type, source_domain_name, target_domain_name, args.balance, args.lr, args.lr_scale,
                                                                  args.iter_factor, args.alpha, args.max_k, args.KK, args.covariance_prior, args.score, args.classifier, args.batch_size)
best_df.to_csv(outcsv)
sys.stdout = old_stdout
log_file.close()

