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
    gpu_ids = select_GPUs(args.gpus)
    output_device = gpu_ids[0]

Cluster = BayesianGaussianMixtureMerge(
    n_components=args.max_k,
    n_init=5,
    weight_concentration_prior=args.alpha / args.max_k,
    weight_concentration_prior_type='dirichlet_process',
    init_params='kmeans_merge',
    covariance_prior=args.covariance_prior * args.bottle_neck_dim * np.identity(
        args.bottle_neck_dim),
    covariance_type='full')


class OptSets():
    def __init__(self, totalNet, lr, min_step, lr_scale=10.0):
        scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=min_step)
        self.optimizer_extractor = OptimWithSheduler(
            optim.SGD(totalNet.feature_extractor.parameters(), lr=lr / lr_scare,
                      weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True), scheduler)

        self.optimizer_linear = OptimWithSheduler(
            optim.SGD(totalNet.classifier.parameters(), lr=lr, weight_decay=args.weight_decay,
                      momentum=args.momentum, nesterov=True),
            scheduler)

        self.optimizer_bottleneck = OptimWithSheduler(
            optim.SGD(totalNet.bottle_neck.parameters(), lr=lr, weight_decay=args.weight_decay,
                      momentum=args.momentum, nesterov=True),
            scheduler)

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
            pred_logits.append(pred_logit.detach().cpu().numpy())
            label_list.append(label.numpy())
    #print(labels)
    feature = np.concatenate(feature_list, axis=0)
    pred_logits = np.concatenate(pred_logits, axis=0)
    entropy_scores = entropy_pytorch(pred_logits)
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
    label_ = np.copy(cos_argmax)
    label_[w_k_posterior <= thresh] = 100
    return label_, cos_max, w_k_posterior, cos_argmax, feature, init_centers, sim_bmm_model



def clustering(tgt_embedding, tgt_member, predict_src):
    tgt_predict = merge_cluster(Cluster, tgt_embedding, tgt_member, predict_src, plot=False, num_src_cls=num_src_cls)
    nmi_v, k_acc, uk_nmi, rec, prec = merge_perf(tgt_member, tgt_predict, ncls=num_src_cls)
    metrics = {'nmi': nmi_v.item(), 'k_acc': k_acc.item(), 'uk_nmi': uk_nmi.item(), 'rec': rec.item(), 'prec': prec.item()}
    return tgt_predict, metrics

def generate_memory(tgt_predict, embedding):
    tgt_predict_post, tgt_match = post_match(tgt_predict)
    memory = Memory(len(np.unique(tgt_predict_post)), feat_dim=args.bottle_neck_dim)
    memory.init(embedding, tgt_predict_post, output_device)
    return memory, tgt_predict_post, tgt_match

def train(ClustNet, train_ds, memory, optSets, epoch_step, global_step, total_step, interval, classifier=False):
    # interval: the repeated time between two clustering
    num_sample = len(train_ds)
    #score_bank = torch.full((num_sample, num_src_cls), 1.0 / num_src_cls).to(output_device)
    score_bank = torch.randn(num_sample, num_src_cls).to(output_device)

    loader = DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True,
                                 num_workers=data_workers, drop_last=False)
    fea_bank = torch.randn(num_sample, args.bottle_neck_dim)
    ClustNet.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            #inputs, (indx, _) = iter_test.next()
            inputs, (indx, _) = next(iter_test)
            # labels = data[1]
            _, output, outputs = ClustNet(inputs.to(output_device))
            output_norm = F.normalize(output)
            outputs = nn.Softmax(-1)(outputs)
            # if args.sharp:
            #     outputs = outputs**2 / ((outputs**2).sum(dim=0))
            fea_bank[indx] = output_norm.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone()
            #print(indx)
        #torch.set_printoptions(threshold=torch.inf)


    ClustNet.train()
    mloss_total_t, closs_total_t, loss_total_t = [], [], []
    for t in range(args.interval):
        iters = tqdm(loader, desc=f'epoch {epoch_step} args.interval {t} ', total=len(loader))
        mloss_total, closs_total, loss_total =  0., 0., 0.
        for i, (im, (idx, plabel)) in enumerate(iters):
            idx = idx.to(output_device)  # pseudolabel with a value between 0 and 1
            plabel = plabel.to(output_device)
            im = im.to(output_device)
            embedding, feature, pred_logit = ClustNet(im)
            softmax_out = nn.Softmax(dim=1)(pred_logit)
            with torch.no_grad():
                output_f = F.normalize(feature).cpu().detach().clone()
                fea_bank[idx] = output_f.detach().clone().cpu()
                score_bank[idx] = softmax_out.detach().clone()
                #print('++',score_bank)
                distance = output_f @ fea_bank.T
                _, idx_near = torch.topk(distance, dim=-1, largest=True, k=args.KK + 1)
                idx_near = idx_near[:, 1:]  # batch x K
                score_near = score_bank[idx_near]  # batch x K x C
                #print('++===',idx_near, score_near)
            softmax_out_un = softmax_out.unsqueeze(1).expand(-1, args.KK, -1)  # batch x K x C
            #print(softmax_out_un, score_near)
            closs = torch.mean(
                (F.kl_div(softmax_out_un.log(), score_near, reduction="none").sum(-1)).sum(1)
            )
            # neg_pred = pairwise_kl_divergence(softmax_out)
            # closs -= neg_pred * args.lambdav


            mloss = memory.forward(feature, plabel)
            mloss = mloss * ExpWeight(global_step, max_iter=total_step*len(loader)*args.interval)
            loss = args.balance*closs  + mloss
            #print('==============', closs.item())
            closs_total += closs.item()
            mloss_total += mloss.item()
            loss_total += loss.item()
            optims = [optSets.optimizer_extractor, optSets.optimizer_bottleneck]
            if classifier == True:
                optims.append(optSets.optimizer_linear)
            with OptimizerManager(optims):
                    #[optSets.optimizer_extractor]):
                    #[optSets.optimizer_extractor, optSets.optimizer_bottleneck, optSets.optimizer_classifier]):
                loss.backward()
            global_step += 1
        tqdm.write(f'EPOCH {epoch_step:03d}: args.interval {t:03d}, closs={closs_total:.4f}, mloss={mloss_total:.4f}, loss={loss_total:.4f}')
        closs_total_t.append(closs_total)
        mloss_total_t.append(mloss_total)
        loss_total_t.append(loss_total)
    return global_step, closs_total_t, mloss_total_t, loss_total_t



now = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
log_dir = f'exp_{args.dataset}/{source_domain_name}_{target_domain_name}/{args.balance}_{args.interval}_{args.lambdav}_{args.lr}/{now}'
logger = SummaryWriter(log_dir)
old_stdout = sys.stdout
log_file = open(f'{log_dir}/message.log', 'w')
sys.stdout = log_file
with open(join(log_dir, 'config.yaml'), 'w') as f:
    f.write(yaml.dump(args))

rpath = '/home/zwa281/UDA'
#rpath = '/storage'



pretrain_file = os.path.join(rpath, 'UDA/pretrained_source/{}_{}.pkl'.format(args.dataset, domain_map[args.dataset][args.source]))

totalNet = SimpleNet(num_cls=num_src_cls, output_device=output_device,
                    bottle_neck_dim=args.bottle_neck_dim, base_model=args.base_model)
totalNet.load_model(pretrain_file, load=('feature_extractor', 'bottleneck', 'classifier'))
optSets = OptSets(totalNet, args.lr, args.total_epoch * len(target_train_dl) * args.interval)
global_step = 0

metrics_epoch = {}
best_hos = 0.
if args.dataset != 'visda':
    threshs = [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]
else:
    threshs = [0.5]

for epoch_id in tqdm(range(args.total_epoch), desc="Processing"):
    d_result = {}
    dt = {}
    for it, t in enumerate(threshs):
        predict_y, cos_max, w_k_posterior, arg_y, feature, init_centers, sim_bmm_model = detect(totalNet, thresh=t)
        if (len(predict_y[predict_y < 100]) == 0) or (len(predict_y[predict_y == 100]) == 0):
            continue
        if it == 0:
            tgt_embedding, tgt_member = gen_cluster_input(totalNet, target_test_dl, output_device)
        tgt_predict, metrics = clustering(tgt_embedding, tgt_member, predict_y)

        sil = silhouette_score(tgt_embedding, tgt_predict)
        d_result[sil] = (t, tgt_predict, metrics)
        dt[t] = sil
    max_sil = max(d_result.keys())
    t, tgt_predict, metrics = d_result[max_sil]
    print('epoch_id {}:; best thresh: {}; silhouette scores: {}'.format(epoch_id, t, dt))

    memory, tgt_predict_post, tgt_match = generate_memory(tgt_predict, tgt_embedding)
    target_train_ds.labels = list(zip([i for i in range(len(target_train_ds.datas))], tgt_predict_post))
    target_train_dl = DataLoader(dataset=target_train_ds, batch_size=args.batch_size, shuffle=True,
                                 num_workers=data_workers, drop_last=True)
    counters, unknown_test_truth, unknown_test_pred, acc_tests, acc_test, hos = valid(memory.memory,
                                                                                      totalNet,
                                                                                      target_test_dl,
                                                                                      output_device,
                                                                                      log_dir,
                                                                                      source_classes,
                                                                                      tgt_match)
    metrics['hos'] = hos.item()
    metrics['acc_tests'] = acc_tests
    metrics['acc_test'] = acc_test.item()
    if hos > best_hos:
        best_hos = hos
        best_metrics = metrics
    global_step, closs_total_t, mloss_total_t, loss_total_t = train(totalNet, target_train_ds, memory, optSets, epoch_id, global_step, args.total_epoch, args.interval)
    metrics['global_step'] = global_step
    metrics['closs_total'] = closs_total_t
    metrics['mloss_total'] = mloss_total_t
    metrics['loss_total'] = loss_total_t
    metrics_epoch[epoch_id] = metrics

with open(f'{log_dir}/output.pkl', 'wb') as file:
    pk.dump([metrics_epoch, best_metrics], file)


best_df = pd.DataFrame([[best_metrics['hos'], best_metrics['acc_test'], best_metrics['nmi'], best_metrics['k_acc'], best_metrics['uk_nmi']]+list(best_metrics['acc_tests'].values())] )
best_df.to_csv('exp_{}_{}_{}_{}_{}_{}.csv'.format(source_domain_name, target_domain_name, args.balance, args.interval, args.lambdav, args.lr), index=False, header=False)

sys.stdout = old_stdout
log_file.close()

