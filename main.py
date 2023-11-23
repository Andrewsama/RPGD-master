import torch, pickle, time, os
import numpy as np
import math
import dgl
import scipy
from options import parse_args
from torch.utils.data import DataLoader
from model import RPGD, BPRLoss
from data_utils import prepare_dgl_graph, MyDataset
from utils import load_data, load_model, save_model, fix_random_seed_as
from tqdm import tqdm
from collections import defaultdict


class Model():
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')

        self.dataset = load_data(args.data_path)
        val_neg_data = load_data(args.val_neg_path)
        test_neg_data = load_data(args.test_neg_path)


        trainset = MyDataset(self.dataset, 'train')
        valset   = MyDataset(self.dataset, 'val',  val_neg_data)
        testset  = MyDataset(self.dataset, 'test', test_neg_data)
        self.trainloader = DataLoader(
            dataset=trainset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        self.valloader = DataLoader(
            dataset=valset,
            batch_size=args.test_batch_size * 101,
            shuffle=False,
            num_workers=args.num_workers
        )
        self.testloader = DataLoader(
            dataset=testset,
            batch_size=args.test_batch_size * 101,
            shuffle=False,
            num_workers=args.num_workers
        )

        self.graph = prepare_dgl_graph(args, self.dataset).to(self.device)

        self.social_pos = self.graph.edata['type'] == 0
        self.ori_graph = dgl.graph((self.graph.edges()[0][self.social_pos], self.graph.edges()[1][self.social_pos]), num_nodes=self.dataset['userCount'])
        self.ori_graph = dgl.add_self_loop(self.ori_graph)

        self.social_dict, self.avg_degree = self.construct_social_dict(self.dataset)
        self.model = RPGD(args, self.dataset['userCount'], self.dataset['itemCount'], self.dataset['categoryCount'])
        self.model = self.model.to(self.device)
        self.criterion = BPRLoss(args.reg)
        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters()}
        ], lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=args.decay_step,
            gamma=args.decay
        )
        if args.checkpoint:
            load_model(self.model, args.checkpoint, self.optimizer) # None


    def train(self):
        flag = 0
        args = self.args
        len_topk = len(args.topk)
        best_hr, best_ndcg, best_epoch, wait = np.array([0.0] * len_topk), np.array([0.0] * len_topk), np.array([0.0] * len_topk), np.array([0.0] * len_topk)
        early_stop_list = [-1] * len_topk
        start_time = time.time()
        for self.epoch in range(1, args.n_epoch + 1):
            if self.epoch % 30 == 0:
                left = []
                right = []
                rep, user_pool = self.model(self.graph, self.ori_graph)
                rep = rep.to('cpu').detach().numpy()
                for key, value in self.social_dict.items():
                    if len(self.social_dict[key]) <= self.avg_degree:
                        continue
                    similarity = np.array([np.dot(rep[key], rep[nei]) / ((len(self.social_dict[key]) - 1) * (len(self.social_dict[nei]) - 1) + 0.00001) for nei in value])

                    # index_list = sorted(np.argsort(similarity)[:math.ceil(math.log10(len(self.social_dict[key])-self.avg_degree)+1)])[::-1]
                    del_list = []
                    index_list = np.argsort(similarity)
                    for i in index_list:
                        if similarity[i] < 0.001:
                            del_list.append(i)
                        else:
                            break
                    if not del_list:
                        continue
                    del_list.sort(reverse=True)
                    for i in del_list:
                        self.social_dict[self.social_dict[key][i]].remove(key)
                        self.social_dict[key].pop(i)
                for key, value in self.social_dict.items():
                    left.extend([key] * len(value))
                    right.extend(value)
                coo_ma = scipy.sparse.coo_matrix(([1]*len(left), (left,right)), shape=(self.dataset['userCount'],self.dataset['userCount']))
                self.dataset['trust'] = coo_ma
                self.graph = prepare_dgl_graph(args, self.dataset).to(self.device)


            epoch_losses = self.train_one_epoch(self.trainloader, self.graph, self.ori_graph)
            print('epoch {} done! elapsed {:.2f}.s, epoch_losses {}'.format(
                self.epoch, time.time() - start_time, epoch_losses
            ), flush=True)

            hr_list, ndcg_list = self.validate(self.testloader, self.graph, self.ori_graph)

            cur_best = hr_list + ndcg_list > best_hr + best_ndcg
            for i in range(len_topk):
                if cur_best[i]:# and early_stop_list[i] == -1:
                    best_hr[i], best_ndcg[i], best_epoch[i] = hr_list[i], ndcg_list[i], self.epoch
                    wait[i] = 0
                else:
                    wait[i] += 1
            for i in range(len(args.topk)):
                print('+ epoch {} tested, elapsed {:.2f}s, N@{}: {:.4f}, R@{}: {:.4f}'.format(
                    self.epoch, time.time() - start_time, args.topk[i], ndcg_list[i], args.topk[i], hr_list[i]
            ), flush=True)

            if args.model_dir and cur_best:
                desc = f'{args.dataset}_hid_{args.n_hid}_layer_{args.n_layers}_mem_{args.mem_size}_' + \
                       f'lr_{args.lr}_reg_{args.reg}_decay_{args.decay}_step_{args.decay_step}_batch_{args.batch_size}'
                perf = '' # f'N/R_{ndcg:.4f}/{hr:.4f}'
                fname = f'{args.desc}_{desc}_{perf}.pth'
                save_model(self.model, os.path.join(args.model_dir, fname), self.optimizer)

            for i in range(len_topk):
                if wait[i] >= args.patience and early_stop_list[i] == -1:
                    flag += 1
                    early_stop_list[i] = self.epoch
            if flag == len_topk:
                break
        for i in range(len_topk):
            print(f'Best N@{args.topk[i]} {best_ndcg[i]:.4f}, R@{args.topk[i]} {best_hr[i]:.4f}', flush=True, end = ' ')
            if early_stop_list[i] != -1:
                print(f'Early stop top{args.topk[i]} at epoch {early_stop_list[i]}, best epoch {best_epoch[i]}')
            else:
                print()
        print('The number of rest edges:',self.dataset['trust'].nnz)


    def train_one_epoch(self, dataloader, graph, ori_graph):
        self.model.train()

        epoch_losses = [0] * 2
        dataloader.dataset.neg_sample()
        tqdm_dataloader = tqdm(dataloader)

        for iteration, batch in enumerate(tqdm_dataloader):
            user_idx, pos_idx, neg_idx = batch
            usr1 = torch.randint(high=self.dataset['userCount'], size=(len(user_idx),))
            usr2 = torch.randint(high=self.dataset['userCount'], size=(len(user_idx),))

            rep, user_pool, sfmLoss, salLoss = self.model(graph, ori_graph, usr1, usr2)
            user = rep[user_idx] + user_pool[user_idx]
            pos  = rep[self.model.n_user + pos_idx]
            neg  = rep[self.model.n_user + neg_idx]
            pos_preds = self.model.predict(user, pos)
            neg_preds = self.model.predict(user, neg)




            loss, losses = self.criterion(pos_preds, neg_preds, user, pos, neg)
            print('BPR_LOSS', losses[0])
            print('REG_LOSS', losses[1])
            print('SFM_LOSS', sfmLoss)
            print('SAL_LOSS', salLoss)

            loss += sfmLoss
            loss += salLoss
            losses.append(sfmLoss.item())
            losses.append(salLoss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_losses = [x + y for x, y in zip(epoch_losses, losses)]
            tqdm_dataloader.set_description('Epoch {}, loss: {:.4f}'.format(self.epoch, loss.item()))


        if self.scheduler is not None:
            self.scheduler.step()

        epoch_losses = [sum(epoch_losses)] + epoch_losses
        return epoch_losses


    def calc_hr_and_ndcg(self, preds  , topk):
        preds = preds.reshape(-1, 101)
        labels = torch.zeros_like(preds)
        labels[:, 0] = 1
        hrs_list = []
        ndcgs_list = []
        for i in range(len(topk)):
            _, indices = preds.topk(topk[i])
            hits = labels.gather(1, indices)
            hrs = hits.sum(1).tolist()
            weights = 1 / torch.log2(torch.arange(2, 2 + topk[i]).float()).to(hits.device)
            ndcgs = (hits * weights).sum(1).tolist()
            hrs_list.append(hrs)
            ndcgs_list.append(ndcgs)
        return hrs_list, ndcgs_list

    def construct_social_dict(self, dataset):
        uids, fids = dataset['trust'].nonzero()
        social_dict = defaultdict(list)
        mid_dgree_list = []
        for l,r in zip(uids, fids):
            social_dict[l].append(r)
        for i in social_dict.values():
            mid_dgree_list.append(len(i))
        return social_dict, sorted(mid_dgree_list)[math.ceil(1/4 * dataset['userCount'])]


    def validate(self, dataloader, graph, ori_graph):
        self.model.eval()
        hrs_list, ndcgs_list = [[] for _ in range(len(self.args.topk))], [[] for _ in range(len(self.args.topk))]

        with torch.no_grad():
            tqdm_dataloader = tqdm(dataloader)
            for iteration, batch in enumerate(tqdm_dataloader, start=1):
                user_idx, item_idx = batch

                rep, user_pool = self.model(graph, ori_graph)
                user = rep[user_idx] + user_pool[user_idx]
                item  = rep[self.model.n_user + item_idx]
                preds = self.model.predict(user, item)

                preds_hrs, preds_ndcgs = self.calc_hr_and_ndcg(preds, self.args.topk)
                for i in range(len(preds_hrs)):
                    hrs_list[i] += preds_hrs[i]
                    ndcgs_list[i] += preds_ndcgs[i]
        # return np.mean(hrs), np.mean(ndcgs)
        return np.array([np.mean(hrs) for hrs in hrs_list]), np.array([np.mean(ndcgs) for ndcgs in ndcgs_list])


    def test(self):
        load_model(self.model, args.checkpoint)
        self.model.eval()

        with torch.no_grad():
            rep, user_pool = self.model(self.graph)

            """ Save embeddings """
            user_emb = (rep[:self.model.n_user] + user_pool).cpu().numpy()
            # user_emb = (rep[:self.model.n_user]).cpu().numpy()
            item_emb = rep[self.model.n_user: self.model.n_user + self.model.n_item].cpu().numpy()
            with open(f'RPGD-{self.args.dataset}-embeds.pkl', 'wb') as f:
                pickle.dump({'user_embed': user_emb, 'item_embed': item_emb}, f)

            """ Save results """
            tqdm_dataloader = tqdm(self.testloader)
            uids, hrs, ndcgs = [], [], []
            for iteration, batch in enumerate(tqdm_dataloader, start=1):
                user_idx, item_idx = batch

                user = rep[user_idx] + user_pool[user_idx]
                item  = rep[self.model.n_user + item_idx]
                preds = self.model.predict(user, item)

                preds_hrs, preds_ndcgs = self.calc_hr_and_ndcg(preds, self.args.topk)
                hrs += preds_hrs
                ndcgs += preds_ndcgs
                uids += user_idx[::101].tolist()

            with open(f'RPGD-{self.args.dataset}-test.pkl', 'wb') as f:
                pickle.dump({uid: (hr, ndcg) for uid, hr, ndcg in zip(uids, hrs, ndcgs)}, f)


if __name__ == "__main__":
    args = parse_args()
    fix_random_seed_as(args.seed)

    app = Model(args)

    app.train()
