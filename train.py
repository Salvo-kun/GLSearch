import torch
from options import opt
from torch.utils.data import DataLoader
from torch.optim import Adam
from data import BatchData, load_dataset_list
from models.gl_search import GLSearch, ModelParams
import logging

torch.autograd.set_detect_anomaly(True)


def train():
    datasets = load_dataset_list(opt.dataset_list)
    num_node_feat = datasets[0].num_node_features
    feat_map = {}

    model = GLSearch(num_node_feat, feat_map).to(opt.device)
    optimizer = Adam(model.parameters(), lr=opt.lr)

    num_iters_total = 0
    num_iters_total_limit = 0

    for curriculum_id, curriculum_dataset in enumerate(datasets):
        num_iters_total_limit += curriculum_dataset.num_iter
        data_loader = DataLoader(curriculum_dataset, batch_size=opt.batch_size, shuffle=opt.shuffle_input)
        num_iters = 0
        total_loss = 0.0

        for data in data_loader:
            if num_iters == opt.only_iters_for_debug or (num_iters_total_limit and num_iters_total == num_iters_total_limit):
                return

            batch_data = BatchData(data, curriculum_dataset.dataset)
            print(f'CID: {curriculum_id} \t Iter: {num_iters}')
            # print(f'Indices: {batch_data.merge_data["ind_list"]}')
            # print(f'Dataset: {curriculum_dataset.dataset}')
            # print(f'Ins: {batch_data.merge_data["merge"].x}')

            model.train()
            model.zero_grad()
            loss = model(ModelParams(curriculum_id, num_iters_total, batch_data))

            if loss is not None:
                if opt.retain_graph:
                    loss.backward(retain_graph=True)
                else:
                    loss.backward()

                optimizer.step()

            total_loss += loss.item() if loss is not None else 0.0
            num_iters += 1
            num_iters_total += 1

if __name__ == '__main__':
    train()