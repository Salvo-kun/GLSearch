from options import opt
from torch.utils.data import DataLoader
from torch.optim import Adam
from data import BatchData, load_dataset_list
from models.gl_search import GLSearch

def train():
    datasets = load_dataset_list(opt.dataset_list)
    
    num_node_feat = datasets[0].num_node_features
    feat_map = {}
    tot_num_train_pairs = sum([x.gid1gid2_list.shape[0] for x in datasets])
    
    model = GLSearch(num_node_feat, feat_map, tot_num_train_pairs).to(opt.device)
    optimizer = Adam(model.parameters(), lr=opt.lr)

    num_iters_total = 0
    num_iters_total_limit = 0

    for curriculum_id, curriculum_dataset in enumerate(datasets):
        num_iters_total_limit += opt.dataset_list[curriculum_id][1]
        data_loader = DataLoader(curriculum_dataset, batch_size=opt.batch_size, shuffle=opt.shuffle_input)
        num_iters = 0

        for iter, data in enumerate(data_loader):
            if num_iters == opt.only_iters_for_debug or (num_iters_total_limit and num_iters_total == num_iters_total_limit):
                return

            batch_data = BatchData(data, curriculum_dataset.dataset)
            
            model.train()
            model.zero_grad()
            loss = model(curriculum_id, iter, batch_data)

            if opt.retain_graph:
                loss.backward(retain_graph=True)
            else:
                loss.backward()
                
            optimizer.step()

            loss = loss.item() if loss is not None else 0.0
            total_loss += loss
            num_iters += 1
            num_iters_total += 1

if __name__ == '__main__':
    train()