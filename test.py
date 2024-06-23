import os
import argparse
import pickle
import yaml
import torch
from glob import glob
from tqdm.auto import tqdm
from easydict import EasyDict
from torch_geometric.data import DataLoader

from models.network import *
from utils.datasets import *
from utils.transforms import *
from utils.misc import *


def num_confs(num:str):
    if num.endswith('x'):
        return lambda x:x*int(num[:-1])
    elif int(num) > 0: 
        return lambda x:int(num)
    else:
        raise ValueError()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default="logs/qm9_para_2024_04_16__10_17_44 1e-3 200 500/checkpoints/500.pt", help='path for loading the checkpoint')
    parser.add_argument('--save_traj', action='store_true', default=True,
                    help='whether store the whole trajectory for sampling')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--tag', type=str, default='')   # resuem or not or other tag
    parser.add_argument('--num_confs', type=num_confs, default=num_confs('2x'))
    parser.add_argument('--test_set', type=str, default=None)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=200)
    parser.add_argument('--out_dir', type=str, default="out_dir")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--clip', type=float, default=1000.0)
    parser.add_argument('--n_steps', type=int, default=5000,
                    help='sampling num steps; for DSM framework, this means num steps for each noise scale')
    parser.add_argument('--sampling_steps', type=int, default=5000,
                        help='sampling steps for DDIM')
    # Parameters for DDPM
    parser.add_argument('--sampling_type', type=str, default='ddpm_noisy',
                    help='generalized, ddpm_noisy, ld: sampling method for DDIM, DDPM or Langevin Dynamics')
    parser.add_argument('--eta', type=float, default=1.0,
                    help='weight for DDIM and DDPM: 0->DDIM, 1->DDPM')
    args = parser.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.ckpt)
    config_path = glob(os.path.join(os.path.dirname(os.path.dirname(args.ckpt)), '*.yml'))[0]
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))              # seed, edge_order, test_set path
    used_seed(config.train.seed)
    log_dir = os.path.dirname(os.path.dirname(args.ckpt))

    # Logging
    output_dir = get_new_log_dir(log_dir, 'sample', tag=args.tag)
    logger = get_logger('test', output_dir)
    logger.info(args)

    # Datasets and loaders
    logger.info('Loading datasets...')
    transforms = Compose([
        CountNodesPerGraph(),
        AddHigherOrderEdges(order=config.model.edge_order), # Offline edge augmentation
    ])     # Transform 
    if args.test_set is None:
        test_set = PackedConformationDataset(config.dataset.test, transform=transforms)
    else:
        test_set = PackedConformationDataset(args.test_set, transform=transforms)

    # Model
    logger.info('Loading model...')
    model = get_model(ckpt['config'].model).to(args.device)
    model.load_state_dict(ckpt['model'])

    test_set_selected = []
    for i, data in enumerate(test_set):
        if not (args.start_idx <= i < args.end_idx): continue
        test_set_selected.append(data)

    done_files = set()
    results = []
    if args.resume is not None:
        with open(args.resume, 'rb') as f:
            results = pickle.load(f)
        for data in results:
            done_files.add(data.smiles)
    
    data_total = []
    data_gen_num_list = []
    data_nodes_per_graph = []
    idx_cur_print = 0
    flag_start = 0
    for i, data in enumerate(tqdm(test_set_selected)):
        if data.smiles in done_files:
            logger.info('Molecule#%d is already done.' % i)
            continue

        if flag_start == 0:
            idx_cur_print = i
            flag_start = 1

        num_refs = data.pos_ref.size(0) // data.num_nodes
        num_samples = args.num_confs(num_refs)
        
        data_input = data.clone()
        data_input['pos_ref'] = None     # [num_refs*num_nodes,3]
        # test_batch = repeat_data(data_input, num_samples).to(args.device)
        data_input['mol_index'] = str(i)
        test_batch = repeat_data(data_input, num_samples)
        data_total += test_batch
        data_gen_num_list.append(num_samples)
        data_nodes_per_graph.append(data.num_nodes_per_graph.item())
        
    test_loader = DataLoader(data_total, config.train.batch_size, shuffle=False)
    
    idx_cur = 0
    last_nosave_pos_gen = []
    last_nosave_pos_gen_traj = []
    clip_local = None
    for _ in range(2):  # Maximum number of retry
        try:
            for j, batch in enumerate(tqdm(test_loader, desc="Test")):
                batch.to(args.device)
                pos_init = torch.randn(batch.num_nodes, 3).to(args.device)
                pos_gen_batch, pos_gen_traj_batch = model.langevin_dynamics_sample(
                    atom_type=batch.atom_type,
                    pos_init=pos_init,
                    bond_index=batch.edge_index,
                    bond_type=batch.edge_type,
                    batch=batch.batch,
                    num_graphs=batch.num_graphs,
                    extend_order=False, # Done in transforms.
                    extend_radius=True,
                    n_steps=args.n_steps,   # max 5000
                    sampling_steps=args.sampling_steps,
                    step_lr=1e-6,
                    w_global=args.w_global,
                    global_start_sigma=args.global_start_sigma,
                    clip=args.clip,
                    clip_local=clip_local,
                    sampling_type=args.sampling_type,
                    eta=args.eta,
                    angle_edge_index=batch.angle_edge_index,
                    torsion_edge_index=batch.torsion_edge_index
                )
                
                if len(last_nosave_pos_gen) == 0:
                    num_cur = batch.batch.max()+1
                    pos_gen_temp = pos_gen_batch
                    pos_gen_traj_temp = torch.stack(pos_gen_traj_batch)
                else:
                    # num_cur = batch.batch.max()+1 + len(last_nosave_pos_gen)/data_nodes_per_graph[idx_cur]
                    num_cur += batch.batch.max()+1
                    pos_gen_temp = torch.cat((last_nosave_pos_gen, pos_gen_batch), 0)
                    pos_gen_traj_temp = torch.cat((last_nosave_pos_gen_traj, torch.stack(pos_gen_traj_batch)), 1)
                    
                for i in range(idx_cur, len(test_set_selected)-idx_cur_print):
                    if num_cur == 0 or num_cur < data_gen_num_list[i]:
                        last_nosave_pos_gen = pos_gen_temp
                        last_nosave_pos_gen_traj = pos_gen_traj_temp
                        idx_cur = i
                        break
                    else:
                        pos_gen = pos_gen_temp[:data_gen_num_list[i]*data_nodes_per_graph[i], :]
                        pos_gen_temp = pos_gen_temp[data_gen_num_list[i]*data_nodes_per_graph[i]:, :]
                        pos_gen_traj = pos_gen_traj_temp[:,:data_gen_num_list[i]*data_nodes_per_graph[i], :]
                        pos_gen_traj_temp = pos_gen_traj_temp[:, data_gen_num_list[i]*data_nodes_per_graph[i]:, :]
                        
                        pos_gen = pos_gen.cpu()
                        data = test_set_selected[idx_cur_print+i]
                        if args.save_traj:
                            # data.pos_gen = torch.stack(pos_gen_traj)
                            data.pos_gen = pos_gen_traj
                        else:
                            data.pos_gen = pos_gen
                        data.num_pos_gen = torch.tensor(data_gen_num_list[i])
                        results.append(data)
                        done_files.add(data.smiles)
                        
                        save_path = os.path.join(output_dir, 'samples_%d.pkl' % (idx_cur_print+i))
                        logger.info('Saving samples to: %s' % save_path)
                        with open(save_path, 'wb') as f:
                            # pickle.dump(results, f)
                            pickle.dump(data, f)
                        
                        num_cur -= data_gen_num_list[i]
                        
            break   # No errors occured, break the retry loop
            
        except FloatingPointError:
            clip_local = 20
            logger.warning('Retrying with local clipping.')

    save_path = os.path.join(output_dir, 'samples_all.pkl')
    logger.info('Saving samples to: %s' % save_path)

    def get_mol_key(data):
        for i, d in enumerate(test_set_selected):
            if d.smiles == data.smiles:
                return i
        return -1
    results.sort(key=get_mol_key)

    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
        
    