import os
from datetime import datetime
import numpy as np
import torch
from torch import nn
import pickle
import tqdm
import pathlib
import json
import pandas as pd
from sklearn.neighbors import KDTree

from torch.utils.tensorboard import SummaryWriter

from eval.evaluate import evaluate, print_eval_stats
from misc.utils import MinkLocParams, get_datetime
from models.loss import make_loss
from models.model_factory import model_factory

VERBOSE = False


def load_geo_coordinates(dataset_folder):
    """加载所有session的地理坐标数据"""
    base_path = "/home/wzj/pan2/Chilean_Underground_Mine_Dataset_Many_Times/"
    runs_folder = "chilean/"

    geo_coords = {}  # key: filename, value: (northing, easting)

    print("正在加载地理坐标数据...")
    # 遍历所有可能的session
    loaded_sessions = 0
    for session in range(100, 210):  # 涵盖训练和测试session
        session_str = str(session)
        csv_path = f"{base_path}{runs_folder}{session_str}/pointcloud_locations_20m_10overlap.csv"

        try:
            df = pd.read_csv(csv_path)
            session_count = 0
            for _, row in df.iterrows():
                # 构建文件路径，与数据集中的格式保持一致
                filename = f"chilean/{session_str}/pointcloud_20m_10overlap/{row['timestamp']}.bin"
                geo_coords[filename] = (row['northing'], row['easting'])
                session_count += 1

            if session_count > 0:
                loaded_sessions += 1
                if loaded_sessions <= 5:  # 只打印前5个session的信息
                    print(f"  Session {session}: 加载了 {session_count} 个点云坐标")

        except Exception as e:
            # 某些session可能不存在，跳过
            continue

    print(f"总共从 {loaded_sessions} 个sessions加载了 {len(geo_coords)} 个点云的地理坐标")
    return geo_coords


def verify_dataset_sampling(params):
    """验证数据集采样的地理距离关系"""
    from datasets.dataset_utils import make_datasets

    print(f"\n🔍 === 数据集采样验证 ===")

    # 加载地理坐标
    geo_coords = load_geo_coordinates(params.dataset_folder)

    if len(geo_coords) == 0:
        print("❌ 未能加载任何地理坐标数据，跳过验证")
        return

    # 加载数据集
    datasets = make_datasets(params, debug=False)
    train_dataset = datasets['train']

    print(f"数据集包含 {len(train_dataset.queries)} 个查询")

    # 随机选择一些查询进行验证
    import random
    sample_queries = random.sample(list(train_dataset.queries.keys()), min(10, len(train_dataset.queries)))

    problem_count = 0
    total_checked = 0

    for query_idx in sample_queries[:5]:  # 只验证前5个，避免输出太多
        query_file = train_dataset.queries[query_idx]['query']
        if query_file not in geo_coords:
            print(f"⚠️  查询 {query_idx}: 找不到文件 {query_file} 的地理坐标")
            continue

        query_coord = geo_coords[query_file]
        positives = train_dataset.get_positives_ndx(query_idx)
        negatives = train_dataset.get_negatives_ndx(query_idx)

        print(f"\n📍 查询 {query_idx}: {query_file.split('/')[-1]}")
        print(f"   坐标: ({query_coord[0]:.1f}, {query_coord[1]:.1f})")
        print(f"   正样本数量: {len(positives)}, 负样本数量: {len(negatives)}")

        # 检查前几个正样本的地理距离
        pos_problems = 0
        for i, pos_idx in enumerate(list(positives)[:3]):
            pos_file = train_dataset.queries[pos_idx]['query']
            if pos_file in geo_coords:
                pos_coord = geo_coords[pos_file]
                dist = np.sqrt((query_coord[0] - pos_coord[0]) ** 2 + (query_coord[1] - pos_coord[1]) ** 2)
                status = "✅" if dist < 7 else "❌"
                if dist >= 7:
                    pos_problems += 1
                    problem_count += 1
                print(f"     正样本 {i}: 距离={dist:.1f}米 {status}")
                total_checked += 1

        # 检查前几个负样本的地理距离
        neg_problems = 0
        for i, neg_idx in enumerate(list(negatives)[:3]):
            neg_file = train_dataset.queries[neg_idx]['query']
            if neg_file in geo_coords:
                neg_coord = geo_coords[neg_file]
                dist = np.sqrt((query_coord[0] - neg_coord[0]) ** 2 + (query_coord[1] - neg_coord[1]) ** 2)
                status = "✅" if dist > 35 else "❌"
                if dist <= 35:
                    neg_problems += 1
                    problem_count += 1
                print(f"     负样本 {i}: 距离={dist:.1f}米 {status}")
                total_checked += 1

        if pos_problems > 0 or neg_problems > 0:
            print(f"   ⚠️  此查询存在问题: {pos_problems} 个错误正样本, {neg_problems} 个错误负样本")

    # 总结
    print(f"\n📊 === 验证总结 ===")
    print(f"检查了 {total_checked} 个样本关系")
    print(f"发现 {problem_count} 个问题样本 ({problem_count / total_checked * 100:.1f}%)")

    if problem_count == 0:
        print("✅ 数据集的正负样本地理关系正确")
    elif problem_count < total_checked * 0.1:
        print("⚠️  发现少量问题，可能是数据边界情况")
    else:
        print("❌ 发现大量问题，数据集可能存在系统性错误")


def print_stats(stats, phase):
    if 'num_pairs' in stats:
        # For batch hard contrastive loss
        s = '{} - Mean loss: {:.6f}    Avg. embedding norm: {:.4f}   Pairs per batch (all/non-zero pos/non-zero neg): {:.1f}/{:.1f}/{:.1f}'
        print(s.format(phase, stats['loss'], stats['avg_embedding_norm'], stats['num_pairs'],
                       stats['pos_pairs_above_threshold'], stats['neg_pairs_above_threshold']))
    elif 'num_triplets' in stats:
        # For triplet loss
        s = '{} - Mean loss: {:.6f}    Avg. embedding norm: {:.4f}   Triplets per batch (all/non-zero): {:.1f}/{:.1f}'
        print(s.format(phase, stats['loss'], stats['avg_embedding_norm'], stats['num_triplets'],
                       stats['num_non_zero_triplets']))
    elif 'num_pos' in stats:
        s = '{} - Mean loss: {:.6f}    Avg. embedding norm: {:.4f}   #positives/negatives: {:.1f}/{:.1f}'
        print(s.format(phase, stats['loss'], stats['avg_embedding_norm'], stats['num_pos'], stats['num_neg']))

    s = ''
    l = []
    if 'mean_pos_pair_dist' in stats:
        s += 'Pos dist (min/mean/max): {:.4f}/{:.4f}/{:.4f}   Neg dist (min/mean/max): {:.4f}/{:.4f}/{:.4f}'
        l += [stats['min_pos_pair_dist'], stats['mean_pos_pair_dist'], stats['max_pos_pair_dist'],
              stats['min_neg_pair_dist'], stats['mean_neg_pair_dist'], stats['max_neg_pair_dist']]
    if 'pos_loss' in stats:
        if len(s) > 0:
            s += '   '
        s += 'Pos loss: {:.4f}  Neg loss: {:.4f}'
        l += [stats['pos_loss'], stats['neg_loss']]
    if len(l) > 0:
        print(s.format(*l))


def tensors_to_numbers(stats):
    stats = {e: stats[e].item() if torch.is_tensor(stats[e]) else stats[e] for e in stats}
    return stats


def do_train(dataloaders, params: MinkLocParams, ckpt=None, debug=False, visualize=False):
    # Create model class
    s = get_datetime()
    now = datetime.now()
    now_strftime = now.strftime("%Y%m%d-%H%M%S")
    model = model_factory(params)
    start_epoch = 1
    if ckpt is not None:
        checkpoint = torch.load(ckpt)
        model.load_state_dict(checkpoint)
        print("Loaded model from ", ckpt)
        start_epoch = int(ckpt.split('/')[-1].split('.')[0].split('epoch')[-1]) + 1
        print("Starting from epoch ", start_epoch)

    model_name = 'model_' + params.model_params.backbone + params.model_params.pooling + \
                 '_' + now_strftime.replace('-', '_')
    print('Model name: {}'.format(model_name))
    weights_path = create_weights_folder()
    model_pathname = os.path.join(weights_path, model_name)
    pathlib.Path(model_pathname).mkdir(parents=False, exist_ok=True)

    # 初始化Loss统计文件
    loss_stats_file = "每轮Loss统计.txt"
    with open(loss_stats_file, 'w', encoding='utf-8') as f:
        f.write("Epoch\tAverage_Loss\tMin_Loss\tMax_Loss\tTotal_Batches\n")

    if hasattr(model, 'print_info'):
        model.print_info()
    else:
        n_params = sum([param.nelement() for param in model.parameters()])
        print('Number of model parameters: {}'.format(n_params))

    # Move the model to the proper device before configuring the optimizer
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{params.model_params.gpu}")
        torch.cuda.set_device(device)
        model.to(device)
    else:
        device = "cpu"

    print('Model device: {}'.format(device))

    loss_fn = make_loss(params)

    # Training elements
    if params.weight_decay is None or params.weight_decay == 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)

    if params.scheduler is None:
        scheduler = None
    else:
        if params.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.epochs + 1,
                                                                   eta_min=params.min_lr)
        elif params.scheduler == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, params.scheduler_milestones, gamma=0.5)
        else:
            raise NotImplementedError('Unsupported LR scheduler: {}'.format(params.scheduler))

    ###########################################################################
    # Initialize TensorBoard writer
    ###########################################################################

    logdir = os.path.join("../tf_logs", now_strftime)
    writer = SummaryWriter(logdir)

    ###########################################################################
    #
    ###########################################################################

    is_validation_set = 'val' in dataloaders
    if is_validation_set:
        phases = ['train', 'val']
    else:
        phases = ['train']

    # Training statistics
    stats = {'train': [], 'val': [], 'eval': []}

    for epoch in tqdm.tqdm(range(start_epoch, params.epochs + 1)):
        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_stats = []  # running stats for the current epoch
            batch_losses = []  # 新增：记录每个batch的loss

            count_batches = 0

            # 添加数据质量检查 - 只在第一个epoch检查
            if epoch == 1 and phase == 'train':
                print("\n=== 数据质量检查 ===")

            for batch, positives_mask, negatives_mask in dataloaders[phase]:
                # batch is (batch_size, n_points, 3) tensor
                # labels is list with indexes of elements forming a batch
                count_batches += 1
                batch_stats = {}

                if debug and count_batches > 2:
                    break

                # Move everything to the device except 'coords' which must stay on CPU
                batch = {e: batch[e].to(device) if e != 'coords' else batch[e] for e in batch}

                n_positives = torch.sum(positives_mask).item()
                n_negatives = torch.sum(negatives_mask).item()

                # 数据质量调试信息 - 只在第一个epoch的前10个batch检查
                if epoch == 1 and phase == 'train' and count_batches <= 10:
                    # 计算每个样本的正负样本数量
                    pos_per_sample = torch.sum(positives_mask, dim=1)
                    neg_per_sample = torch.sum(negatives_mask, dim=1)

                    print(f"Batch {count_batches}: positives={n_positives}, negatives={n_negatives}")
                    print(f"🔍 详细分析 Batch {count_batches}:")
                    print(f"   Total positives: {n_positives}, Total negatives: {n_negatives}")
                    print(
                        f"   Pos per sample: min={pos_per_sample.min().item()}, max={pos_per_sample.max().item()}, mean={pos_per_sample.float().mean().item():.1f}")
                    print(
                        f"   Neg per sample: min={neg_per_sample.min().item()}, max={neg_per_sample.max().item()}, mean={neg_per_sample.float().mean().item():.1f}")

                if n_positives == 0 or n_negatives == 0:
                    # Skip a batch without positives or negatives
                    print('WARNING: Skipping batch without positive or negative examples')
                    continue
                else:
                    if epoch == 1 and phase == 'train' and count_batches <= 10:
                        print(f"Processing batch {count_batches}: {n_positives} positives, {n_negatives} negatives")

                optimizer.zero_grad()
                if visualize:
                    # visualize_batch(batch)
                    pass

                with torch.set_grad_enabled(phase == 'train'):
                    # Compute embeddings of all elements
                    torch.autograd.set_detect_anomaly(True)
                    embeddings = model(batch)
                    loss, temp_stats, _ = loss_fn(embeddings, positives_mask, negatives_mask)

                    temp_stats = tensors_to_numbers(temp_stats)
                    batch_stats.update(temp_stats)
                    batch_stats['loss'] = loss.item()

                    # 新增：记录batch loss
                    if phase == 'train':
                        batch_losses.append(loss.item())

                    if phase == 'train':
                        # 添加梯度裁剪
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        loss.backward()
                        optimizer.step()

                running_stats.append(batch_stats)
                torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

            # ******* PHASE END *******
            # 计算epoch统计并写入文件
            if phase == 'train' and batch_losses:
                avg_loss = np.mean(batch_losses)
                min_loss = np.min(batch_losses)
                max_loss = np.max(batch_losses)
                total_batches = len(batch_losses)

                # 写入Loss统计文件
                with open(loss_stats_file, 'a', encoding='utf-8') as f:
                    f.write(f"{epoch}\t{avg_loss:.6f}\t{min_loss:.6f}\t{max_loss:.6f}\t{total_batches}\n")

                print(
                    f"Epoch {epoch} Loss Statistics - Avg: {avg_loss:.6f}, Min: {min_loss:.6f}, Max: {max_loss:.6f}, Batches: {total_batches}")

            # Compute mean stats for the epoch
            epoch_stats = {}
            for key in running_stats[0].keys():
                temp = [e[key] for e in running_stats]
                epoch_stats[key] = np.mean(temp)

            stats[phase].append(epoch_stats)
            print_stats(epoch_stats, phase)

        # ******* EPOCH END *******

        if scheduler is not None:
            scheduler.step()

        loss_metrics = {'train': stats['train'][-1]['loss']}
        if 'val' in phases:
            loss_metrics['val'] = stats['val'][-1]['loss']
        writer.add_scalars('Loss', loss_metrics, epoch)

        if 'num_triplets' in stats['train'][-1]:
            nz_metrics = {'train': stats['train'][-1]['num_non_zero_triplets']}
            if 'val' in phases:
                nz_metrics['val'] = stats['val'][-1]['num_non_zero_triplets']
            writer.add_scalars('Non-zero triplets', nz_metrics, epoch)

        elif 'num_pairs' in stats['train'][-1]:
            nz_metrics = {'train_pos': stats['train'][-1]['pos_pairs_above_threshold'],
                          'train_neg': stats['train'][-1]['neg_pairs_above_threshold']}
            if 'val' in phases:
                nz_metrics['val_pos'] = stats['val'][-1]['pos_pairs_above_threshold']
                nz_metrics['val_neg'] = stats['val'][-1]['neg_pairs_above_threshold']
            writer.add_scalars('Non-zero pairs', nz_metrics, epoch)

        if params.batch_expansion_th is not None:
            # Dynamic batch expansion
            epoch_train_stats = stats['train'][-1]
            if 'num_non_zero_triplets' not in epoch_train_stats:
                print('WARNING: Batch size expansion is enabled, but the loss function is not supported')
            else:
                # Ratio of non-zero triplets
                rnz = epoch_train_stats['num_non_zero_triplets'] / epoch_train_stats['num_triplets']
                if rnz < params.batch_expansion_th:
                    dataloaders['train'].batch_sampler.expand_batch()

        print('')

        if params.dataset_name != 'TUM' and epoch % 10 != 0 and epoch > 0 and epoch < params.epochs:
            continue

        # Save model weights
        final_model_path = os.path.join(model_pathname, f'epoch{epoch}.pth')
        torch.save(model.state_dict(), final_model_path)

        # Evaluate the final model
        model.eval()
        with torch.no_grad():
            final_eval_stats = evaluate(model, device, params)
        print(f'\nEpoch{epoch} model:')
        print_eval_stats(final_eval_stats)
        stats['eval'].append({f'epoch{epoch}': final_eval_stats})
        print('')
        for database_name in final_eval_stats:
            nz_metric1 = {f'{database_name}': final_eval_stats[database_name]['ave_one_percent_recall'].item()}
            nz_metric2 = {f'{database_name}': final_eval_stats[database_name]['ave_recall'][0].item()}
            writer.add_scalars('Eval Mean recall', nz_metric2, epoch)
            writer.add_scalars('Eval One percent recall', nz_metric1, epoch)
            writer.flush()

    stats = {'stats': stats, 'params': vars(params)}
    with open(os.path.join(logdir, 'config.json'), 'w') as f:
        json.dump(stats, f, indent=6, default=lambda o: getattr(o, '__dict__', str(o)))


def export_eval_stats(file_name, prefix, eval_stats, dataset_name):
    s = '\n' + prefix + '\n\n'
    ave_1p_recall_l = []
    ave_recall_l = []
    # Print results on the final model
    with open(file_name, "a") as f:
        if dataset_name == 'USyd':
            ave_1p_recall = eval_stats['usyd']['ave_one_percent_recall']
            ave_1p_recall_l.append(ave_1p_recall)
            ave_recall = eval_stats['usyd']['ave_recall'][0]
            ave_recall_l.append(ave_recall)
            s += ", {:0.2f}, {:0.2f}".format(ave_1p_recall, ave_recall)

        elif dataset_name == 'IntensityOxford':
            ave_1p_recall = eval_stats['intensityOxford']['ave_one_percent_recall']
            ave_1p_recall_l.append(ave_1p_recall)
            ave_recall = eval_stats['intensityOxford']['ave_recall'][0]
            ave_recall_l.append(ave_recall)
            s += ", {:0.2f}, {:0.2f}".format(ave_1p_recall, ave_recall)
        elif dataset_name == 'TUM':
            ave_1p_recall = eval_stats['tum']['ave_one_percent_recall']
            ave_1p_recall_l.append(ave_1p_recall)
            ave_recall = eval_stats['tum']['ave_recall'][0]
            ave_recall_l.append(ave_recall)
            s += 'Average 1% recall: {:0.2f}\n'.format(ave_1p_recall)
            s += 'Average Recall: {:0.2f}\n'.format(ave_recall)
        else:
            for ds in ['oxford', 'university', 'residential', 'business']:
                ave_1p_recall = eval_stats[ds]['ave_one_percent_recall']
                ave_1p_recall_l.append(ave_1p_recall)
                ave_recall = eval_stats[ds]['ave_recall'][0]
                ave_recall_l.append(ave_recall)
                s += ", {:0.2f}, {:0.2f}".format(ave_1p_recall, ave_recall)

        mean_1p_recall = np.mean(ave_1p_recall_l)
        mean_recall = np.mean(ave_recall_l)
        s += 'Mean 1% Recall @N: {:0.2f}\nMean Recall {:0.2f}\n\n '.format(mean_1p_recall, mean_recall)
        f.write(s)


def create_weights_folder():
    # Create a folder to save weights of trained models
    this_file_path = pathlib.Path(__file__).parent.absolute()
    temp, _ = os.path.split(this_file_path)
    weights_path = os.path.join(temp, 'weights')
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    assert os.path.exists(weights_path), 'Cannot create weights folder: {}'.format(weights_path)
    return weights_path
