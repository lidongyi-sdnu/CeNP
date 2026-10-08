import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms
import random
import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from scipy.stats import mode
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, balanced_accuracy_score, accuracy_score, hamming_loss, f1_score, recall_score
from sklearn.model_selection import KFold
from collections import OrderedDict
import json


def train(args, train_df, milnet, criterion, optimizer):
    milnet.train()
    dirs = shuffle(train_df)
    total_loss = 0
    pseudo_bags_base_dir = "/home/student5/data8t/WSI_cls/pseudo_bags_6/"
    Tensor = torch.cuda.FloatTensor
    for i, item in enumerate(dirs):
        bag_name = os.path.splitext(os.path.basename(item))[0]
        pseudo_bags_dir = os.path.join(pseudo_bags_base_dir, bag_name)
        pseudo_bags_paths = glob.glob(os.path.join(pseudo_bags_dir, "*.pt"))

        item_data = torch.load(item, map_location='cuda:0')
        if item_data.numel() == 0:  # 检查张量是否为空
            print(f"Warning: File {item} is empty. Skipping...")
            continue  # 跳过当前迭代，继续处理下一个文件
        try:
            bag_label = Tensor(item_data[0, args.feats_size:]).unsqueeze(0)
            # print(f"bag_label is: {bag_label.shape}")
        except IndexError as e:
            print(f"Error: {e}")
            # print(f"bag_data shape: {bag_label.shape}")
            raise

        psebag_pred_collect = []
        pseins_pred_collect = []

        for pseudo_bag_path in pseudo_bags_paths:
            # print(f"Processing: {pseudo_bag_path}")
            stacked_data = torch.load(pseudo_bag_path, map_location='cuda:0')
            if stacked_data.numel() == 0:  # 检查张量是否为空
                print(f"Warning: File {pseudo_bag_path} is empty. Skipping...")
                continue  # 跳过当前迭代，继续处理下一个文件
            # print(f"Data shape: {stacked_data.shape}")

            try:
                bag_feats = Tensor(stacked_data[:, :args.feats_size])
                bag_feats = dropout_patches(bag_feats, 1 - args.dropout_patch)
                bag_feats = bag_feats.view(-1, args.feats_size)
            except IndexError as e:
                print(f"Error: {e}")
                # print(f"stacked_data shape: {stacked_data.shape}")
                raise
            # 前向传播
            optimizer.zero_grad()
            ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
            # print(bag_prediction.shape)
            max_prediction, _ = torch.max(ins_prediction, 0)
            psebag_pred_collect.append(bag_prediction)
            pseins_pred_collect.append(max_prediction)

        if len(psebag_pred_collect) > 0 and len(pseins_pred_collect) > 0:
            stacked_predictions_bag = torch.stack(psebag_pred_collect)
            final_prediction_bag = torch.mean(stacked_predictions_bag, dim=0, keepdim=True)
            stacked_predictions_ins = torch.stack(pseins_pred_collect)
            final_prediction_ins = torch.mean(stacked_predictions_ins, dim=0, keepdim=True)
            # print("\nFinal aggregated prediction:")
            # print(final_prediction_bag.shape)
        else:
            print("Warning: No pseudo-bags processed!")

        # print(final_prediction_bag.view(1, -1).shape)
        bag_loss = criterion(final_prediction_bag.view(1, -1), bag_label.view(1, -1))
        max_loss = criterion(final_prediction_ins.view(1, -1), bag_label.view(1, -1))
        loss = 0.5 * bag_loss + 0.5 * max_loss
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_df), loss.item()))
    return total_loss / len(train_df)

# 实现特征的随机丢弃，也就是所谓的dropout。Dropout是一种正则化技术，用于防止神经网络过拟合。
def dropout_patches(feats, p):
    num_rows = feats.size(0)
    num_rows_to_select = int(num_rows * p)
    random_indices = torch.randperm(num_rows)[:num_rows_to_select]
    selected_rows = feats[random_indices]
    return selected_rows


def test(args, test_df, milnet, criterion, thresholds=None, return_predictions=False):
    milnet.eval()
    total_loss = 0
    test_labels = []
    test_predictions = []
    pseudo_bags_base_dir = "/home/student5/data8t/WSI_cls/pseudo_bags_6/"
    Tensor = torch.cuda.FloatTensor
    with torch.no_grad():
        for i, item in enumerate(test_df):
            bag_name = os.path.splitext(os.path.basename(item))[0]
            pseudo_bags_dir = os.path.join(pseudo_bags_base_dir, bag_name)
            pseudo_bags_paths = glob.glob(os.path.join(pseudo_bags_dir, "*.pt"))
            item_data = torch.load(item, map_location='cuda:0')
            if item_data.numel() == 0:  # 检查张量是否为空
                print(f"Warning: File {item_data} is empty. Skipping...")
                continue  # 跳过当前迭代，继续处理下一个文件
            try:
                bag_label = Tensor(item_data[0, args.feats_size:]).unsqueeze(0)
                # print(f"bag_label is: {bag_label.shape}")
            except IndexError as e:
                print(f"Error: {e}")
                # print(f"bag_data shape: {bag_label.shape}")
                raise

            psebag_predtest_collect = []
            pseins_predtest_collect = []

            for pseudo_bag_path_test in pseudo_bags_paths:
                stacked_data = torch.load(pseudo_bag_path_test, map_location='cuda:0')
                if stacked_data.numel() == 0:  # 检查张量是否为空
                    print(f"Warning: File {stacked_data} is empty. Skipping...")
                    continue  # 跳过当前迭代，继续处理下一个文件
                try:
                    bag_feats = Tensor(stacked_data[:, :args.feats_size])
                    bag_feats = dropout_patches(bag_feats, 1 - args.dropout_patch)
                    bag_feats = bag_feats.view(-1, args.feats_size)
                    # -1表示自动计算该维度的特征
                except IndexError as e:
                    print(f"Error: {e}")
                    raise

                ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
                max_prediction, _ = torch.max(ins_prediction, 0)
                psebag_predtest_collect.append(bag_prediction)
                pseins_predtest_collect.append(max_prediction)

            if len(psebag_predtest_collect) > 0 and len(pseins_predtest_collect) > 0:
                stacked_predictions_bag_test = torch.stack(psebag_predtest_collect)
                final_prediction_bag_test = torch.mean(stacked_predictions_bag_test, dim=0, keepdim=True)
                stacked_predictions_ins_test = torch.stack(pseins_predtest_collect)
                final_prediction_ins_test = torch.mean(stacked_predictions_ins_test, dim=0, keepdim=True)
                # print("\nFinal aggregated prediction:")
                # print(final_prediction_bag_test.shape)
            else:
                print("Warning: No pseudo-bags processed!")

            bag_loss = criterion(final_prediction_bag_test.view(1, -1), bag_label.view(1, -1))
            max_loss = criterion(final_prediction_ins_test.view(1, -1), bag_label.view(1, -1))
            loss = 0.5 * bag_loss + 0.5 * max_loss
            total_loss = total_loss + loss.item()
            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_df), loss.item()))
            test_labels.extend([bag_label.squeeze().cpu().numpy().astype(int)])
            if args.average:
                test_predictions.extend(
                    [(torch.sigmoid(max_prediction) + torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
            else:
                test_predictions.extend([torch.sigmoid(bag_prediction).squeeze().cpu().numpy()])
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)

    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)
    if thresholds: thresholds_optimal = thresholds
    if args.num_classes == 1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions >= thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions < thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
    else:
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i] >= thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i] < thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
    bag_score = 0
    for i in range(0, len(test_df)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score
    avg_score = bag_score / len(test_df)

    # 计算F1 Score和Recall
    if args.num_classes == 1:
        f1 = f1_score(test_labels, test_predictions)
        recall = recall_score(test_labels, test_predictions)
    else:
        f1 = f1_score(test_labels, test_predictions, average='macro')  # 多标签用macro平均
        recall = recall_score(test_labels, test_predictions, average='macro')

    if return_predictions:
        return total_loss / len(test_df), avg_score, auc_value, thresholds_optimal, test_predictions, test_labels, f1, recall
    return total_loss / len(test_df), avg_score, auc_value, thresholds_optimal, f1, recall


# 用于多标签分类任务的 ROC（Receiver Operating Characteristic）曲线计算
# 和 AUC（Area Under the ROC Curve）评分的函数。它还计算了每个类别的最优阈值
def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape) == 1:
        predictions = predictions[:, None]
    if labels.ndim == 1:
        labels = np.expand_dims(labels, axis=-1)
    for c in range(0, num_classes):
        # 提取 labels 矩阵的第c列
        label = labels[:, c]
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        # c_auc = roc_auc_score(label, prediction)
        try:
            c_auc = roc_auc_score(label, prediction)
            print("ROC AUC score:", c_auc)
        except ValueError as e:
            if str(e) == "Only one class present in y_true. ROC AUC score is not defined in that case.":
                print("ROC AUC score is not defined when only one class is present in y_true. c_auc is set to 1.")
                c_auc = 1
            else:
                raise e

        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal


def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


def print_epoch_info(epoch, args, train_loss_bag, val_loss_bag, avg_score, aucs, val_f1, val_recall):
    if args.dataset.startswith('TCGA'):
        print('\r Epoch [%d/%d] train loss: %.4f Val loss: %.4f, Val_avg_score: %.4f, Val_auc_LGG: %.4f,'
              ' Val_auc_GBM: %.4f, Val F1: %.4f, Val Recall: %.4f' %
              (epoch, args.num_epochs, train_loss_bag, val_loss_bag, avg_score, aucs[0], aucs[1], val_f1, val_recall))
    else:
        print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, avg score: %.4f, F1: %.4f, Recall: %.4f, AUC: ' %
              (epoch, args.num_epochs, train_loss_bag, val_loss_bag, avg_score, val_f1, val_recall) +
              '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs)))


def get_current_score(avg_score, aucs):
    current_score = (sum(aucs) + avg_score) / 2
    return current_score


def save_model(args, fold, run, save_path, model, thresholds_optimal):
    # Construct the filename including the fold number
    save_name = os.path.join(save_path, f'fold_{fold}_{run + 1}.pth')
    torch.save(model.state_dict(), save_name)
    print_save_message(args, save_name, thresholds_optimal)
    file_name = os.path.join(save_path, f'fold_{fold}_{run + 1}.json')
    with open(file_name, 'w') as f:
        json.dump([float(x) for x in thresholds_optimal], f)


def print_save_message(args, save_name, thresholds_optimal):
    if args.dataset.startswith('TCGA'):
        print('Best model saved at: ' + save_name + ' Best thresholds: LGG %.4f, GBM %.4f' % (
        thresholds_optimal[0], thresholds_optimal[1]))
    else:
        print('Best model saved at: ' + save_name)
        print('Best thresholds ===>>> ' + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)))



def main():
    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=6144, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=0.0001, type=float, help='Initial learning rate [0.0001]')
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of total training epochs [100]')
    parser.add_argument('--stop_epochs', default=10, type=int,
                        help='Skip remaining epochs if training has not improved after N epochs [10]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(1,), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight decay [1e-3]')
    parser.add_argument('--dataset', default='TCGA', type=str, help='Dataset folder name')
    parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
    parser.add_argument('--model', default='dsmil', type=str, help='MIL model [dsmil]')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--average', type=bool, default=False,
                        help='Average the score of max-pooling and bag aggregating')
    parser.add_argument('--eval_scheme', default='5-fold-train+valid+test', type=str,
                        help='Evaluation scheme [5-fold-cv | 5-fold-cv-standalone-test | 5-fold-train+valid+test ]')

    args = parser.parse_args()
    print(args.eval_scheme)

    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)

    if args.model == 'dsmil':
        import dsmil as mil
    elif args.model == 'abmil':
        import abmil as mil

    def apply_sparse_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def init_model(args):
        i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
        b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes,
                                       dropout_v=args.dropout_node, nonlinear=args.non_linearity).cuda()
        milnet = mil.MILNet(i_classifier, b_classifier).cuda()
        milnet.apply(lambda m: apply_sparse_init(m))
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)
        return milnet, criterion, optimizer, scheduler

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    if args.eval_scheme == '5-fold-cv':
        temp_train_dir = "/home/ldy/WSI_cls/temp_train_1"
        # 加载所有 .pt 文件
        bags_path = glob.glob(os.path.join(temp_train_dir, "*.pt"))

        # bags_path = bags_path.sample(n=200)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_results = []

        save_path = os.path.join('weights_1', datetime.date.today().strftime("%Y%m%d"))
        os.makedirs(save_path, exist_ok=True)
        run = len(glob.glob(os.path.join(save_path, '*.pth')))

        for fold, (train_index, test_index) in enumerate(kf.split(bags_path)):
            print(f"Starting CV fold {fold}.")
            milnet, criterion, optimizer, scheduler = init_model(args)
            train_path = [bags_path[i] for i in train_index]
            test_path = [bags_path[i] for i in test_index]
            fold_best_score = 0
            best_ac = 0
            best_auc = 0
            counter = 0

            for epoch in range(1, args.num_epochs + 1):
                counter += 1
                train_loss_bag = train(args, train_path, milnet, criterion, optimizer)  # iterate all bags
                test_loss_bag, avg_score, aucs, thresholds_optimal = test(args, test_path, milnet, criterion)

                print_epoch_info(epoch, args, train_loss_bag, test_loss_bag, avg_score, aucs)
                scheduler.step()

                current_score = get_current_score(avg_score, aucs)
                if current_score > fold_best_score:
                    counter = 0
                    fold_best_score = current_score
                    best_ac = avg_score
                    best_auc = aucs
                    save_model(args, fold, run, save_path, milnet, thresholds_optimal)
                if counter > args.stop_epochs: break
            fold_results.append((best_ac, best_auc))
        mean_ac = np.mean(np.array([i[0] for i in fold_results]))
        mean_auc = np.mean(np.array([i[1] for i in fold_results]), axis=0)
        # Print mean and std deviation for each class
        print(f"Final results: Mean Accuracy: {mean_ac}")
        for i, mean_score in enumerate(mean_auc):
            print(f"Class {i}: Mean AUC = {mean_score:.4f}")

    elif args.eval_scheme == '5-fold-train+valid+test':
        temp_train_dir = "/home/student5/data8t/WSI_cls/temp_train_6/"
        # 加载所有 .pt 文件
        bags_path = glob.glob(os.path.join(temp_train_dir, "*.pt"))
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_results = []

        save_path = os.path.join('weights_ours', datetime.date.today().strftime("%Y%m%d"))
        os.makedirs(save_path, exist_ok=True)
        run = len(glob.glob(os.path.join(save_path, '*.pth')))

        # 打开文件以保存测试结果
        with open('text_result_1.txt', 'w') as result_file:
            result_file.write("Iteration\tTest Accuracy\tTest AUC\tTest F1\tTest Recall\n")  # 写入表头

            for fold, (train_index, test_index) in enumerate(kf.split(bags_path)):
                print(f"Starting CV fold: {fold + 1}.")
                milnet, criterion, optimizer, scheduler = init_model(args)

                # bags_path = shuffle(bags_path)
                # total_samples = len(bags_path)
                # train_end = int(total_samples * (1 - args.split - 0.1))
                # val_end = train_end + int(total_samples * 0.1)
                #
                # train_path = bags_path[:train_end]
                # val_path = bags_path[train_end:val_end]
                # test_path = bags_path[val_end:]
                # 划分训练集和测试集
                train_path = [bags_path[i] for i in train_index]
                test_path = [bags_path[i] for i in test_index]

                # 进一步划分训练集和验证集
                val_size = int(len(train_path) * 0.1)  # 10% 作为验证集
                val_path = train_path[:val_size]
                train_path = train_path[val_size:]

                fold_best_score = 0
                best_ac = 0
                best_auc = 0
                counter = 0

                for epoch in range(1, args.num_epochs + 1):
                    counter += 1
                    train_loss_bag = train(args, train_path, milnet, criterion, optimizer)  # iterate all bags
                    val_loss_bag, avg_score, val_aucs, thresholds_optimal, val_f1, val_recall = test(args, val_path, milnet, criterion)

                    print_epoch_info(epoch, args, train_loss_bag, val_loss_bag, avg_score, val_aucs, val_f1, val_recall)
                    scheduler.step()

                    current_score = get_current_score(avg_score, val_aucs)
                    if current_score > fold_best_score:
                        counter = 0
                        fold_best_score = current_score
                        best_ac = avg_score
                        best_auc = val_aucs
                        save_model(args, fold, run, save_path, milnet, thresholds_optimal)
                        best_model = copy.deepcopy(milnet)
                    if counter > args.stop_epochs:
                        break

                # 在每次迭代结束后，立即对测试集进行测试
                test_loss_bag, test_avg_score, test_aucs, _, test_f1, test_recall = test(args, test_path, best_model, criterion)
                fold_results.append((best_ac, best_auc, test_avg_score, test_aucs, test_f1, test_recall))  # 记录验证集和测试集的结果

                # 将测试结果写入文件
                result_file.write(
                    f"{fold + 1}\t{test_avg_score:.4f}\t{', '.join([f'{auc:.4f}' for auc in test_aucs])}\t{test_f1:.4f}\t{test_recall:.4f}\n")
                print(f"Fold {fold + 1} results saved to text_result.txt.")

        # 计算并打印最终结果
        mean_ac = np.mean(np.array([i[0] for i in fold_results]))  # 验证集的平均准确率
        mean_auc = np.mean(np.array([i[1] for i in fold_results]), axis=0)  # 验证集的平均AUC
        mean_test_ac = np.mean(np.array([i[2] for i in fold_results]))  # 测试集的平均准确率
        mean_test_auc = np.mean(np.array([i[3] for i in fold_results]), axis=0)  # 测试集的平均AUC

        print(f"Final results - Validation: Mean Accuracy: {mean_ac}")
        for i, mean_score in enumerate(mean_auc):
            print(f"Class {i}: Mean AUC = {mean_score:.4f}")

        print(f"Final results - Test: Mean Accuracy: {mean_test_ac}")
        for i, mean_score in enumerate(mean_test_auc):
            print(f"Class {i}: Mean AUC = {mean_score:.4f}")

        mean_test_f1 = np.mean(np.array([i[4] for i in fold_results]))
        mean_test_recall = np.mean(np.array([i[5] for i in fold_results]))
        print(f"Final results - Test: Mean F1: {mean_test_f1:.4f}, Mean Recall: {mean_test_recall:.4f}")


    # elif args.eval_scheme == '5-time-train+valid+test':
    #     bags_path = glob.glob('temp_train/*.pt')
    #     # bags_path = bags_path.sample(n=50, random_state=42)
    #     fold_results = []
    #
    #     save_path = os.path.join('weights', datetime.date.today().strftime("%Y%m%d"))
    #     os.makedirs(save_path, exist_ok=True)
    #     run = len(glob.glob(os.path.join(save_path, '*.pth')))
    #
    #     for iteration in range(5):
    #         print(f"Starting iteration {iteration + 1}.")
    #         milnet, criterion, optimizer, scheduler = init_model(args)
    #
    #         bags_path = shuffle(bags_path)
    #         total_samples = len(bags_path)
    #         train_end = int(total_samples * (1 - args.split - 0.1))
    #         val_end = train_end + int(total_samples * 0.1)
    #
    #         train_path = bags_path[:train_end]
    #         val_path = bags_path[train_end:val_end]
    #         test_path = bags_path[val_end:]
    #
    #         fold_best_score = 0
    #         best_ac = 0
    #         best_auc = 0
    #         counter = 0
    #
    #         for epoch in range(1, args.num_epochs + 1):
    #             counter += 1
    #             train_loss_bag = train(args, train_path, milnet, criterion, optimizer)  # iterate all bags
    #             test_loss_bag, avg_score, aucs, thresholds_optimal = test(args, val_path, milnet, criterion)
    #
    #             print_epoch_info(epoch, args, train_loss_bag, test_loss_bag, avg_score, aucs)
    #             scheduler.step()
    #
    #             current_score = get_current_score(avg_score, aucs)
    #             if current_score > fold_best_score:
    #                 counter = 0
    #                 fold_best_score = current_score
    #                 best_ac = avg_score
    #                 best_auc = aucs
    #                 save_model(args, iteration, run, save_path, milnet, thresholds_optimal)
    #                 best_model = copy.deepcopy(milnet)
    #             if counter > args.stop_epochs: break
    #         test_loss_bag, avg_score, aucs, thresholds_optimal = test(test_path, best_model, criterion, args)
    #         fold_results.append((best_ac, best_auc))
    #     mean_ac = np.mean(np.array([i[0] for i in fold_results]))
    #     mean_auc = np.mean(np.array([i[1] for i in fold_results]), axis=0)
    #     # Print mean and std deviation for each class
    #     print(f"Final results: Mean Accuracy: {mean_ac}")
    #     for i, mean_score in enumerate(mean_auc):
    #         print(f"Class {i}: Mean AUC = {mean_score:.4f}")

    # 使用5折交叉验证训练模型，并在一个独立的测试集上评估模型性能。
    if args.eval_scheme == '5-fold-cv-standalone-test':
        bags_path = glob.glob('temp_train/*.pt')
        bags_path = shuffle(bags_path)
        reserved_testing_bags = bags_path[:int(args.split * len(bags_path))]
        bags_path = bags_path[int(args.split * len(bags_path)):]
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_results = []
        fold_models = []

        save_path = os.path.join('weights', datetime.date.today().strftime("%Y%m%d"))
        os.makedirs(save_path, exist_ok=True)
        run = len(glob.glob(os.path.join(save_path, '*.pth')))
        # 打开文件以保存测试结果

        with open('text_result.txt', 'w') as result_file:
            result_file.write("Iteration\tTest Accuracy\tTest AUC\n")  # 写入表头

        for fold, (train_index, val_index) in enumerate(kf.split(bags_path)):
            print(f"Starting CV fold {fold}.")
            milnet, criterion, optimizer, scheduler = init_model(args)
            train_path = [bags_path[i] for i in train_index]
            val_path = [bags_path[i] for i in val_index]
            fold_best_score = 0
            best_ac = 0
            best_auc = 0
            counter = 0
            best_model = []

            for epoch in range(1, args.num_epochs + 1):
                counter += 1
                train_loss_bag = train(args, train_path, milnet, criterion, optimizer)  # iterate all bags
                test_loss_bag, avg_score, aucs, thresholds_optimal = test(args, val_path, milnet, criterion)

                print_epoch_info(epoch, args, train_loss_bag, test_loss_bag, avg_score, aucs)
                scheduler.step()

                current_score = get_current_score(avg_score, aucs)
                if current_score > fold_best_score:
                    counter = 0
                    fold_best_score = current_score
                    best_ac = avg_score
                    best_auc = aucs
                    save_model(args, fold, run, save_path, milnet, thresholds_optimal)
                    best_model = [copy.deepcopy(milnet.cpu()), thresholds_optimal]
                    milnet.cuda()
                if counter > args.stop_epochs: break
            fold_results.append((best_ac, best_auc))
            fold_models.append(best_model)

        fold_predictions = []
        for item in fold_models:
            best_model = item[0]
            optimal_thresh = item[1]
            test_loss_bag, avg_score, aucs, thresholds_optimal, test_predictions, test_labels = test(args,
                                                                                                     reserved_testing_bags,
                                                                                                     best_model.cuda(),
                                                                                                     criterion,
                                                                                                     thresholds=optimal_thresh,
                                                                                                     return_predictions=True)
            fold_predictions.append(test_predictions)
        predictions_stack = np.stack(fold_predictions, axis=0)
        mode_result = mode(predictions_stack, axis=0)

        combined_predictions = mode_result.mode[0]
        combined_predictions = combined_predictions.squeeze()

        if args.num_classes > 1:
            # Compute Hamming Loss
            hammingloss = hamming_loss(test_labels, combined_predictions)
            print("Hamming Loss:", hammingloss)
            # Compute Subset Accuracy
            subset_accuracy = accuracy_score(test_labels, combined_predictions)
            print("Subset Accuracy (Exact Match Ratio):", subset_accuracy)
        else:
            accuracy = accuracy_score(test_labels, combined_predictions)
            print("Accuracy:", accuracy)
            balanced_accuracy = balanced_accuracy_score(test_labels, combined_predictions)
            print("Balanced Accuracy:", balanced_accuracy)

        os.makedirs('test', exist_ok=True)
        with open("test/test_list.json", "w") as file:
            json.dump(reserved_testing_bags, file)

        for i, item in enumerate(fold_models):
            best_model = item[0]
            optimal_thresh = item[1]
            torch.save(best_model.state_dict(), f"test/mil_weights_fold_{i}.pth")
            with open(f"test/mil_threshold_fold_{i}.json", "w") as file:
                optimal_thresh = [float(i) for i in optimal_thresh]
                json.dump(optimal_thresh, file)


if __name__ == '__main__':
    main()