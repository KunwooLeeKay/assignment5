import numpy as np
import argparse

import torch
from models import seg_model, seg_model_DGCNN
from utils import create_dir, viz_seg


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=500, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i', type=list,
                        default=[0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600],
                        help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/seg/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/seg/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output/seg')

    parser.add_argument('--exp_name', type=str, default="exp2", help='The name of the experiment')
    parser.add_argument('--rotation_angle', type=float, default=np.pi/4, help='Maximum rotation angle for exp1')

    parser.add_argument('--dgcnn', action='store_true', help='Use DGCNN model if specified')

    # to avoid OOM with DGCNN
    parser.add_argument('--eval_batch_size', type=int, default=16, help='Batch size for evaluation')

    return parser


def forward_in_batches_seg(model, data, device, batch_size):
    """
    data: (M, N, 3) on CPU
    returns logits: (M, N, C) on CPU
    """
    model.eval()
    preds = []
    M = data.shape[0]
    with torch.no_grad():
        for start in range(0, M, batch_size):
            end = min(start + batch_size, M)
            batch = data[start:end].to(device=device, dtype=torch.float32)  # (b, N, 3)
            logits = model(batch)                                           # (b, N, C)
            preds.append(logits.cpu())
    return torch.cat(preds, dim=0)  # (M, N, C)


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ Initialize Model for Segmentation Task  ------
    model = seg_model(num_seg_classes=args.num_seg_class) \
        if not args.dgcnn else seg_model_DGCNN(num_seg_classes=args.num_seg_class)
    model = model.to(args.device)

    suffix = '_dgcnn' if args.dgcnn else ''

    # Load Model Checkpoint
    model_path = './checkpoints/seg{}/{}.pt'.format(suffix, args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print("successfully loaded checkpoint from {}".format(model_path))

    # ----- Load full test data once -----
    full_test_data_np = np.load(args.test_data)          # (M, 10000, 3)
    full_test_label_np = np.load(args.test_label)        # (M, 10000)
    M, N_all, _ = full_test_data_np.shape

    # Sample points per object
    ind = np.random.choice(N_all, args.num_points, replace=False)
    test_data = torch.from_numpy(full_test_data_np[:, ind, :])      # (M, num_points, 3)  CPU
    test_label = torch.from_numpy(full_test_label_np[:, ind])       # (M, num_points)     CPU

    # ------ Make Prediction (batched) ------
    logits = forward_in_batches_seg(model, test_data, args.device, args.eval_batch_size)  # (M, N, C)
    pred_label = torch.argmax(logits, dim=2)                                              # (M, N) CPU

    # accuracy per point
    total_points = test_label.numel()
    correct_points = pred_label.eq(test_label).sum().item()
    test_accuracy = correct_points / total_points
    print("test accuracy: {}".format(test_accuracy))

    # ------ Visualize default (no rotation) ------
    for index in args.i:
        viz_seg(
            test_data[index].cpu(),            # (N, 3)
            test_label[index].cpu(),           # (N,)
            "{}/seg{}_{}_gt_default.gif".format(args.output_dir, suffix, index),
            args.device,
            args.num_points,
        )
        viz_seg(
            test_data[index].cpu(),
            pred_label[index].cpu(),
            "{}/seg{}_{}_pred_default.gif".format(args.output_dir, suffix, index),
            args.device,
            args.num_points,
        )

    # ------ exp1: rotation robustness ------
    if args.exp_name in ["exp1", "both"]:
        rotated_data = test_data.clone()  # CPU

        for i in range(rotated_data.shape[0]):
            theta = np.random.uniform(0, args.rotation_angle)
            R = torch.tensor(
                [[np.cos(theta), -np.sin(theta), 0],
                 [np.sin(theta),  np.cos(theta), 0],
                 [0,              0,             1]],
                dtype=torch.float32,
            )
            rotated_data[i] = torch.matmul(test_data[i], R)

        logits_rot = forward_in_batches_seg(model, rotated_data, args.device, args.eval_batch_size)
        pred_rot = torch.argmax(logits_rot, dim=2)  # (M, N) CPU

        correct_rot = pred_rot.eq(test_label).sum().item()
        rotated_test_accuracy = correct_rot / total_points
        print("test accuracy with rotated input: {}".format(rotated_test_accuracy))

        exp1_dir = "{}/exp1".format(args.output_dir)
        create_dir(exp1_dir)

        for index in args.i:
            viz_seg(
                rotated_data[index].cpu(),
                test_label[index].cpu(),
                "{}/seg{}_{}_gt_{}.gif".format(exp1_dir, suffix, index, args.exp_name),
                args.device,
                args.num_points,
            )
            viz_seg(
                rotated_data[index].cpu(),
                pred_rot[index].cpu(),
                "{}/seg{}_{}_pred_{}.gif".format(exp1_dir, suffix, index, args.exp_name),
                args.device,
                args.num_points,
            )

    # ------ exp2: accuracy vs #points ------
    if args.exp_name in ["exp2", "both"]:
        exp2_accs = []
        exp2_dir = "{}/exp2".format(args.output_dir)
        create_dir(exp2_dir)

        for exp in range(10):
            num_points = 100
            ind_small = np.random.choice(N_all, num_points, replace=False)
            test_data_small = torch.from_numpy(full_test_data_np[:, ind_small, :])    # (M, 100, 3)
            test_label_small = torch.from_numpy(full_test_label_np[:, ind_small])     # (M, 100)

            logits_small = forward_in_batches_seg(model, test_data_small, args.device, args.eval_batch_size)
            pred_small = torch.argmax(logits_small, dim=2)                             # (M, 100)

            total_points_small = test_label_small.numel()
            correct_small = pred_small.eq(test_label_small).sum().item()
            acc = correct_small / total_points_small
            exp2_accs.append(acc)
            print("[Exp2 - iter {}] test accuracy with {} points: {}".format(exp, num_points, acc))

            for index in [args.i[0], args.i[-1]]:
                viz_seg(
                    test_data_small[index].cpu(),
                    test_label_small[index].cpu(),
                    "{}/seg{}_{}_gt_{}_{}.gif".format(exp2_dir, suffix, index, args.exp_name, exp + 1),
                    args.device,
                    num_points,
                )
                viz_seg(
                    test_data_small[index].cpu(),
                    pred_small[index].cpu(),
                    "{}/seg{}_{}_pred_{}_{}.gif".format(exp2_dir, suffix, index, args.exp_name, exp + 1),
                    args.device,
                    num_points,
                )

    # ------ write accuracies to a text file ------
    out_txt = "{}/seg{}_accuracy_{}.txt".format(args.output_dir, suffix, args.exp_name)
    with open(out_txt, 'w') as f:
        f.write("Test accuracy: {}\n".format(test_accuracy))
        if args.exp_name in ["exp1", "both"]:
            f.write("Test accuracy with rotated input: {}\n".format(rotated_test_accuracy))
        if args.exp_name in ["exp2", "both"]:
            for exp, acc in enumerate(exp2_accs):
                f.write("[Exp2 - iter {}] test accuracy with 100 points: {}\n".format(exp, acc))

    print("wrote accuracy log to", out_txt)