import numpy as np
import argparse

import torch
from models import cls_model, cls_model_DGCNN
from utils import create_dir, viz_cls

from pdb import set_trace as st


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=1000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i', type=list, default=[0, 617, 719, 406, 651, 726],
                        help="indices of objects to visualize")

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output/cls')

    parser.add_argument('--exp_name', type=str, default="exp2", help='The name of the experiment')
    parser.add_argument('--rotation_angle', type=float, default=np.pi/4, help='Maximum rotation angle for exp1')

    parser.add_argument('--dgcnn', action='store_true', help='Use DGCNN model if specified')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='Batch size for evaluation to avoid OOM')

    return parser


def forward_in_batches(model, data, device, batch_size):
    """
    data: (M, N, 3) on CPU
    returns logits: (M, num_classes) on CPU
    """
    model.eval()
    preds = []
    M = data.shape[0]
    with torch.no_grad():
        for start in range(0, M, batch_size):
            end = min(start + batch_size, M)
            batch = data[start:end].to(device=device, dtype=torch.float32)  # (b, N, 3)
            logits = model(batch)                                           # (b, C)
            preds.append(logits.cpu())
    return torch.cat(preds, dim=0)  # (M, C)


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ Initialize model ------
    model = cls_model(num_classes=args.num_cls_class) if not args.dgcnn else cls_model_DGCNN(num_classes=args.num_cls_class)
    model = model.to(args.device)

    suffix = '_dgcnn' if args.dgcnn else ''

    # Load checkpoint
    model_path = './checkpoints/cls{}/{}.pt'.format(suffix, args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print("successfully loaded checkpoint from {}".format(model_path))

    # Load full test data on CPU
    full_test_data_np = np.load(args.test_data)             # (M, 10000, 3)
    full_test_label = torch.from_numpy(np.load(args.test_label)).long()  # (M,)

    M, N_all, _ = full_test_data_np.shape

    # Sample points per object
    ind = np.random.choice(N_all, args.num_points, replace=False)
    test_data = torch.from_numpy(full_test_data_np[:, ind, :])  # (M, num_points, 3)
    test_label = full_test_label                                # (M,)

    # ------ Make prediction (batched) ------
    logits = forward_in_batches(model, test_data, args.device, args.eval_batch_size)  # (M, C)
    pred_label = torch.argmax(logits, dim=1)                                          # (M,)

    # Compute accuracy
    test_accuracy = pred_label.eq(test_label).sum().item() / test_label.size(0)
    print("test accuracy: {}".format(test_accuracy))

    # ------------------------------------------------------------------
    # Visualization of selected examples
    # ------------------------------------------------------------------
    for index in args.i:
        cls = pred_label[index].item()
        out_path = "{}/cls{}_pred_obj{}_class{}.gif".format(args.output_dir, suffix, index, cls)
        viz_cls(
            test_data[index].cpu(),         # (num_points, 3)
            pred_label[index].cpu(),
            path=out_path,
            device=args.device,
            num_points=args.num_points,
        )
        print("saved", out_path)

    # ------------------------------------------------------------------
    # exp1: rotation robustness
    # ------------------------------------------------------------------
    if args.exp_name in ["exp1", "both"]:
        rotated_data = test_data.clone()  # still on CPU

        for i in range(rotated_data.shape[0]):
            theta = np.random.uniform(0, args.rotation_angle)
            R = torch.tensor(
                [[np.cos(theta), -np.sin(theta), 0],
                 [np.sin(theta),  np.cos(theta), 0],
                 [0,              0,             1]],
                dtype=torch.float32,
            )
            rotated_data[i] = torch.matmul(test_data[i], R)

        rotated_logits = forward_in_batches(model, rotated_data, args.device, args.eval_batch_size)
        rotated_pred = torch.argmax(rotated_logits, dim=1)

        for idx in args.i:
            cls_true = test_label[idx].item()
            out_path = "{}/cls_exp1_{}_pred_obj{}_class{}.gif".format(
                args.output_dir, suffix, idx, cls_true
            )
            viz_cls(
                rotated_data[idx].cpu(),
                rotated_pred[idx].cpu(),
                path=out_path,
                device=args.device,
                num_points=args.num_points,
            )
            print("saved", out_path)

        rotated_test_accuracy = rotated_pred.eq(test_label).sum().item() / test_label.size(0)
        print("test accuracy with rotated input: {}".format(rotated_test_accuracy))

    # ------------------------------------------------------------------
    # exp2: accuracy vs #points
    # ------------------------------------------------------------------
    if args.exp_name in ["exp2", "both"]:
        exp2_accs = []
        for exp in range(10):
            num_points = 100
            ind_small = np.random.choice(N_all, num_points, replace=False)
            test_data_small = torch.from_numpy(full_test_data_np[:, ind_small, :])  # (M, 100, 3)

            logits_small = forward_in_batches(model, test_data_small, args.device, args.eval_batch_size)
            pred_label_small = torch.argmax(logits_small, dim=1)

            acc = pred_label_small.eq(test_label).sum().item() / test_label.size(0)
            exp2_accs.append(acc)
            print("[Exp {}] test accuracy with {} points: {}".format(exp, num_points, acc))

            # Visualize just first and last index from args.i
            for idx in [args.i[0], args.i[-1]]:
                cls_true = test_label[idx].item()
                out_path = "{}/cls_exp2_{}_{}_pred_obj{}_class{}.gif".format(
                    args.output_dir, exp + 1, suffix, idx, cls_true
                )
                viz_cls(
                    test_data_small[idx].cpu(),
                    pred_label_small[idx].cpu(),
                    path=out_path,
                    device=args.device,
                    num_points=num_points,
                )
                print("saved", out_path)

    # write accuracies to a text file
    out_txt = "{}/cls{}_accuracy_{}.txt".format(args.output_dir, suffix, args.exp_name)
    with open(out_txt, 'w') as f:
        f.write("Test accuracy: {}\n".format(test_accuracy))
        if args.exp_name in ["exp1", "both"]:
            f.write("Test accuracy with rotated input: {}\n".format(rotated_test_accuracy))
        if args.exp_name in ["exp2", "both"]:
            for exp, acc in enumerate(exp2_accs):
                f.write("[Exp {}] test accuracy with 100 points: {}\n".format(exp, acc))

    print("wrote accuracy log to", out_txt)