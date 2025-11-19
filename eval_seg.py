import numpy as np
import argparse

import torch
from models import seg_model
from data_loader import get_data_loader
from utils import create_dir, viz_seg


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=1000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp2", help='The name of the experiment')
    parser.add_argument('--rotation_angle', type=float, default=np.pi/4, help='Maximum rotation angle for exp1')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Segmentation Task  ------
    model = seg_model(num_seg_classes = args.num_seg_class).to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/seg/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy((np.load(args.test_label))[:,ind])

    # ------ TO DO: Make Prediction ------
    pred_label = model(test_data.to(args.device))
    pred_label = torch.argmax(pred_label, dim = 2).cpu()

    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.reshape((-1,1)).size()[0])
    print ("test accuracy: {}".format(test_accuracy))



    # Visualize Segmentation Result (Pred VS Ground Truth)
    viz_seg(test_data[args.i], test_label[args.i], "{}/gt_{}.gif".format(args.output_dir, 'default'), args.device)
    viz_seg(test_data[args.i], pred_label[args.i], "{}/pred_{}.gif".format(args.output_dir, 'default'), args.device)


    if args.exp_name in ["exp1", "both"]:
        # rotate input with random rotation that varies for each object
        rotated_data = test_data.clone()

        for i in range(rotated_data.shape[0]):
            theta = np.random.uniform(0, args.rotation_angle)
            rotation_matrix = torch.tensor([[np.cos(theta), -np.sin(theta), 0],
                                            [np.sin(theta),  np.cos(theta), 0],
                                            [0,              0,             1]], dtype=torch.float32)
            rotated_data[i] = torch.matmul(test_data[i], rotation_matrix)

        pred_label = model(rotated_data.to(args.device))
        pred_label = torch.argmax(pred_label, dim = 2).cpu()
        test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.size()[0])
        print ("test accuracy with rotated input: {}".format(test_accuracy))

        viz_seg(test_data[args.i], test_label[args.i], "{}/gt_{}.gif".format(args.output_dir, args.exp_name), args.device)
        viz_seg(test_data[args.i], pred_label[args.i], "{}/pred_{}.gif".format(args.output_dir, args.exp_name), args.device)

    if args.exp_name in ["exp2", "both"]:
        for exp in range(10):
            num_points = 100
            ind = np.random.choice(10000, num_points, replace=False)
            test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
            pred_label = model(test_data)
            pred_label = torch.argmax(pred_label, dim = 2)
            acc = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.size()[0])
            print ("[Exp {}] test accuracy with {} points: {}".format(exp, num_points, acc))
            viz_seg(test_data[args.i], test_label[args.i], "{}/gt_{}_{}.gif".format(args.output_dir, args.exp_name, exp + 1), args.device)
            viz_seg(test_data[args.i], pred_label[args.i], "{}/pred_{}_{}.gif".format(args.output_dir, args.exp_name, exp + 1), args.device)
        