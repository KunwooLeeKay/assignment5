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
    parser.add_argument('--i', type=list, default=[0, 617, 719, 406, 651, 726], help="index of the object to visualize")
    # first 3 are correct, last 3 are wrong predictions from the very first classification model.

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output/cls')

    parser.add_argument('--exp_name', type=str, default="exp2", help='The name of the experiment')
    parser.add_argument('--rotation_angle', type=float, default=np.pi/4, help='Maximum rotation angle for exp1')

    parser.add_argument('--dgcnn', action='store_true', help='Use DGCNN model if specified')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Classification Task ------
    
    model = cls_model() if not args.dgcnn else cls_model_DGCNN()

    suffix = '_dgcnn' if args.dgcnn else ''

    # Load Model Checkpoint
    model_path = './checkpoints/cls{}/{}.pt'.format(suffix, args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(10000, args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy(np.load(args.test_label))


    # ------ TO DO: Make Prediction ------
    pred_label = model(test_data)
    pred_label = torch.argmax(pred_label, dim = 1)

    # Compute Accuracy
    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.size()[0])
    print ("test accuracy: {}".format(test_accuracy))

    # NOTE only run this part for first run to determine correct & incorrect viz
    # '''
    # Visualize a few random test point clouds and mention the predicted classes for each. 
    # Also visualize at least 1 failure prediction for each class (chair, vase and lamp), and provide interpretation in a few sentences.
    # '''
    # # find wrong prediction for each class
    # class_0_idx = np.where(test_label.data==0)[0]
    # class_1_idx = np.where(test_label.data==1)[0]
    # class_2_idx = np.where(test_label.data==2)[0]
    # # among each class, find wrong predictions
    # class_0_wrong = [idx for idx in class_0_idx if pred_label[idx] != 0]
    # class_1_wrong = [idx for idx in class_1_idx if pred_label[idx] != 1]
    # class_2_wrong = [idx for idx in class_2_idx if pred_label[idx] != 2]
    
    # for cls, wrong_list in zip([0,1,2], [class_0_wrong, class_1_wrong, class_2_wrong]):
    #     if len(wrong_list) == 0:
    #         print ("No wrong prediction for class {}".format(cls))
    #         continue
    #     idx = wrong_list[0]
    #     viz_cls(test_data[idx], pred_label[idx], 
    #             path="{}/cls{}_wrong_pred_obj{}_class{}.gif".format(args.output_dir, suffix, idx, cls),
    #             device=args.device,
    #             num_points=args.num_points)
    
    # class_0_correct = [idx for idx in class_0_idx if pred_label[idx] == 0]
    # class_1_correct = [idx for idx in class_1_idx if pred_label[idx] == 1]
    # class_2_correct = [idx for idx in class_2_idx if pred_label[idx] == 2]

    # for cls, correct_list in zip([0,1,2], [class_0_correct, class_1_correct, class_2_correct]):
    #     if len(correct_list) == 0:
    #         print ("No correct prediction for class {}".format(cls))
    #         continue
    #     idx = correct_list[0]
    #     viz_cls(test_data[idx], pred_label[idx], 
    #             path="{}/cls{}_correct_pred_obj{}_class{}.gif".format(args.output_dir, suffix, idx, cls),
    #             device=args.device,
    #             num_points=args.num_points)

    for index in args.i:
        viz_cls(test_data[index], pred_label[index], 
                path="{}/cls{}_pred_obj{}_class{}.gif".format(args.output_dir, suffix, index, pred_label[index]),
                device=args.device,
                num_points=args.num_points)
        
        
    if args.exp_name in ["exp1", "both"]:
        # rotate input with random rotation that varies for each object
        rotated_data = test_data.clone()

        for i in range(rotated_data.shape[0]):
            theta = np.random.uniform(0, args.rotation_angle)
            rotation_matrix = torch.tensor([[np.cos(theta), -np.sin(theta), 0],
                                            [np.sin(theta),  np.cos(theta), 0],
                                            [0,              0,             1]], dtype=torch.float32)
            rotated_data[i] = torch.matmul(test_data[i], rotation_matrix)

        pred_label = model(rotated_data)
        pred_label = torch.argmax(pred_label, dim = 1)

        for idx in args.i:
            cls = test_label[idx].item()
            viz_cls(rotated_data[idx], pred_label[idx], 
                    path="{}/cls_exp1_{}_pred_obj{}_class{}.gif".format(args.output_dir, suffix, idx, cls),
                    device=args.device,
                    num_points=args.num_points)

        rotated_test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.size()[0])
        print ("test accuracy with rotated input: {}".format(rotated_test_accuracy))

    if args.exp_name in ["exp2", "both"]:
        exp2_accs = []
        for exp in range(10):
            num_points = 100
            ind = np.random.choice(10000, num_points, replace=False)
            test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
            pred_label = model(test_data)
            pred_label = torch.argmax(pred_label, dim = 1)
            acc = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.size()[0])
            exp2_accs.append(acc)
            print ("[Exp {}] test accuracy with {} points: {}".format(exp, num_points, acc))

            for idx in [args.i[0], args.i[-1]]:
                cls = test_label[idx].item()
                viz_cls(rotated_data[idx], pred_label[idx], 
                        path="{}/cls_exp2_{}_{}_pred_obj{}_class{}.gif".format(args.output_dir, exp+1,  suffix, idx, cls),
                        device=args.device,
                        num_points=num_points)


    # write accuracies to a text file
    with open("{}/cls{}_accuracy_{}.txt".format(args.output_dir, suffix, args.exp_name), 'w') as f:
        f.write("Test accuracy: {}\n".format(test_accuracy))
        if args.exp_name in ["exp1", "both"]:
            f.write("Test accuracy with rotated input: {}\n".format(rotated_test_accuracy))
        if args.exp_name in ["exp2", "both"]:
            for exp, acc in enumerate(exp2_accs):
                f.write("[Exp {}] test accuracy with 100 points: {}\n".format(exp, acc))



        