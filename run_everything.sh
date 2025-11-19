# echo "run cls training"
# python3 train.py --task cls

# echo "evaluate cls model"
# python3 eval_cls.py --exp_name both

echo "run seg training"
python3 train.py --task seg

echo "evaluate seg model"
python3 eval_seg.py --exp_name both

echo "run cls_dgcnn training"
python3 train.py --task cls_dgcnn

echo "evaluate cls_dgcnn model"
python3 eval_cls.py --exp_name both --dgcnn

echo "run seg_dgcnn training"
python3 train.py --task seg_dgcnn

echo "evaluate seg_dgcnn model"
python3 eval_seg.py --exp_name both --dgcnn