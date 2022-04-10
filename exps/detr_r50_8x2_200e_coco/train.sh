
START=$PWD
cd $WORK

singularity run --nv -H $WORK sif/python.sif bash \
    src/mmdetection/tools/dist_train.sh \
    src/mmdetection/configs/detr/detr_r50_8x2_200e_coco.py \
    8 \
    --seed 1 \
    --work-dir src/mmdetection/exps/detr_r50_8x2_200e_coco

cd $START
