START=$PWD

cd $WORK 

singularity run --nv -H $WORK $WORK/sif/python.sif python \
    $WORK/src/mmdetection/tools/detr/collect_output.py \
    $WORK/src/mmdetection/configs/detr/detr_r50_8x2_150e_coco.py \
    $WORK/checkpoints/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth \
    output.pkl


singularity run --nv -H $WORK $WORK/sif/python.sif python \
    $WORK/src/mmdetection/tools/detr/coco_eval_from_pkl.py \
    output.pkl \
    results.json 

rm output.pkl
rm results.json

cd $START
