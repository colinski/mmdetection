START=$PWD

cd $WORK 

singularity run --nv -H $WORK --bind /work:/work $WORK/python.sif python \
    $WORK/mmdetection/tools/detr/collect_output.py \
    /work/meet_and_colin_umass_edu/exps/detr_r50_16x4_5e_output_heads_only/detr_r50_4x16_5e_output_heads_only_seed_0/detr_r50_4x16_5e_output_heads_only.py \
    /work/meet_and_colin_umass_edu/exps/detr_r50_16x4_5e_output_heads_only/detr_r50_4x16_5e_output_heads_only_seed_0/latest.pth \
    /work/meet_and_colin_umass_edu/exps/detr_r50_16x4_5e_output_heads_only/detr_r50_4x16_5e_output_heads_only_seed_0/output.pkl


singularity run --nv -H $WORK --bind /work:/work $WORK/python.sif python \
    $WORK/mmdetection/tools/detr/coco_eval_from_pkl.py \
    /work/meet_and_colin_umass_edu/exps/detr_r50_16x4_5e_output_heads_only/detr_r50_4x16_5e_output_heads_only_seed_0/output.pkl \
    /work/meet_and_colin_umass_edu/exps/detr_r50_16x4_5e_output_heads_only/detr_r50_4x16_5e_output_heads_only_seed_0/results.json 


cd $START
