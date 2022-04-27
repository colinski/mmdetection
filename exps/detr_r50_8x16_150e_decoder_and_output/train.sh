module load singularity

START=$PWD
cd $WORK
for seed in {100..1100..100}; do
    srun -p gypsum-titanx-phd -G 4 -c 24 --mem=150GB --exclude=ials-gpu015,gpu003,gpu004,ials-gpu006,ials-gpu033,ials-gpu029 \
        singularity run --nv -H $WORK --bind /work:/work \
            python.sif bash \
            mmdetection/tools/dist_train.sh \
            mmdetection/configs/detr/detr_r50_8x16_150e_decoder_and_output.py \
            4 \
            --seed $seed \
            --work-dir logs/detr_r50_8x16_150e_decoder_and_output_$seed &
done
cd $START
