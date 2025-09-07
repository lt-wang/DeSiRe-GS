# python render_occlusion.py --config_path eval_output/waymo_reconstruction/084_stage2/config.yaml
# python render.py --config_path eval_output/waymo_reconstruction/084_stage2/config.yaml



# Stage 1
python train.py \
--config configs/emer_reconstruction_stage1.yaml \
source_path=dataset/waymo/0145050 \
model_path=eval_output/waymo/0145050/stage1

# Stage 2
python train.py \
--config configs/emer_reconstruction_stage2.yaml \
source_path=dataset/waymo/0145050 \
model_path=eval_output/waymo/0145050/stage2
uncertainty_model_path=eval_output/waymo/0145050/stage1/uncertainty_model30000.pth