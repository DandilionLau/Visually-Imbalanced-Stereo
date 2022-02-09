CUDA_LAUNCH_BLOCKING=1 python train.py --only_test 1 --input_nc 3 --dataset data_scene_flow --gpu_num 1 --lr 1e-5 --nEpochs 100 --loss Smooth-L1 --batchSize 1 --testBatchSize 1 --loading_weights 1 --scale_factor 1 \
--filter_size_horizontal 201 --filter_size_vertical 0  --image_width 1242 --image_height 375 --weight_source ours
#--filter_size_horizontal 201 --filter_size_vertical 81  --image_width 1242 --image_height 375


# Update Feb: old flag --dataset toy_kitti  , new --dataset 
