cuda_device="1"

# feature_root='/storage/jmabq/VirtualStaining/features/GCHTID'
# save_root='./results/GCHTID'
# test_models=("bci")

# feature_root="/storage/jmabq/VirtualStaining/features/UniToPath"
# save_root="./results/UniToPatho"
# test_models=("bci")

feature_root="/storage/jmabq/VirtualStaining/features/WSSS4LUAD"
save_root="./results/WSSS4LUAD"
test_models=("bci")

# test_models=(
#     "asp"
#     "cyclegan"
#     "her2"
#     "ours_512"
#     "p2p"
#     "reggan"
#     "st"
#     "stegogan"
#     "unsb"
# )




for i in ${!test_models[@]}; do
    data_root="$feature_root/${test_models[$i]}/extracted_features.pth"
    output_dir="$save_root/${test_models[$i]}"

    mkdir -p $output_dir

    CUDA_VISIBLE_DEVICES=$cuda_device nohup python linear_prob.py \
        --pth_file $data_root \
        --output_dir $output_dir \
        --batch_size 32 \
        --mode "HE&IHC" \
        --epochs 100 \
        --seed 120 \
        --learning_rate 0.002 > $output_dir/train_he.log 2>&1 &
done

