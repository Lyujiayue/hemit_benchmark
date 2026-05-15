data_roots=(
    "/storage/jmabq/VirtualStaining/features/GCHTID/ours_512/extracted_features.pth"
    "/storage/jmabq/VirtualStaining/features/UniToPath/ours_512/extracted_features.pth"
    "/storage/jmabq/VirtualStaining/features/WSSS4LUAD/ours_512/extracted_features.pth"
)


output_dirs=(
    "./results/GCHTID/HE"
    "./results/UniToPatho/HE"
    "./results/WSSS4LUAD/HE"
)

cuda_devices=("1" "1" "1")

for i in ${!data_roots[@]}; do
    data_root=${data_roots[$i]}
    output_dir=${output_dirs[$i]}
    cuda_device=${cuda_devices[$i]}

    mkdir -p $output_dir

    CUDA_VISIBLE_DEVICES=$cuda_device nohup python linear_prob.py \
        --pth_file $data_root \
        --output_dir $output_dir \
        --batch_size 32 \
        --mode HE \
        --epochs 50 \
        --learning_rate 0.001 > $output_dir/train_he.log 2>&1 &
done

