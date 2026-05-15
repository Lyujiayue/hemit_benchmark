cuda_device="4"

# feature_root='/storage/jmabq/VirtualStaining/features/GCHTID'
# save_root='./results/GCHTID'

feature_root="/storage/jmabq/VirtualStaining/features/UniToPath"
save_root="./results/UniToPatho"

# feature_root="/storage/jmabq/VirtualStaining/features/WSSS4LUAD"
# save_root="./results/WSSS4LUAD"

data_roots=(
    "$feature_root/HE/extracted_features.pth"
    "$feature_root/asp/extracted_features.pth"
    "$feature_root/cyclegan/extracted_features.pth"
    "$feature_root/her2/extracted_features.pth"
    "$feature_root/ours_512/extracted_features.pth"
    "$feature_root/p2p/extracted_features.pth"
    "$feature_root/reggan/extracted_features.pth"
    "$feature_root/st/extracted_features.pth"
    "$feature_root/stegogan/extracted_features.pth"
    "$feature_root/unsb/extracted_features.pth"
)

output_dirs=(
    "$save_root/HE"
    "$save_root/asp"
    "$save_root/cyclegan"
    "$save_root/her2"
    "$save_root/ours_512"
    "$save_root/p2p"
    "$save_root/reggan"
    "$save_root/st"
    "$save_root/stegogan"
    "$save_root/unsb"
)

modes=(
    "HE"
    "HE&IHC"
    "HE&IHC"
    "HE&IHC"
    "HE&IHC"
    "HE&IHC"
    "HE&IHC"
    "HE&IHC"
    "HE&IHC"
    "HE&IHC"
)

for i in ${!data_roots[@]}; do
    data_root=${data_roots[$i]}
    output_dir=${output_dirs[$i]}
    mode=${modes[$i]}
    mkdir -p $output_dir

    CUDA_VISIBLE_DEVICES=$cuda_device nohup python evaluate.py \
        --pth_file $data_root \
        --output_dir $output_dir \
        --batch_size 32 \
        --mode $mode > $output_dir/evaluate.log 2>&1 &
done

