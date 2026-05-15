# hard coded paths
prefix="/jhcnas4/lwq/virtual_staining/downstream_tasks/predictions/UniToPatho"
prefix_save="/storage/jmabq/VirtualStaining/UniToPath"
csv_file="/jhcnas3/VirtualStaining/downstream_tasks/labels/UniToPatho_labels.csv"

DATA_ROOTS=(
    # "$prefix/cyclegan/predictions"
    # "$prefix/her2/predictions"
    # "$prefix/p2p/predictions"
    # "$prefix/reggan/predictions"
    # "$prefix/asp/predictions"
    # "$prefix/ours_512/predictions"
    # "$prefix/st/predictions"
    # "$prefix/stegogan/predictions"
    # "$prefix/unsb/predictions"
    "$prefix/bci/predictions"
)

SAVE_DIRS=(
    # "$prefix_save/cyclegan"
    # "$prefix_save/her2"
    # "$prefix_save/p2p"
    # "$prefix_save/reggan"
    # "$prefix_save/asp"
    # "$prefix_save/ours_512"
    # "$prefix_save/st"
    # "$prefix_save/stegogan"
    # "$prefix_save/unsb"
    "$prefix_save/bci"
)

device='cuda:2'


for i in "${!DATA_ROOTS[@]}"; do
    data_root="${DATA_ROOTS[$i]}"
    save_dir="${SAVE_DIRS[$i]}"
    echo "Processing data root: $data_root"
    echo "Saving features to: $save_dir"
    mkdir -p "$save_dir"
    python3 ./generate_feature.py \
    --data_root "$data_root" \
    --save_dir "$save_dir" \
    --csv_file "$csv_file" \
    --device "$device"
    
    echo "Finished processing $data_root"
    echo "-----------------------------------"
done
