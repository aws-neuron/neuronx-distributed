nxd_convert_zero_checkpoints --input_dir test_data --output_dir zero1_full --convert_to_full
nxd_convert_zero_checkpoints --input_dir zero1_full --output_dir zero1_sharder --convert_to_sharded --dp_size 4
python3 check_zero1_equal.py

ret_val=$?
echo $ret_val
if [ $ret_val -eq 0 ]; then
    success=1
else
    success=0
fi
dump_to_s3_update_json_scr=$SCRIPT_DIR/../../../../dump_to_s3_update_test_json.sh
if [ -e $dump_to_s3_update_json_scr ]; then
    $dump_to_s3_update_json_scr $@ --key=inference_success --value=$success || echo "Unable to update test result JSON."
else
    echo "WARNING: Script $dump_to_s3_update_json_scr not found. Not updating test result JSON."
fi

exit $ret_val
