#!/bin/bash
# usage: ./run_all_predictors.sh wp
# usage: ./run_all_predictors.sh re

target="$1"

if [ -d output ]; then
  rm -rf output
fi
mkdir output

if [ -f "$target"_predictions.log ]; then
  rm "$target"_predictions.log
fi
log_file="$target"_predictions.log

if [ "$target" == "wp" ]; then
  python extract_channel_reputation.py -i ./ -o ./output/train_channel_watch_percentage.txt -f "$target" >> "$log_file"
else
  python extract_channel_reputation.py -i ./ -o ./output/train_channel_relative_engagement.txt -f "$target" >> "$log_file"
fi

sleep 60
echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' >> "$log_file"

python duration_predictor.py -i ./ -o ./output -f "$target" >> "$log_file"

sleep 60
echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' >> "$log_file"

python context_predictor.py -i ./ -o ./output -f "$target" >> "$log_file"

sleep 60
echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' >> "$log_file"

python sparse_topic_predictor.py -i ./ -o ./output -f "$target" >> "$log_file"

sleep 60
echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' >> "$log_file"

python sparse_context_topic_predictor.py -i ./ -o ./output -f "$target" >> "$log_file"

sleep 60
echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' >> "$log_file"

python reputation_predictor.py -i ./ -o ./output -f "$target" >> "$log_file"

sleep 60
echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' >> "$log_file"

python python sparse_all_predictor.py -i ./ -o ./output -f "$target" >> "$log_file"

sleep 60
echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' >> "$log_file"

python python channel_specific_predictor.py -i ./ -o ./output -f "$target" >> "$log_file"

sleep 60
echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' >> "$log_file"

if [ "$target" == "wp" ]; then
  python python construct_pandas_frame.py -i ./output -o ./output/predicted_wp_df.csv -f "$target" >> "$log_file"
else
  python python construct_pandas_frame.py -i ./output -o ./output/predicted_re_df.csv -f "$target" >> "$log_file"
fi
