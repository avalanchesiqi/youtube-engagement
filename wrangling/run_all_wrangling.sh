#!/bin/bash
# usage: ./run_all_wrangling.sh

log_file=data_wrangling.log

if [ -f "$log_file" ]; then
  rm "$log_file"
fi

python construct_formatted_dataset.py -i ../data/tweeted_videos -o ../data/formatted_tweeted_videos >> "$log_file"

sleep 60
echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' >> "$log_file"

python construct_formatted_dataset.py -i ../data/quality_videos -o ../data/formatted_quality_videos >> "$log_file"

sleep 60
echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' >> "$log_file"

python extract_engagement_map.py -i ../data/formatted_tweeted_videos -o ../data/engagement_map.p >> "$log_file"

sleep 60
echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' >> "$log_file"

python split_dataset_and_append_relative_engagement.py -i ../data/formatted_tweeted_videos -o ../engagement_prediction >> "$log_file"

sleep 60
echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' >> "$log_file"

python construct_channel_view_dataset.py -i ../engagement_prediction >> "$log_file"
