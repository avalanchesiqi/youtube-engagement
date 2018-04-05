#!/bin/bash
# usage: ./run_all_temporal_analysis.sh

log_file=temporal_analysis.log

if [ -f "$log_file" ]; then
  rm "$log_file"
fi

python extract_temporal_engagement_data.py -i ../data/formatted_tweeted_videos -o ./temporal_engagement_data >> "$log_file"

sleep 60
echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' >> "$log_file"

python extract_temporal_engagement_map.py -i ./temporal_engagement_data -o ./temporal_engagement_map >> "$log_file"

sleep 60
echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' >> "$log_file"

python extract_temporal_engagement_dynamics.py -i ../data/formatted_tweeted_videos >> "$log_file"

sleep 60
echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' >> "$log_file"

python run_engagement_temporal_fitting.py -i ./sliding_engagement_dynamics.csv -o ./sliding_fitting_results.csv >> "$log_file"
