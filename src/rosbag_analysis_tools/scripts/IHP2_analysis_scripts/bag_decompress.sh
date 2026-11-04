#!/bin/bash

bag_files=()
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "Below are the bag files that will be decompressed."
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
#List bag files in current directory
for file in *."bag"; do
  if [ -f "$file" ]; then
    echo "$file"
    bag_files+=("$file")
  fi
done
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

read -p "Press Enter to continue or Ctrl+C to exit."

#Make directory to store original bag files
mkdir -p origbag

#Decompress bag files
for file in "${bag_files[@]}";
do
    if [ -f "$file" ]; then
      if [ $(rosbag info -y -k compression $file) == "none" ]; then
        echo "$file is already decompressed."
      else
        echo "Processing file: $file"
        chmod 777 $file
        rosbag decompress $file
        prefix="${file%%.*}"
        mv "${prefix}.orig.bag" origbag/
        echo "Moved ${prefix}.orig.bag to origbag."
        rosbag reindex $file
        rm "${prefix}.orig.bag"
        echo "$file decompression complete."
      fi
    else
        echo "Error: file not found: $file"
    fi
done
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
