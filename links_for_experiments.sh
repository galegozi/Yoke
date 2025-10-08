#!/bin/bash

# Locations
SOURCE_DIR="/lustre/scratch5/exempt/artimis/mpmm/galgal/runs/chicoma_lsc_loderunner-ch-subsampling"
TARGET_DIR="/usr/projects/artimis/mpmm/galgal/Yoke/applications/harnesses/chicoma_lsc_loderunner-ch-subsampling/runs"

# Input file with directory names
LIST_FILE="dir_list.txt"

# Read each line from the list file
while IFS= read -r d || [[ -n "$d" ]]; do
    # Skip empty lines or lines starting with #
    [[ -z "$d" || "$d" =~ ^# ]] && continue

    mkdir -p "$SOURCE_DIR/$d"                 # make the directory
    ln -sfn "$SOURCE_DIR/$d" "$TARGET_DIR/$d" # create/update symlink
done < "$LIST_FILE"