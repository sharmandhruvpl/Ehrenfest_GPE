#!/bin/bash
# Script to automatically find and delete duplicate quantum dynamics files
# keeping only the newest version of each simulation type

# Set this to 1 to actually delete files, 0 for dry run (just list files that would be deleted)
DELETE_FILES=0

# Directory where files are located (current directory by default)
DATA_DIR="."

# Change to data directory
cd "$DATA_DIR" || { echo "Could not change to data directory"; exit 1; }

echo "Looking for quantum dynamics files in: $(pwd)"

# Function to extract core pattern from filename
extract_core() {
  filename="$1"
  # Pattern: quantum_dynamics_e{val}_z{val}_{forces}_TIMESTAMP.h5
  core=$(echo "$filename" | sed -E 's/quantum_dynamics_(e[0-9]+\.[0-9]+)_(z[0-9]+\.[0-9]+|zfunc)_(.+?)_[0-9]{8}_[0-9]{6}\.h5/\1_\2_\3/')
  echo "$core"
}

# Find all quantum dynamics .h5 files
files=$(find . -maxdepth 1 -name "quantum_dynamics_*.h5" | sort)

if [ -z "$files" ]; then
  echo "No quantum dynamics files found in the current directory."
  exit 0
fi

# Create a temporary file for processing
tmp_file=$(mktemp)

# Process each file and organize by core pattern
echo "$files" | while read -r filepath; do
  filename=$(basename "$filepath")
  core_pattern=$(extract_core "$filename")
  timestamp=$(echo "$filename" | grep -o '[0-9]\{8\}_[0-9]\{6\}')
  
  echo "$core_pattern::$timestamp::$filename" >> "$tmp_file"
done

# Find duplicates and determine which ones to delete
echo "Identifying duplicates..."
echo

declare -A patterns
while IFS= read -r line; do
  core_pattern=$(echo "$line" | cut -d':' -f1)
  timestamp=$(echo "$line" | cut -d':' -f3)
  filename=$(echo "$line" | cut -d':' -f5)
  
  if [ -n "${patterns[$core_pattern]}" ]; then
    # Compare timestamps to determine which to keep/delete
    existing_timestamp=$(echo "${patterns[$core_pattern]}" | cut -d':' -f1)
    existing_file=$(echo "${patterns[$core_pattern]}" | cut -d':' -f2)
    
    ts1=$(echo "$existing_timestamp" | tr -d '_')
    ts2=$(echo "$timestamp" | tr -d '_')
    
    if [ "$ts1" -lt "$ts2" ]; then
      echo "Found duplicate for pattern: $core_pattern"
      echo "  Will keep newer: $filename"
      echo "  Will delete older: $existing_file"
      echo
      
      # Mark for deletion
      if [ "$DELETE_FILES" -eq 1 ]; then
        if [ -f "$existing_file" ]; then
          rm "$existing_file"
          echo "  DELETED: $existing_file"
        else
          echo "  WARNING: File not found - $existing_file"
        fi
      fi
      
      # Update the newest file in our tracking
      patterns[$core_pattern]="$timestamp:$filename"
    else
      echo "Found duplicate for pattern: $core_pattern"
      echo "  Will keep newer: $existing_file"
      echo "  Will delete older: $filename"
      echo
      
      # Mark for deletion
      if [ "$DELETE_FILES" -eq 1 ]; then
        if [ -f "$filename" ]; then
          rm "$filename"
          echo "  DELETED: $filename"
        else
          echo "  WARNING: File not found - $filename"
        fi
      fi
    fi
  else
    # First time seeing this pattern, track it
    patterns[$core_pattern]="$timestamp:$filename"
  fi
done < <(sort -t':' -k1,1 "$tmp_file")

# Clean up
rm "$tmp_file"

if [ "$DELETE_FILES" -eq 0 ]; then
  echo -e "\nThis was a DRY RUN. No files were actually deleted."
  echo "To delete the files, set DELETE_FILES=1 at the top of the script."
else
  echo -e "\nDuplicates have been deleted."
fi
