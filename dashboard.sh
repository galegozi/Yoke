echo "Debug Jobs"
squeue -u galgal --partition gpu_debug

echo; echo; echo;

echo "Running Jobs"
squeue -u galgal -t R

echo; echo; echo;

echo "ETA for jobs"
squeue -u galgal --start

# echo "All Jobs"
# squeue -u galgal