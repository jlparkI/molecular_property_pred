#!/bin/bash

usage(){
    echo "Usage: cmd [-j coupling constant type] [-o] [-l]

[-o]: output model predictions for coupling constant j
[-l]: dataset is large and features should not be built in memory

Note that pipeline.sh will try to train a new model for j UNLESS you specify -o.
If -o is indicated, pipeline.sh will try to find an existing model for j and generate predictions for the test set. It will only generate predictions for the test set if -o is specified. -l should be used for 2JHC and 3JHC. -j is the only
required option. In other words, you should build a model by running pipeline,
then run it again with -o flagged to generate predictions for the test set.

The coupling constant must be one of the following: 1JHN, 2JHN, 3JHN, 1JHC, 2JHC, 2JHH, 3JHC, 3JHH."
}

run_pipeline(){
if [ "$testonly" = "true" ]; then
    echo "Generating test predictions for ${jtype}"
    python scripts/runtestpreds.py $jtype
else
    if [ "$largedataset" = "true" ]; then
        python scripts/gen_train_datasets.py $jtype large
        python scripts/train_models.py $jtype large       
    else
        python scripts/gen_train_datasets.py $jtype normal
        python scripts/train_models.py $jtype normal
    fi
fi
}

#Note that since bash does not have boolean variables I am storing user
#preferences as strings "true" or "false" for code readability.
testonly="false"
largedataset="false"

while getopts "olj:" opt; do
    case "$opt" in
    o)
        testonly="true";;
    l)
        largedataset="true";;
    j)
        jtype=$OPTARG;;
    ?)
        usage
        exit 1;;
    esac
done
shift "$(($OPTIND -1))"

if [ -z "$jtype" ];
then
    usage
    exit 1
fi
case $jtype in 
    1JHC|2JHC|3JHC|1JHN|2JHN|3JHN|2JHH|3JHH)
        run_pipeline;;
    *)
        usage;;
esac
