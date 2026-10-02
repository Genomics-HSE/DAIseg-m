#!/bin/bash

#nohup ./prep.sh GBR > GBR.log 2>&1 &
# ./prep.sh CEU 4
POP=$1

DIR="/home/ailina/MXL.grch38.v2"



process_chromosome() {
    i=$1
    echo "Starting chromosome $i\n"

    json="${DIR}/MXL.grch38.chr${i}.json"
    echo $json

    [[ -f "$json" ]] && \
    echo "Processing $json" && \
    echo "  Step 1: restrict_1kG" && \
    python daiseg.py restrict_1kG -json "$json" -threads 16 && \
    echo "  Step 2: callability" && \
    python daiseg.py callability -json "$json" -threads 16 && \
    echo "  Step 3: main.prep" && \
    python daiseg.py main.prep -json "$json" -threads 16 && \
    echo "Finished chromosome $i successfully " || \
    { 
        if [[ ! -f "$json" ]]; then
            echo "ERROR: JSON file $json not found"
        else
            echo "ERROR: Processing failed for chr$i"
        fi
        return 1
    }

    return 0
}

export -f process_chromosome
export POP
export JOBS

chromosomes=({22..2})

echo "Chromosomes to process: ${chromosomes[@]}"

for chr in "${chromosomes[@]}"; do
    echo "Starting chromosome $chr at $(date)"
    process_chromosome "$chr" || {
        echo "ERROR: Failed on chromosome $chr"
        exit 1
    }

    echo "Completed chromosome $chr at $(date)"
done

