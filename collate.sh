langs=("hau" "swa") #  "hau" "swa"

# experiment_name_prefix=afriberta_base

experiment_name=afriberta_large
# for i in "${langs[@]}"
#    do 
#        experiment_name="${experiment_name_prefix}_${i}"
declare -a arr=("amh" "hau" "swa" "ibo" "kin" "lug" "luo" "pcm" "wol" "yor")
for i in "${arr[@]}"
    do
        python ner_scripts/collate_results.py --lang "$i" --experiment-name "${experiment_name}_ner_results"
    done
#    done