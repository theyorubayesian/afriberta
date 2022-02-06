tokenizer_path=$1
OUTPUT_FOLDER="token_analysis"

langs=("amh" "hau" "swa")
LANGUAGES=("amharic" "hausa" "swahili")

mkdir -p $OUTPUT_FOLDER

for i in ${!langs[@]}
    do
        for j in "${!langs[@]}"
            do
                output_dir="$OUTPUT_FOLDER/$tokenizer_path/${langs[i]}_${langs[j]}_overlap"
                output_dir_2="$OUTPUT_FOLDER/$tokenizer_path/${langs[j]}_${langs[i]}_overlap"

                if [ ${langs[i]} == ${langs[j]} ] || [ -d $output_dir ] || [ -d $output_dir_2 ]
                then
                    echo "Skipping analysis for language pair (${langs[i]}, ${langs[j]})."
                    echo "Analysis may already be calculated or language pair may be same"
                else
                    mkdir -p $output_dir
                    python token_analysis.py \
                    --lang_file_1 data/train/${langs[i]}/"train.${LANGUAGES[i]}" \
                    --lang_file_2 data/train/${langs[j]}/"train.${LANGUAGES[j]}" \
                    --tokenizer_path $tokenizer_path \
                    --output_dir $output_dir
                    --logfile "$output_dir/log.txt" >> "$output_dir/log.txt"
                fi
            done
    done