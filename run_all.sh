#!/bin/bash
#SBATCH --mem=128G 
#SBATCH --cpus-per-task=10
#SBATCH --time=36:00:0 
#SBATCH --gres=gpu:a100:2
#SBATCH -o ./logs/%j.log

export CUDA_AVAILABLE_DEVICES=0,1

# Tasks to perform using this script
tasks=("lm" "ner" "classification")

experiment_name=afriberta_large
vocab_size=20k

if [[ ${tasks[*]} =~ (^|[[:space:]])"lm"($|[[:space:]]) ]]; then
    # Train Masked Language Model
    python main.py --experiment_name $experiment_name --config_path=mlm_configs/afriberta_large.yml
fi

# Evaluate on Named Entity Recognition

ner_model_path="${experiment_name}_ner_model"
tokenizer_path=tokenizers/$vocab_size # afriberta_tokenizer_70k # specify tokenizer path

mkdir -p $PWD/$ner_model_path

cp $PWD/experiments/$experiment_name/pytorch_model.bin $PWD/$ner_model_path/pytorch_model.bin
cp $PWD/experiments/$experiment_name/config.json $PWD/$ner_model_path/config.json

MAX_LENGTH=164
MODEL_PATH=$ner_model_path
BATCH_SIZE=16
NUM_EPOCHS=50
SAVE_STEPS=1000
TOK_PATH=$tokenizer_path
declare -a arr=("amh" "hau" "swa" "ibo" "kin" "lug" "luo" "pcm" "wol" "yor")

if [[ ${tasks[*]} =~ (^|[[:space:]])"ner"($|[[:space:]]) ]]; then
    for SEED in 1 2 3 4 5
        do
            output_dir=ner_results/"${experiment_name}_ner_results_${SEED}"
            mkdir -p $PWD/$output_dir

            for i in "${arr[@]}"
                do
                    OUTPUT_DIR=$PWD/$output_dir/"$i"
                    DATA_DIR=ner_data/"$i"
                    python -m ner_scripts.train_ner --data_dir $DATA_DIR \
                    --model_type nil \
                    --model_name_or_path $MODEL_PATH \
                    --tokenizer_path $TOK_PATH \
                    --output_dir $OUTPUT_DIR \
                    --max_seq_length $MAX_LENGTH \
                    --num_train_epochs $NUM_EPOCHS \
                    --per_gpu_train_batch_size $BATCH_SIZE \
                    --per_gpu_eval_batch_size $BATCH_SIZE \
                    --save_steps $SAVE_STEPS \
                    --seed $SEED \
                    --do_train \
                    --do_eval \
                    --do_predict || rm -r $OUTPUT_DIR

                done
        done

    # Collate results across all seeds
    for i in "${arr[@]}"
        do
            python ner_scripts/collate_results.py --lang "$i" --experiment-name "${experiment_name}_ner_results"
        done
fi


# Evaluate on Text Classification

export PYTHONPATH=$PWD

if [[ ${tasks[*]} =~ (^|[[:space:]])"classification"($|[[:space:]]) ]]; then
    for SEED in 1 2 3 4 5
        do 

            output_dir=classification_results/"${MODEL_PATH}_hausa_${SEED}"
            python classification_scripts/classification_trainer.py --data_dir hausa_classification_data \
            --model_dir $MODEL_PATH \
            --tok_dir $TOK_PATH \
            --output_dir $output_dir \
            --language hausa \
            --seed $SEED \
            --max_seq_length 500 || rm -r $output_dir


            output_dir=classification_results/"${MODEL_PATH}_yoruba_${SEED}"
            python classification_scripts/classification_trainer.py --data_dir yoruba_classification_data \
            --model_dir $MODEL_PATH \
            --tok_dir $TOK_PATH \
            --output_dir $output_dir \
            --language yoruba \
            --seed $SEED \
            --max_seq_length 500 || rm -r $output_dir

        done

    # Collate results across all seeds
    python classification_scripts/collate_results.py \
    --results-dir classification_results \
    --results-file test_results.txt \
    --experiment-name "${MODEL_PATH}_hausa" \
    --n-seeds 5 \
    --output-file results.json

    python classification_scripts/collate_results.py \
    --results-dir classification_results \
    --results-file test_results.txt \
    --experiment-name "${MODEL_PATH}_yoruba" \
    --n-seeds 5 \
    --output-file results.json
fi