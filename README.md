## Preprocessing
Data preprocessing for question answering
```
python3 preprocessing_for_qa.py --context_dir /path/to/context.json --data_dir /path/to/data.json --output_dir /path/to/processed-data.json
```
Do the following to process the training and validation data
```
python3 preprocessing_for_qa.py --context_dir ntuadl2023hw1/context.json --data_dir ntuadl2023hw1/train.json --output_dir ntuadl2023hw1/train_qa.json
python3 preprocessing_for_qa.py --context_dir ntuadl2023hw1/context.json --data_dir ntuadl2023hw1/valid.json --output_dir ntuadl2023hw1/valid_qa.json
```

## Training
Model training for multiple-choice
```
python3 run_swag_no_trainer.py \
    --model_name_or_path hfl/chinese-roberta-wwm-ext \
    --train_file ntuadl2023hw1/train.json \
    --validation_file ntuadl2023hw1/valid.json \
    --context_file ntuadl2023hw1/context.json \
    --pad_to_max_length \
    --max_length 512 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --num_warmup_steps 300 \
    --output_dir result_1
```

Model training for question-answering
```
python3 run_qa_no_trainer.py \
    --model_name_or_path hfl/chinese-roberta-wwm-ext \
    --train_file ntuadl2023hw1/train_qa.json \
    --validation_file ntuadl2023hw1/valid_qa.json \
    --max_seq_length 512 \
    --pad_to_max_length \
    --num_train_epochs 2 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --num_warmup_steps 300 \
    --output_dir result_2
```

## Prediction
After data preprocessing and model training, simply run the following shell script
```
bash run.sh ntuadl2023hw1/context.json ntuadl2023hw1/test.json prediction.csv
```
After that, prediction.csv can be submitted to Kaggle 