
CUDA_VISIBLE_DEVICES=4 nohup python classification_llama.py -t ../datasets/PACS/art_painting_all_description.csv --base_model llama_model_base_path --dataset_name PACS --lora_weights output_path/art --prompt_template_name pacs_demos > ../logs/DG/PACS/art.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 nohup python classification_llama.py -t ../datasets/PACS/sketch_all_description.csv --base_model llama_model_base_path --dataset_name PACS --lora_weights output_path/sketch --prompt_template_name pacs_demos > ../logs/DG/PACS/sketch.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 nohup python classification_llama.py -t ../datasets/PACS/photo_all_description.csv --base_model llama_model_base_path --dataset_name PACS --lora_weights output_path/photo --prompt_template_name pacs_demos > ../logs/DG/PACS/photo.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 nohup python classification_llama.py -t ../datasets/PACS/cartoon_all_description.csv --base_model llama_model_base_path --dataset_name PACS --lora_weights output_path/cartoon --prompt_template_name pacs_demos > ../logs/DG/PACS/cartoon.log 2>&1 &