#CUDA_VISIBLE_DEVICES=5 python finetune_T5.py -s ../datasets/Office_home/Ar_description.csv ../datasets/Office_home/Cl_description.csv ../datasets/Office_home/Pr_description.csv -o data/huggingface_model/T5_OfficeHome/Rw -p officehome_demos --dataset_name OfficeHome
#
#CUDA_VISIBLE_DEVICES=5 python classification_t5.py -t ../datasets/Office_home/Rw_description.csv --dataset_name OfficeHome --base_model data/huggingface_model/T5_OfficeHome/Rw/simple-t5-best --prompt_template_name officehome_demos > ../logs/DG/OfficeHome/t5/rw.log 2>&1
#
#CUDA_VISIBLE_DEVICES=5 python finetune_T5.py -s ../datasets/Office_home/Ar_description.csv ../datasets/Office_home/Rw_description.csv ../datasets/Office_home/Pr_description.csv -o data/huggingface_model/T5_OfficeHome/Cl -p officehome_demos --dataset_name OfficeHome
#
#CUDA_VISIBLE_DEVICES=5 python classification_t5.py -t ../datasets/Office_home/Cl_description.csv --dataset_name OfficeHome --base_model data/huggingface_model/T5_OfficeHome/Cl/simple-t5-best --prompt_template_name officehome_demos > ../logs/DG/OfficeHome/t5/cl.log 2>&1
#
#CUDA_VISIBLE_DEVICES=5 python finetune_T5.py -s ../datasets/Office_home/Ar_description.csv ../datasets/Office_home/Cl_description.csv ../datasets/Office_home/Rw_description.csv -o data/huggingface_model/T5_OfficeHome/Pr -p officehome_demos --dataset_name OfficeHome
#
#CUDA_VISIBLE_DEVICES=5 python classification_t5.py -t ../datasets/Office_home/Pr_description.csv --dataset_name OfficeHome --base_model data/huggingface_model/T5_OfficeHome/Pr/simple-t5-best --prompt_template_name officehome_demos > ../logs/DG/OfficeHome/t5/pr.log 2>&1

#CUDA_VISIBLE_DEVICES=5 python finetune_T5.py -s ../datasets/Office_home/Rw_description.csv ../datasets/Office_home/Cl_description.csv ../datasets/Office_home/Pr_description.csv -o data/huggingface_model/T5_OfficeHome/Ar -p officehome_demos --dataset_name OfficeHome
#
#CUDA_VISIBLE_DEVICES=5 python classification_t5.py -t ../datasets/Office_home/Ar_description.csv --dataset_name OfficeHome --base_model data/huggingface_model/T5_OfficeHome/Ar/simple-t5-best --prompt_template_name officehome_demos > ../logs/DG/OfficeHome/t5/ar.log 2>&1


CUDA_VISIBLE_DEVICES=3 python classification_llama.py -t ../datasets/Office_home/Ar_description.csv --base_model data/huggingface_model/llama-7b --dataset_name OfficeHome --lora_weights data/huggingface_model/LoRa_llama1_Office_home_DG/Ar --prompt_template_name officehome_demos > ../logs/DG/LLama1_OfficHome/ar.log 2>&1

CUDA_VISIBLE_DEVICES=3 python classification_llama.py -t ../datasets/Office_home/Cl_description.csv --base_model data/huggingface_model/llama-7b --dataset_name OfficeHome --lora_weights data/huggingface_model/LoRa_llama1_Office_home_DG/Cl --prompt_template_name officehome_demos > ../logs/DG/LLama1_OfficHome/cl.log 2>&1

CUDA_VISIBLE_DEVICES=3 python classification_llama.py -t ../datasets/Office_home/Rw_description.csv --base_model data/huggingface_model/llama-7b --dataset_name OfficeHome --lora_weights data/huggingface_model/LoRa_llama1_Office_home_DG/Rw --prompt_template_name officehome_demos > ../logs/DG/LLama1_OfficHome/rw.log 2>&1

CUDA_VISIBLE_DEVICES=3 python classification_llama.py -t ../datasets/Office_home/Pr_description.csv --base_model data/huggingface_model/llama-7b --dataset_name OfficeHome --lora_weights data/huggingface_model/LoRa_llama1_Office_home_DG/Pr --prompt_template_name officehome_demos > ../logs/DG/LLama1_OfficHome/pr.log 2>&1