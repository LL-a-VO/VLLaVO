
#CUDA_VISIBLE_DEVICES=7 python DG_finetune_llama.py --base_model data/huggingface_model/Llama-2-7b-chat-hf --source_data_paths "../datasets/PACS/art_painting_all_description.csv ../datasets/PACS/photo_all_description.csv ../datasets/PACS/cartoon_all_description.csv ../datasets/PACS/sketch_all_description.csv" --prompt_template_name pacs_demos --output_dir data/huggingface_model/LoRa_PACS_DG/all --num_epochs 2 --cutoff_len 2048 --micro_batch_size 32 --learning_rate 1e-3 --batch_size 128 --train_on_inputs False --wandb_project llama2_PACS_DG --wandb_run_name sketch --dataset_name PACS --max_steps 100
#
#CUDA_VISIBLE_DEVICES=7 python classification_llama.py -t ../datasets/PACS/art_painting_all_description.csv --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name PACS --lora_weights data/huggingface_model/LoRa_PACS_DG/all --prompt_template_name pacs_demos > ../logs/DG/PACS/all_PACS/art.log 2>&1
#
#CUDA_VISIBLE_DEVICES=7 python classification_llama.py -t ../datasets/PACS/sketch_all_description.csv --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name PACS --lora_weights data/huggingface_model/LoRa_PACS_DG/all --prompt_template_name pacs_demos > ../logs/DG/PACS/all_PACS/sketch.log 2>&1
#
#CUDA_VISIBLE_DEVICES=7 python classification_llama.py -t ../datasets/PACS/photo_all_description.csv --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name PACS --lora_weights data/huggingface_model/LoRa_PACS_DG/all --prompt_template_name pacs_demos > ../logs/DG/PACS/all_PACS/photo.log 2>&1
#
#CUDA_VISIBLE_DEVICES=7 python classification_llama.py -t ../datasets/PACS/cartoon_all_description.csv --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name PACS --lora_weights data/huggingface_model/LoRa_PACS_DG/all --prompt_template_name pacs_demos > ../logs/DG/PACS/all_PACS/cartoon.log 2>&1
#
#
#
#CUDA_VISIBLE_DEVICES=7 python classification_llama.py -t ../datasets/Office_home/Ar_description.csv --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name OfficeHome --lora_weights data/huggingface_model/LoRa_PACS_DG/all --prompt_template_name officehome_demos > ../logs/DG/OfficeHome/all_PACS/ar.log 2>&1
#
#CUDA_VISIBLE_DEVICES=7 python classification_llama.py -t ../datasets/Office_home/Cl_description.csv --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name OfficeHome --lora_weights data/huggingface_model/LoRa_PACS_DG/all --prompt_template_name officehome_demos > ../logs/DG/OfficeHome/all_PACS/cl.log 2>&1
#
#CUDA_VISIBLE_DEVICES=7 python classification_llama.py -t ../datasets/Office_home/Rw_description.csv --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name OfficeHome --lora_weights data/huggingface_model/LoRa_PACS_DG/all --prompt_template_name officehome_demos > ../logs/DG/OfficeHome/all_PACS/rw.log 2>&1
#
#CUDA_VISIBLE_DEVICES=7 python classification_llama.py -t ../datasets/Office_home/Pr_description.csv --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name OfficeHome --lora_weights data/huggingface_model/LoRa_PACS_DG/all --prompt_template_name officehome_demos > ../logs/DG/OfficeHome/all_PACS/pr.log 2>&1
#
#
#CUDA_VISIBLE_DEVICES=7 python classification_llama.py -t ../datasets/VLCS/VOC2007_description.csv --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name VLCS --lora_weights data/huggingface_model/LoRa_PACS_DG/all --prompt_template_name vlcs_demos > ../logs/DG/VLCS/all_PACS/VOC2007.log 2>&1
#
#CUDA_VISIBLE_DEVICES=7 python classification_llama.py -t ../datasets/VLCS/Caltech101_description.csv --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name VLCS --lora_weights data/huggingface_model/LoRa_PACS_DG/all --prompt_template_name vlcs_demos > ../logs/DG/VLCS/all_PACS/Caltech101.log 2>&1
#
#CUDA_VISIBLE_DEVICES=7 python classification_llama.py -t ../datasets/VLCS/SUN09_description.csv --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name VLCS --lora_weights data/huggingface_model/LoRa_PACS_DG/all --prompt_template_name vlcs_demos > ../logs/DG/VLCS/all_PACS/SUN09.log 2>&1
#
#CUDA_VISIBLE_DEVICES=7 python classification_llama.py -t ../datasets/VLCS/LabelMe_description.csv --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name VLCS --lora_weights data/huggingface_model/LoRa_PACS_DG/all --prompt_template_name vlcs_demos > ../logs/DG/VLCS/all_PACS/LabelMe.log 2>&1
#
#
#
#
#CUDA_VISIBLE_DEVICES=7 python classification_llama.py -t ../datasets/DomainNet/clipart_test_description.csv ../datasets/DomainNet/clipart_train_description.csv --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name DomainNet --lora_weights data/huggingface_model/LoRa_PACS_DG/all --prompt_template_name domainnet_demos > ../logs/DG/DomainNet/all_PACS/clipart.log 2>&1
#
#CUDA_VISIBLE_DEVICES=7 python classification_llama.py -t ../datasets/DomainNet/painting_test_description.csv ../datasets/DomainNet/painting_train_description.csv --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name DomainNet --lora_weights data/huggingface_model/LoRa_PACS_DG/all --prompt_template_name domainnet_demos > ../logs/DG/DomainNet/all_PACS/painting.log 2>&1
#
#CUDA_VISIBLE_DEVICES=7 python classification_llama.py -t ../datasets/DomainNet/infograph_test_description.csv ../datasets/DomainNet/infograph_train_description.csv --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name DomainNet --lora_weights data/huggingface_model/LoRa_PACS_DG/all --prompt_template_name domainnet_demos > ../logs/DG/DomainNet/all_PACS/infograph.log 2>&1
#
#CUDA_VISIBLE_DEVICES=7 python classification_llama.py -t ../datasets/DomainNet/sketch_test_description.csv ../datasets/DomainNet/sketch_train_description.csv --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name DomainNet --lora_weights data/huggingface_model/LoRa_PACS_DG/all --prompt_template_name domainnet_demos > ../logs/DG/DomainNet/all_PACS/sketch.log 2>&1
#
#CUDA_VISIBLE_DEVICES=7 python classification_llama.py -t ../datasets/DomainNet/quickdraw_test_description.csv ../datasets/DomainNet/quickdraw_train_description.csv --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name DomainNet --lora_weights data/huggingface_model/LoRa_PACS_DG/all --prompt_template_name domainnet_demos > ../logs/DG/DomainNet/all_PACS/quickdraw.log 2>&1
#
#CUDA_VISIBLE_DEVICES=0 python classification_llama.py -t ../datasets/DomainNet/real_test_description.csv ../datasets/DomainNet/real_train_description.csv --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name DomainNet --lora_weights data/huggingface_model/LoRa_PACS_DG/all --prompt_template_name domainnet_demos > ../logs/DG/DomainNet/all_PACS/real.log 2>&1
#


CUDA_VISIBLE_DEVICES=0 python classification_llama.py -t ../datasets/DomainNet/clipart_test_description.csv ../datasets/DomainNet/clipart_train_description.csv --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name subDomainNet --lora_weights data/huggingface_model/LoRa_PACS_DG/all --prompt_template_name subdomainnet_demos > ../logs/DG/DomainNet/all_PACS/new/clipart.log 2>&1

CUDA_VISIBLE_DEVICES=4 nohup python classification_llama.py -t ../datasets/DomainNet/painting_test_description.csv ../datasets/DomainNet/painting_train_description.csv --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name subDomainNet --lora_weights data/huggingface_model/LoRa_PACS_DG/all --prompt_template_name subdomainnet_demos > ../logs/DG/DomainNet/all_PACS/new/painting.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 python classification_llama.py -t ../datasets/DomainNet/infograph_test_description.csv ../datasets/DomainNet/infograph_train_description.csv --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name subDomainNet --lora_weights data/huggingface_model/LoRa_PACS_DG/all --prompt_template_name subdomainnet_demos > ../logs/DG/DomainNet/all_PACS/new/infograph.log 2>&1

CUDA_VISIBLE_DEVICES=0 python classification_llama.py -t ../datasets/DomainNet/sketch_test_description.csv ../datasets/DomainNet/sketch_train_description.csv --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name subDomainNet --lora_weights data/huggingface_model/LoRa_PACS_DG/all --prompt_template_name subdomainnet_demos > ../logs/DG/DomainNet/all_PACS/new/sketch.log 2>&1

CUDA_VISIBLE_DEVICES=0 python classification_llama.py -t ../datasets/DomainNet/quickdraw_test_description.csv ../datasets/DomainNet/quickdraw_train_description.csv --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name subDomainNet --lora_weights data/huggingface_model/LoRa_PACS_DG/all --prompt_template_name subdomainnet_demos > ../logs/DG/DomainNet/all_PACS/new/quickdraw.log 2>&1

CUDA_VISIBLE_DEVICES=0 python classification_llama.py -t ../datasets/DomainNet/real_test_description.csv ../datasets/DomainNet/real_train_description.csv --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name subDomainNet --lora_weights data/huggingface_model/LoRa_PACS_DG/all --prompt_template_name subdomainnet_demos > ../logs/DG/DomainNet/all_PACS/new/real.log 2>&1