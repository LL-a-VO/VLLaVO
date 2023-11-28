#CUDA_VISIBLE_DEVICES=4 python zeroShot_llama.py  -t ../datasets/PACS/art_painting_all_description.csv  --prompt_template_name pacs_demos --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name PACS > ../logs/DG/PACS/zeroShot/new/art.log 2>&1
#
#CUDA_VISIBLE_DEVICES=4 python zeroShot_llama.py  -t ../datasets/PACS/photo_all_description.csv --prompt_template_name pacs_demos --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name PACS > ../logs/DG/PACS/zeroShot/new/photo.log 2>&1
#
#CUDA_VISIBLE_DEVICES=4 python zeroShot_llama.py  -t ../datasets/PACS/cartoon_all_description.csv --prompt_template_name pacs_demos --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name PACS > ../logs/DG/PACS/zeroShot/new/cartoon.log 2>&1
#
#CUDA_VISIBLE_DEVICES=4 python zeroShot_llama.py  -t ../datasets/PACS/sketch_all_description.csv --prompt_template_name pacs_demos --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name PACS > ../logs/DG/PACS/zeroShot/new/sketch.log 2>&1


#CUDA_VISIBLE_DEVICES=4 python zeroShot_llama.py -t ../datasets/Office_home/Ar_description.csv --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name OfficeHome --prompt_template_name officehome_demos > ../logs/DG/OfficeHome/zeroShot/ar.log 2>&1
#
#CUDA_VISIBLE_DEVICES=4 python zeroShot_llama.py -t ../datasets/Office_home/Cl_description.csv --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name OfficeHome --prompt_template_name officehome_demos > ../logs/DG/OfficeHome/zeroShot/cl.log 2>&1
#
#CUDA_VISIBLE_DEVICES=4 python zeroShot_llama.py -t ../datasets/Office_home/Rw_description.csv --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name OfficeHome --prompt_template_name officehome_demos > ../logs/DG/OfficeHome/zeroShot/rw.log 2>&1
#
#CUDA_VISIBLE_DEVICES=4 python zeroShot_llama.py -t ../datasets/Office_home/Pr_description.csv --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name OfficeHome --prompt_template_name officehome_demos > ../logs/DG/OfficeHome/zeroShot/pr.log 2>&1
##
##
##
#CUDA_VISIBLE_DEVICES=4 python zeroShot_llama.py -t ../datasets/VLCS/VOC2007_description.csv --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name VLCS --prompt_template_name vlcs_demos > ../logs/DG/VLCS/zeroShot/VOC2007.log 2>&1
#
#CUDA_VISIBLE_DEVICES=4 python zeroShot_llama.py -t ../datasets/VLCS/Caltech101_description.csv --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name VLCS --prompt_template_name vlcs_demos > ../logs/DG/VLCS/zeroShot/Caltech101.log 2>&1
#
#CUDA_VISIBLE_DEVICES=4 python zeroShot_llama.py -t ../datasets/VLCS/SUN09_description.csv --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name VLCS --prompt_template_name vlcs_demos > ../logs/DG/VLCS/zeroShot/SUN09.log 2>&1
#
#CUDA_VISIBLE_DEVICES=4 python zeroShot_llama.py -t ../datasets/VLCS/LabelMe_description.csv --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name VLCS --prompt_template_name vlcs_demos > ../logs/DG/VLCS/zeroShot/LabelMe.log 2>&1


CUDA_VISIBLE_DEVICES=4 python zeroShot_llama.py -t ../datasets/DomainNet/clipart_test_description.csv ../datasets/DomainNet/clipart_train_description.csv --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name subDomainNet --prompt_template_name subdomainnet_demos > ../logs/DG/DomainNet/zeroShot/clipart.log 2>&1

CUDA_VISIBLE_DEVICES=4 python zeroShot_llama.py -t ../datasets/DomainNet/painting_test_description.csv ../datasets/DomainNet/painting_train_description.csv --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name subDomainNet --prompt_template_name subdomainnet_demos > ../logs/DG/DomainNet/zeroShot/painting.log 2>&1

CUDA_VISIBLE_DEVICES=4 python zeroShot_llama.py -t ../datasets/DomainNet/infograph_test_description.csv ../datasets/DomainNet/infograph_train_description.csv --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name subDomainNet --prompt_template_name subdomainnet_demos > ../logs/DG/DomainNet/zeroShot/infograph.log 2>&1

CUDA_VISIBLE_DEVICES=4 python zeroShot_llama.py -t ../datasets/DomainNet/sketch_test_description.csv ../datasets/DomainNet/sketch_train_description.csv --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name subDomainNet --prompt_template_name subdomainnet_demos > ../logs/DG/DomainNet/zeroShot/sketch.log 2>&1

CUDA_VISIBLE_DEVICES=4 python zeroShot_llama.py -t ../datasets/DomainNet/quickdraw_test_description.csv ../datasets/DomainNet/quickdraw_train_description.csv --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name subDomainNet --prompt_template_name subdomainnet_demos > ../logs/DG/DomainNet/zeroShot/quickdraw.log 2>&1

CUDA_VISIBLE_DEVICES=4 python zeroShot_llama.py -t ../datasets/DomainNet/real_test_description.csv ../datasets/DomainNet/real_train_description.csv --base_model data/huggingface_model/Llama-2-7b-chat-hf --dataset_name subDomainNet --prompt_template_name subdomainnet_demos > ../logs/DG/DomainNet/zeroShot/real.log 2>&1