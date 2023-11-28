
#CUDA_VISIBLE_DEVICES=1 python finetune_T5.py -s ../datasets/Office_home/Ar_description.csv -o data/huggingface_model/T5_OfficeHome_UDA/Ar -p officehome_demos --dataset_name OfficeHome
#
#CUDA_VISIBLE_DEVICES=1 python finetune_T5.py -s ../datasets/Office_home/Cl_description.csv -o data/huggingface_model/T5_OfficeHome_UDA/Cl -p officehome_demos --dataset_name OfficeHome
#
#CUDA_VISIBLE_DEVICES=1 nohup python finetune_T5.py -s ../datasets/Office_home/Pr_description.csv -o data/huggingface_model/T5_OfficeHome_UDA/Pr -p officehome_demos --dataset_name OfficeHome &

#CUDA_VISIBLE_DEVICES=1 python finetune_T5.py -s ../datasets/Office_home/Rw_description.csv -o data/huggingface_model/T5_OfficeHome_UDA/Rw -p officehome_demos --dataset_name OfficeHome

#CUDA_VISIBLE_DEVICES=1 python classification_t5.py -s ../datasets/Office_home/Ar_description.csv -t ../datasets/Office_home/Cl_description.csv --base_model data/huggingface_model/T5_OfficeHome_UDA/Ar/simple-t5-best --dataset_name OfficeHome --prompt_template_name officehome_demos --save_path ../datasets/Office_home/t5_pseudo_labels/Ar2Cl.csv > ../logs/OfficeHome/t5_step1/ar2cl.log 2>&1
#
#CUDA_VISIBLE_DEVICES=1 python classification_t5.py -s ../datasets/Office_home/Ar_description.csv -t ../datasets/Office_home/Pr_description.csv --base_model data/huggingface_model/T5_OfficeHome_UDA/Ar/simple-t5-best --dataset_name OfficeHome --prompt_template_name officehome_demos --save_path ../datasets/Office_home/t5_pseudo_labels/Ar2Pr.csv > ../logs/OfficeHome/t5_step1/ar2pr.log 2>&1
#
#CUDA_VISIBLE_DEVICES=1 python classification_t5.py -s ../datasets/Office_home/Ar_description.csv -t ../datasets/Office_home/Rw_description.csv --base_model data/huggingface_model/T5_OfficeHome_UDA/Ar/simple-t5-best --dataset_name OfficeHome --prompt_template_name officehome_demos --save_path ../datasets/Office_home/t5_pseudo_labels/Ar2Rw.csv > ../logs/OfficeHome/t5_step1/ar2rw.log 2>&1

#CUDA_VISIBLE_DEVICES=1 python classification_t5.py -s ../datasets/Office_home/Cl_description.csv -t ../datasets/Office_home/Ar_description.csv --base_model data/huggingface_model/T5_OfficeHome_UDA/Cl/simple-t5-best --dataset_name OfficeHome --prompt_template_name officehome_demos --save_path ../datasets/Office_home/t5_pseudo_labels/Cl2Ar.csv > ../logs/OfficeHome/t5_step1/cl2ar.log 2>&1
#
#CUDA_VISIBLE_DEVICES=1 python classification_t5.py -s ../datasets/Office_home/Cl_description.csv -t ../datasets/Office_home/Pr_description.csv --base_model data/huggingface_model/T5_OfficeHome_UDA/Cl/simple-t5-best --dataset_name OfficeHome --prompt_template_name officehome_demos --save_path ../datasets/Office_home/t5_pseudo_labels/Cl2Pr.csv > ../logs/OfficeHome/t5_step1/cl2pr.log 2>&1
#
#CUDA_VISIBLE_DEVICES=1 python classification_t5.py -s ../datasets/Office_home/Cl_description.csv -t ../datasets/Office_home/Rw_description.csv --base_model data/huggingface_model/T5_OfficeHome_UDA/Cl/simple-t5-best --dataset_name OfficeHome --prompt_template_name officehome_demos --save_path ../datasets/Office_home/t5_pseudo_labels/Cl2Rw.csv > ../logs/OfficeHome/t5_step1/cl2rw.log 2>&1
#
#CUDA_VISIBLE_DEVICES=1 python classification_t5.py -s ../datasets/Office_home/Pr_description.csv -t ../datasets/Office_home/Ar_description.csv --base_model data/huggingface_model/T5_OfficeHome_UDA/Pr/simple-t5-best --dataset_name OfficeHome --prompt_template_name officehome_demos --save_path ../datasets/Office_home/t5_pseudo_labels/Pr2Ar.csv > ../logs/OfficeHome/t5_step1/pr2ar.log 2>&1
#
#CUDA_VISIBLE_DEVICES=1 python classification_t5.py -s ../datasets/Office_home/Pr_description.csv -t ../datasets/Office_home/Cl_description.csv --base_model data/huggingface_model/T5_OfficeHome_UDA/Pr/simple-t5-best --dataset_name OfficeHome --prompt_template_name officehome_demos --save_path ../datasets/Office_home/t5_pseudo_labels/Pr2Cl.csv > ../logs/OfficeHome/t5_step1/pr2cl.log 2>&1
#
#CUDA_VISIBLE_DEVICES=1 python classification_t5.py -s ../datasets/Office_home/Pr_description.csv -t ../datasets/Office_home/Rw_description.csv --base_model data/huggingface_model/T5_OfficeHome_UDA/Pr/simple-t5-best --dataset_name OfficeHome --prompt_template_name officehome_demos --save_path ../datasets/Office_home/t5_pseudo_labels/Pr2Rw.csv > ../logs/OfficeHome/t5_step1/pr2rw.log 2>&1
#
#CUDA_VISIBLE_DEVICES=1 python classification_t5.py -s ../datasets/Office_home/Rw_description.csv -t ../datasets/Office_home/Ar_description.csv --base_model data/huggingface_model/T5_OfficeHome_UDA/Rw/simple-t5-best --dataset_name OfficeHome --prompt_template_name officehome_demos --save_path ../datasets/Office_home/t5_pseudo_labels/Rw2Ar.csv > ../logs/OfficeHome/t5_step1/rw2ar.log 2>&1
#
#CUDA_VISIBLE_DEVICES=1 python classification_t5.py -s ../datasets/Office_home/Rw_description.csv -t ../datasets/Office_home/Cl_description.csv --base_model data/huggingface_model/T5_OfficeHome_UDA/Rw/simple-t5-best --dataset_name OfficeHome --prompt_template_name officehome_demos --save_path ../datasets/Office_home/t5_pseudo_labels/Rw2Cl.csv > ../logs/OfficeHome/t5_step1/rw2cl.log 2>&1
#
#CUDA_VISIBLE_DEVICES=1 python classification_t5.py -s ../datasets/Office_home/Rw_description.csv -t ../datasets/Office_home/Pr_description.csv --base_model data/huggingface_model/T5_OfficeHome_UDA/Rw/simple-t5-best --dataset_name OfficeHome --prompt_template_name officehome_demos --save_path ../datasets/Office_home/t5_pseudo_labels/Rw2Pr.csv > ../logs/OfficeHome/t5_step1/rw2pr.log 2>&1
#
#CUDA_VISIBLE_DEVICES=1 python finetune_T5.py -s ../datasets/Office_home/Ar_description.csv -t ../datasets/Office_home/t5_pseudo_labels/Ar2Cl.csv -o data/huggingface_model/T5_OfficeHome_UDA/Ar2Cl -p officehome_demos --dataset_name OfficeHome
#
#CUDA_VISIBLE_DEVICES=1 python finetune_T5.py -s ../datasets/Office_home/Ar_description.csv -t ../datasets/Office_home/t5_pseudo_labels/Ar2Pr.csv -o data/huggingface_model/T5_OfficeHome_UDA/Ar2Pr -p officehome_demos --dataset_name OfficeHome
#
#CUDA_VISIBLE_DEVICES=1 python finetune_T5.py -s ../datasets/Office_home/Ar_description.csv -t ../datasets/Office_home/t5_pseudo_labels/Ar2Rw.csv -o data/huggingface_model/T5_OfficeHome_UDA/Ar2Rw -p officehome_demos --dataset_name OfficeHome
#
#CUDA_VISIBLE_DEVICES=1 python finetune_T5.py -s ../datasets/Office_home/Cl_description.csv -t ../datasets/Office_home/t5_pseudo_labels/Cl2Ar.csv -o data/huggingface_model/T5_OfficeHome_UDA/Cl2Ar -p officehome_demos --dataset_name OfficeHome
#
#CUDA_VISIBLE_DEVICES=1 python finetune_T5.py -s ../datasets/Office_home/Cl_description.csv -t ../datasets/Office_home/t5_pseudo_labels/Cl2Pr.csv -o data/huggingface_model/T5_OfficeHome_UDA/Cl2Pr -p officehome_demos --dataset_name OfficeHome
#
#CUDA_VISIBLE_DEVICES=1 python finetune_T5.py -s ../datasets/Office_home/Cl_description.csv -t ../datasets/Office_home/t5_pseudo_labels/Cl2Rw.csv -o data/huggingface_model/T5_OfficeHome_UDA/Cl2Rw -p officehome_demos --dataset_name OfficeHome
#
#CUDA_VISIBLE_DEVICES=1 python finetune_T5.py -s ../datasets/Office_home/Pr_description.csv -t ../datasets/Office_home/t5_pseudo_labels/Pr2Ar.csv -o data/huggingface_model/T5_OfficeHome_UDA/Pr2Ar -p officehome_demos --dataset_name OfficeHome
#
#CUDA_VISIBLE_DEVICES=1 python finetune_T5.py -s ../datasets/Office_home/Pr_description.csv -t ../datasets/Office_home/t5_pseudo_labels/Pr2Cl.csv -o data/huggingface_model/T5_OfficeHome_UDA/Pr2Cl -p officehome_demos --dataset_name OfficeHome
#
#CUDA_VISIBLE_DEVICES=1 python finetune_T5.py -s ../datasets/Office_home/Pr_description.csv -t ../datasets/Office_home/t5_pseudo_labels/Pr2Rw.csv -o data/huggingface_model/T5_OfficeHome_UDA/Pr2Rw -p officehome_demos --dataset_name OfficeHome
#
#CUDA_VISIBLE_DEVICES=1 python finetune_T5.py -s ../datasets/Office_home/Rw_description.csv -t ../datasets/Office_home/t5_pseudo_labels/Rw2Ar.csv -o data/huggingface_model/T5_OfficeHome_UDA/Rw2Ar -p officehome_demos --dataset_name OfficeHome
#
#CUDA_VISIBLE_DEVICES=1 python finetune_T5.py -s ../datasets/Office_home/Rw_description.csv -t ../datasets/Office_home/t5_pseudo_labels/Rw2Cl.csv -o data/huggingface_model/T5_OfficeHome_UDA/Rw2Cl -p officehome_demos --dataset_name OfficeHome
#
#CUDA_VISIBLE_DEVICES=1 python finetune_T5.py -s ../datasets/Office_home/Rw_description.csv -t ../datasets/Office_home/t5_pseudo_labels/Rw2Pr.csv -o data/huggingface_model/T5_OfficeHome_UDA/Rw2Pr -p officehome_demos --dataset_name OfficeHome



CUDA_VISIBLE_DEVICES=1 python classification_t5.py -s ../datasets/Office_home/Ar_description.csv -t ../datasets/Office_home/Cl_description.csv --base_model data/huggingface_model/T5_OfficeHome_UDA/Ar2Cl/simple-t5-best --dataset_name OfficeHome --prompt_template_name officehome_demos > ../logs/OfficeHome/t5_step2/ar2cl.log 2>&1

CUDA_VISIBLE_DEVICES=1 python classification_t5.py -s ../datasets/Office_home/Ar_description.csv -t ../datasets/Office_home/Pr_description.csv --base_model data/huggingface_model/T5_OfficeHome_UDA/Ar2Pr/simple-t5-best --dataset_name OfficeHome --prompt_template_name officehome_demos > ../logs/OfficeHome/t5_step2/ar2pr.log 2>&1

CUDA_VISIBLE_DEVICES=1 python classification_t5.py -s ../datasets/Office_home/Ar_description.csv -t ../datasets/Office_home/Rw_description.csv --base_model data/huggingface_model/T5_OfficeHome_UDA/Ar2Rw/simple-t5-best --dataset_name OfficeHome --prompt_template_name officehome_demos > ../logs/OfficeHome/t5_step2/ar2rw.log 2>&1

CUDA_VISIBLE_DEVICES=1 python classification_t5.py -s ../datasets/Office_home/Cl_description.csv -t ../datasets/Office_home/Ar_description.csv --base_model data/huggingface_model/T5_OfficeHome_UDA/Cl2Ar/simple-t5-best --dataset_name OfficeHome --prompt_template_name officehome_demos > ../logs/OfficeHome/t5_step2/cl2ar.log 2>&1

CUDA_VISIBLE_DEVICES=1 python classification_t5.py -s ../datasets/Office_home/Cl_description.csv -t ../datasets/Office_home/Pr_description.csv --base_model data/huggingface_model/T5_OfficeHome_UDA/Cl2Pr/simple-t5-best --dataset_name OfficeHome --prompt_template_name officehome_demos > ../logs/OfficeHome/t5_step2/cl2pr.log 2>&1

CUDA_VISIBLE_DEVICES=1 python classification_t5.py -s ../datasets/Office_home/Cl_description.csv -t ../datasets/Office_home/Rw_description.csv --base_model data/huggingface_model/T5_OfficeHome_UDA/Cl2Rw/simple-t5-best --dataset_name OfficeHome --prompt_template_name officehome_demos > ../logs/OfficeHome/t5_step2/cl2rw.log 2>&1

CUDA_VISIBLE_DEVICES=1 python classification_t5.py -s ../datasets/Office_home/Pr_description.csv -t ../datasets/Office_home/Ar_description.csv --base_model data/huggingface_model/T5_OfficeHome_UDA/Pr2Ar/simple-t5-best --dataset_name OfficeHome --prompt_template_name officehome_demos > ../logs/OfficeHome/t5_step2/pr2ar.log 2>&1

CUDA_VISIBLE_DEVICES=1 python classification_t5.py -s ../datasets/Office_home/Pr_description.csv -t ../datasets/Office_home/Cl_description.csv --base_model data/huggingface_model/T5_OfficeHome_UDA/Pr2Cl/simple-t5-best --dataset_name OfficeHome --prompt_template_name officehome_demos > ../logs/OfficeHome/t5_step2/pr2cl.log 2>&1

CUDA_VISIBLE_DEVICES=1 python classification_t5.py -s ../datasets/Office_home/Pr_description.csv -t ../datasets/Office_home/Rw_description.csv --base_model data/huggingface_model/T5_OfficeHome_UDA/Pr2Rw/simple-t5-best --dataset_name OfficeHome --prompt_template_name officehome_demos > ../logs/OfficeHome/t5_step2/pr2rw.log 2>&1

CUDA_VISIBLE_DEVICES=1 python classification_t5.py -s ../datasets/Office_home/Rw_description.csv -t ../datasets/Office_home/Ar_description.csv --base_model data/huggingface_model/T5_OfficeHome_UDA/Rw2Ar/simple-t5-best --dataset_name OfficeHome --prompt_template_name officehome_demos > ../logs/OfficeHome/t5_step2/rw2ar.log 2>&1

CUDA_VISIBLE_DEVICES=1 python classification_t5.py -s ../datasets/Office_home/Rw_description.csv -t ../datasets/Office_home/Cl_description.csv --base_model data/huggingface_model/T5_OfficeHome_UDA/Rw2Cl/simple-t5-best --dataset_name OfficeHome --prompt_template_name officehome_demos > ../logs/OfficeHome/t5_step2/rw2cl.log 2>&1

CUDA_VISIBLE_DEVICES=1 python classification_t5.py -s ../datasets/Office_home/Rw_description.csv -t ../datasets/Office_home/Pr_description.csv --base_model data/huggingface_model/T5_OfficeHome_UDA/Rw2Pr/simple-t5-best --dataset_name OfficeHome --prompt_template_name officehome_demos > ../logs/OfficeHome/t5_step2/rw2pr.log 2>&1