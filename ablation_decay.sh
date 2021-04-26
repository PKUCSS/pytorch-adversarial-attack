python3 black_box_test.py --attack mi_fgsm --step_num 10 --step_size 0.01 --epsilon 0.031 --alpha 0.01 --log_file ./log/ablation_decay.log  --dataset cifar10 --model preActResnet18 --model_path ./saved_model/CIFAR10_PreActResNet18.checkpoint --sub_model preActResnet18 --sub_model_path ./saved_model/sub_preact_resnet18/model.pt ;
python3 black_box_test.py --attack mi_fgsm --step_num 10 --step_size 0.01 --epsilon 0.031 --alpha 0.025 --log_file ./log/ablation_decay.log --dataset cifar10 --model preActResnet18 --model_path ./saved_model/CIFAR10_PreActResNet18.checkpoint --sub_model preActResnet18 --sub_model_path ./saved_model/sub_preact_resnet18/model.pt;
python3 black_box_test.py --attack mi_fgsm --step_num 10 --step_size 0.01 --epsilon 0.031 --alpha 0.05 -log_file ./log/ablation_decay.log --dataset cifar10 --model preActResnet18 --model_path ./saved_model/CIFAR10_PreActResNet18.checkpoint --sub_model preActResnet18 --sub_model_path ./saved_model/sub_preact_resnet18/model.pt;
python3 black_box_test.py --attack mi_fgsm --step_num 10 --step_size 0.01 --epsilon 0.031 --alpha 0.075 --log_file ./log/ablation_decay.log --dataset cifar10 --model preActResnet18 --model_path ./saved_model/CIFAR10_PreActResNet18.checkpoint --sub_model preActResnet18 --sub_model_path ./saved_model/sub_preact_resnet18/model.pt;
python3 black_box_test.py --attack mi_fgsm --step_num 10 --step_size 0.01 --epsilon 0.031 --alpha 0.1 --log_file ./log/ablation_decay.log --dataset cifar10 --model preActResnet18 --model_path ./saved_model/CIFAR10_PreActResNet18.checkpoint --sub_model preActResnet18 --sub_model_path ./saved_model/sub_preact_resnet18/model.pt;
python3 black_box_test.py --attack mi_fgsm --step_num 10 --step_size 0.01 --epsilon 0.031 --alpha 0.2 --log_file ./log/ablation_decay.log --dataset cifar10 --model preActResnet18 --model_path ./saved_model/CIFAR10_PreActResNet18.checkpoint --sub_model preActResnet18 --sub_model_path ./saved_model/sub_preact_resnet18/model.pt;
python3 black_box_test.py --attack mi_fgsm --step_num 10 --step_size 0.01 --epsilon 0.031 --alpha 0.3 --log_file ./log/ablation_decay.log --dataset cifar10 --model preActResnet18 --model_path ./saved_model/CIFAR10_PreActResNet18.checkpoint --sub_model preActResnet18 --sub_model_path ./saved_model/sub_preact_resnet18/model.pt;
python3 black_box_test.py --attack mi_fgsm --step_num 10 --step_size 0.01 --epsilon 0.031 --alpha 0.4 --log_file ./log/ablation_decay.log --dataset cifar10 --model preActResnet18 --model_path ./saved_model/CIFAR10_PreActResNet18.checkpoint --sub_model preActResnet18 --sub_model_path ./saved_model/sub_preact_resnet18/model.pt;
python3 black_box_test.py --attack mi_fgsm --step_num 10 --step_size 0.01 --epsilon 0.031 --alpha 0.5 --log_file ./log/ablation_decay.log --dataset cifar10 --model preActResnet18 --model_path ./saved_model/CIFAR10_PreActResNet18.checkpoint --sub_model preActResnet18 --sub_model_path ./saved_model/sub_preact_resnet18/model.pt;
python3 black_box_test.py --attack mi_fgsm --step_num 10 --step_size 0.01 --epsilon 0.031 --alpha 0.6 --log_file ./log/ablation_decay.log --dataset cifar10 --model preActResnet18 --model_path ./saved_model/CIFAR10_PreActResNet18.checkpoint --sub_model preActResnet18 --sub_model_path ./saved_model/sub_preact_resnet18/model.pt;
python3 black_box_test.py --attack mi_fgsm --step_num 10 --step_size 0.01 --epsilon 0.031 --alpha 0.7 --log_file ./log/ablation_decay.log --dataset cifar10 --model preActResnet18 --model_path ./saved_model/CIFAR10_PreActResNet18.checkpoint --sub_model preActResnet18 --sub_model_path ./saved_model/sub_preact_resnet18/model.pt;
python3 black_box_test.py --attack mi_fgsm --step_num 10 --step_size 0.01 --epsilon 0.031 --alpha 0.8 --log_file ./log/ablation_decay.log --dataset cifar10 --model preActResnet18 --model_path ./saved_model/CIFAR10_PreActResNet18.checkpoint --sub_model preActResnet18 --sub_model_path ./saved_model/sub_preact_resnet18/model.pt;
python3 black_box_test.py --attack mi_fgsm --step_num 10 --step_size 0.01 --epsilon 0.031 --alpha 0.9 --log_file ./log/ablation_decay.log --dataset cifar10 --model preActResnet18 --model_path ./saved_model/CIFAR10_PreActResNet18.checkpoint --sub_model preActResnet18 --sub_model_path ./saved_model/sub_preact_resnet18/model.pt;
python3 black_box_test.py --attack mi_fgsm --step_num 10 --step_size 0.01 --epsilon 0.031 --alpha 1.0 --log_file ./log/ablation_decay.log --dataset cifar10 --model preActResnet18 --model_path ./saved_model/CIFAR10_PreActResNet18.checkpoint --sub_model preActResnet18 --sub_model_path ./saved_model/sub_preact_resnet18/model.pt;

python3 black_box_test.py --attack mi_fgsm --step_num 10 --step_size 0.1 --epsilon 0.3 --alpha 0.01 --log_file ./log/ablation_decay.log;
python3 black_box_test.py --attack mi_fgsm --step_num 10 --step_size 0.1 --epsilon 0.3 --alpha 0.025 --log_file ./log/ablation_decay.log;
python3 black_box_test.py --attack mi_fgsm --step_num 10 --step_size 0.1 --epsilon 0.3 --alpha 0.05 -log_file ./log/ablation_decay.log;
python3 black_box_test.py --attack mi_fgsm --step_num 10 --step_size 0.1 --epsilon 0.3 --alpha 0.075 --log_file ./log/ablation_decay.log;
python3 black_box_test.py --attack mi_fgsm --step_num 10 --step_size 0.1 --epsilon 0.3 --alpha 0.1 --log_file ./log/ablation_decay.log;
python3 black_box_test.py --attack mi_fgsm --step_num 10 --step_size 0.1 --epsilon 0.3 --alpha 0.2 --log_file ./log/ablation_decay.log;
python3 black_box_test.py --attack mi_fgsm --step_num 10 --step_size 0.1 --epsilon 0.3 --alpha 0.3 --log_file ./log/ablation_decay.log;
python3 black_box_test.py --attack mi_fgsm --step_num 10 --step_size 0.1 --epsilon 0.3 --alpha 0.4 --log_file ./log/ablation_decay.log;
python3 black_box_test.py --attack mi_fgsm --step_num 10 --step_size 0.1 --epsilon 0.3 --alpha 0.5 --log_file ./log/ablation_decay.log;
python3 black_box_test.py --attack mi_fgsm --step_num 10 --step_size 0.1 --epsilon 0.3 --alpha 0.6 --log_file ./log/ablation_decay.log;
python3 black_box_test.py --attack mi_fgsm --step_num 10 --step_size 0.1 --epsilon 0.3 --alpha 0.7 --log_file ./log/ablation_decay.log;
python3 black_box_test.py --attack mi_fgsm --step_num 10 --step_size 0.1 --epsilon 0.3 --alpha 0.8 --log_file ./log/ablation_decay.log;
python3 black_box_test.py --attack mi_fgsm --step_num 10 --step_size 0.1 --epsilon 0.3 --alpha 0.9 --log_file ./log/ablation_decay.log;
python3 black_box_test.py --attack mi_fgsm --step_num 10 --step_size 0.1 --epsilon 0.3 --alpha 1.0 --log_file ./log/ablation_decay.log;

