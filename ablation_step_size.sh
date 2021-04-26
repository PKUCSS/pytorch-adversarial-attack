python3 white_box_test.py --epsilon 0.031 --attack pgd --step_size 0.00125 --step_num 10 --dataset cifar10 --model preActResnet18 --model_path ./saved_model/CIFAR10_PreActResNet18.checkpoint --log_file ./log/ablation_step_size.log;
python3 white_box_test.py --epsilon 0.031 --attack pgd --step_size 0.0025 --step_num 10 --dataset cifar10 --model preActResnet18 --model_path ./saved_model/CIFAR10_PreActResNet18.checkpoint --log_file ./log/ablation_step_size.log;
python3 white_box_test.py --epsilon 0.031 --attack pgd --step_size 0.005 --step_num 10 --dataset cifar10 --model preActResnet18 --model_path ./saved_model/CIFAR10_PreActResNet18.checkpoint --log_file ./log/ablation_step_size.log;
python3 white_box_test.py --epsilon 0.031 --attack pgd --step_size 0.01 --step_num 10 --dataset cifar10 --model preActResnet18 --model_path ./saved_model/CIFAR10_PreActResNet18.checkpoint --log_file ./log/ablation_step_size.log;
python3 white_box_test.py --epsilon 0.031 --attack pgd --step_size 0.02 --step_num 10 --dataset cifar10 --model preActResnet18 --model_path ./saved_model/CIFAR10_PreActResNet18.checkpoint --log_file ./log/ablation_step_size.log;
python3 white_box_test.py --epsilon 0.031 --attack pgd --step_size 0.04 --step_num 10 --dataset cifar10 --model preActResnet18 --model_path ./saved_model/CIFAR10_PreActResNet18.checkpoint --log_file ./log/ablation_step_size.log;
python3 white_box_test.py --epsilon 0.031 --attack pgd --step_size 0.08 --step_num 10 --dataset cifar10 --model preActResnet18 --model_path ./saved_model/CIFAR10_PreActResNet18.checkpoint --log_file ./log/ablation_step_size.log;



python3 white_box_test.py --epsilon 0.3 --attack pgd --step_size 0.0125 --step_num 10  --log_file ./log/ablation_step_size.log;
python3 white_box_test.py --epsilon 0.3 --attack pgd --step_size 0.025 --step_num 10 --log_file ./log/ablation_step_size.log ;
python3 white_box_test.py --epsilon 0.3 --attack pgd --step_size 0.05 --step_num 10 --log_file ./log/ablation_step_size.log;
python3 white_box_test.py --epsilon 0.3 --attack pgd --step_size 0.1 --step_num 10 --log_file ./log/ablation_step_size.log;
python3 white_box_test.py --epsilon 0.3 --attack pgd --step_size 0.2 --step_num 10 --log_file ./log/ablation_step_size.log;
python3 white_box_test.py --epsilon 0.3 --attack pgd --step_size 0.4 --step_num 10 --log_file ./log/ablation_step_size.log;
python3 white_box_test.py --epsilon 0.3 --attack pgd --step_size 0.8 --step_num 10 --log_file ./log/ablation_step_size.log;


