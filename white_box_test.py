import os
import torch
import argparse
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from loguru import logger
from utils import get_model, load_model, get_dataloader
from attacks import fgsm_attack, pgd_attack, mi_fgsm_attack

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0',
                        type=str, required=False, help='GPU ids')
    parser.add_argument('--model', default='smallCNN',
                        type=str, required=False, help='Model Type')
    parser.add_argument('--model_path', default='./saved_model/MNIST_small_cnn.checkpoint',
                        type=str, help='model path')
    parser.add_argument('--attack', default='fgsm',type=str, choices=['fgsm','pgd','mi_fgsm'])
    parser.add_argument('--alpha', default=1.0, type=float, help='momentum decay')
    parser.add_argument('--dataset', default='mnist', type=str, choices=['mnist','cifar10'])
    parser.add_argument('--epsilon', default=0.3, type=float, help='L-infinite norm limit')
    parser.add_argument('--step_size', default=0.1, type=float, help='PGD step size')
    parser.add_argument('--step_num', default=10, type=int, help='the number of PGD steps')
    parser.add_argument('--log_file', type=str, default='./log/default.log')
    args = parser.parse_args()

    log_file_name = args.log_file
    logger.add(log_file_name)
    logger.info('args:\n' + args.__repr__())

    model = get_model(args.model)
    model = load_model(model, args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    test_loader = get_dataloader(args.dataset, batch_size=1)

    correct = 0
    adv_sample_num = 0
    #adv_examples = []

    epsilon = args.epsilon
    step_num = args.step_num
    step_size = args.step_size 

    for img, label in tqdm(test_loader):

        img, label = img.to(device), label.to(device)
        img.requires_grad = True
        logits = model(img)
        pred = logits.data.max(1)[1]
        if pred.item() != label.item():
            continue
        adv_sample_num += 1
        if args.attack == 'fgsm':
            loss = CrossEntropyLoss()(logits, label)
            model.zero_grad()
            loss.backward()
            data_grad = img.grad.data
            perturbed_img = fgsm_attack(img, epsilon, data_grad)
        elif args.attack == 'pgd':
            perturbed_img = pgd_attack(img, label, device, model, step_size=args.step_size, step_num=args.step_num, epsilon=args.epsilon)        
        elif args.attack == 'mi_fgsm':
            perturbed_img = mi_fgsm_attack(img, label, device, model, step_size=args.step_size, step_num=args.step_num, epsilon=args.epsilon, alpha=args.alpha)     
        else:
            raise NotImplementedError
        output = model(perturbed_img)
        final_pred = output.data.max(1)[1]
        if final_pred.item() == label.item():
            correct += 1

    final_acc = correct/float(len(test_loader))
    asr = (adv_sample_num-correct)/adv_sample_num

    logger.info("Epsilon: {}\tTest Accuracy = {}, ASR = {}".format(epsilon, final_acc, asr))

if __name__ == '__main__':
    main()


    





