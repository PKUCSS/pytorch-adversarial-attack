import torch
from torch.nn import CrossEntropyLoss

def fgsm_attack(img, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_img = img + epsilon*sign_data_grad
    perturbed_img = torch.clamp(perturbed_img, 0, 1)
    return perturbed_img


def pgd_attack(img, label, device, model, step_size, step_num, epsilon):

    img, label = img.to(device), label.to(device)
    ori_img = img.data  
        
    for i in range(step_num) :    
        img.requires_grad = True
        logits = model(img)
        model.zero_grad()
        loss = CrossEntropyLoss()(logits, label)
        loss.backward()
        eta = step_size*img.grad.sign()
        adv_img = img + eta
        eta = torch.clamp(adv_img - ori_img, min=-epsilon, max=epsilon)
        img = torch.clamp(ori_img + eta, min=0, max=1).detach_()   
    return img

def mi_fgsm_attack(img, label, device, model, step_size, step_num, epsilon, alpha=1.0):

    img, label = img.to(device), label.to(device)
    ori_img = img.data  
    momentum = torch.zeros_like(img).to(device)

    for i in range(step_num) :    
        img.requires_grad = True
        logits = model(img)

        model.zero_grad()
        loss = CrossEntropyLoss()(logits, label)
        loss.backward()

        grad = img.grad
        grad_norm = torch.norm(grad.reshape(-1),1)
        grad = grad/grad_norm + alpha*momentum
        momentum = grad

        eta = step_size*grad.sign()

        adv_img = img + eta
        eta = torch.clamp(adv_img - ori_img, min=-epsilon, max=epsilon)
        img = torch.clamp(ori_img + eta, min=0, max=1).detach_()   
    return img





