def get_ratio(teacher_logits, logits, mu=0.5):
    # [B, L, V]
    teacher_logits = torch.masked_fill(teacher_logits, torch.isinf(teacher_logits), 0).to(torch.float32)
    logits = torch.masked_fill(logits, torch.isinf(logits), 0).to(torch.float32)
    
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32).detach()

    re_teacher_probs, idx = teacher_probs.sort(dim=-1, descending=True)
    re_student_probs = student_probs.gather(dim=-1, index=idx)

    errors = torch.abs(re_teacher_probs - re_student_probs)

    cum_sum = torch.cumsum(re_teacher_probs, dim=-1) # B,L,V
    mask = cum_sum > mu
    mask[:,:,0]=False #第一个概率一定要置False，对应第一个概率>0.5时mask全True

    s1 = torch.masked_fill(errors, mask, 0.0).sum(dim=-1)
    s2 = torch.masked_fill(errors, ~mask, 0.0).sum(dim=-1)


    return s1/(s1+s2), s2/(s1+s2)

def get_kl(teacher_logits, logits, inf_mask, mask, ratio=None):
    #ratio: [B,L]
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    teacher_prod_probs =  torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
    teacher_x =  torch.sum(teacher_prod_probs, dim=-1).view(-1)

    logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1) # [B,L]->[BL]

    if ratio == None:
        distil_loss = torch.sum((teacher_x-x) * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    else:
        distil_loss = torch.sum((teacher_x-x) * ratio.view(-1) * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def AKL(teacher_logits, logits):
    inf_mask = torch.isinf(logits) # [batch, seq, vocab]
    mask = (no_model_batch["label"] != -100).int() # [batch, seq]
    
    h_ratio, l_ratio = get_ratio(teacher_logits, logits)
    distil_loss = get_kl(teacher_logits, logits, inf_mask, mask, h_ratio) + get_kl(logits,teacher_logits, inf_mask, mask, l_ratio)
