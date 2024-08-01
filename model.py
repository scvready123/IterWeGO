import torch
import numpy as np
import torch.nn as nn
import time
import torch.nn.functional as F
from en_decoder import Encoder, EncoderLayer
from torch.autograd import Variable
from transformers import CLIPModel, BertModel

def loss1(out_logits):
    label_im = torch.zeros(out_logits.size(0), out_logits.size(0))
    label_im += torch.triu(torch.ones(out_logits.size(0), out_logits.size(0)))
    label_im = label_im.bool().cuda()
    cm_im = torch.tril(torch.ones_like(out_logits[:, :, 0]), diagonal=-1) + torch.triu(torch.ones_like(out_logits[:, :, 0]), diagonal=1)
    cm_im = (cm_im == 1).float()
    tf_im = out_logits.view(-1, out_logits.size(2))
    lf_im = label_im.view(-1)
    mf_im = cm_im.view(-1).bool().cuda()
    st_im = tf_im[mf_im]
    sl_im = lf_im[mf_im]
    loss = F.cross_entropy(st_im, sl_im.long())
    return loss


class Classifier(nn.Module):

    def __init__(self, opt):
        super(Classifier, self).__init__()
        if opt.do_test == True:
            self.dropout = nn.Dropout(0.0)
        else:
            self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(opt.d_mlp * 2, opt.d_mlp)
        self.classifier = nn.Linear(opt.d_mlp, opt.num_classes)

    def forward(self, input):
        lin_out = torch.tanh(self.dropout(self.linear(input)))
        logits = self.classifier(lin_out)
        return logits

class Text_encoder(nn.Module):
    def __init__(self, args):
        super(Text_encoder, self).__init__()

        self.emb_dp = args.input_drop_ratio
        self.model_dp = args.drop_ratio

        h_dim = args.d_rnn
        selfatt_layer = EncoderLayer(h_dim, 4, 512, args.attdp)
        self.encoder = Encoder(selfatt_layer, args.gnnl)
        self.classifier = Classifier(args)
        self.bert = BertModel.from_pretrained("bert-base-uncased")

    def forward(self, captions):
        paramemory = self.encode(captions).squeeze(0) 
        new_shape = (paramemory.size(0), paramemory.size(0), 2*paramemory.size(1))
        cp_mem = torch.zeros(new_shape).cuda()
        for i in range(paramemory.size(0)):
            for j in range(paramemory.size(0)):
                con_sub = torch.cat((paramemory[i], paramemory[j]), dim=0)
                cp_mem [i, j] = con_sub
        out = self.classifier(cp_mem) 

        return out
    
    def eval_forward(self, captions, idx):
        paramemory = self.encode(captions).squeeze(0) 
        paramemory = paramemory[idx]
        new_shape = (paramemory.size(0), paramemory.size(0), 2*paramemory.size(1))
        cp_mem = torch.zeros(new_shape).cuda()
        for i in range(paramemory.size(0)):
            for j in range(paramemory.size(0)):
                con_sub = torch.cat((paramemory[i], paramemory[j]), dim=0)
                cp_mem[i, j] = con_sub

        out = self.classifier(cp_mem) 
        return out
    
    def encode(self, captions):
        out = self.bert(input_ids = captions['input_ids'],
        attention_mask = captions['attention_mask'], token_type_ids = captions['token_type_ids']) 
        out = out[0].mean(1)
        sentences = out.unsqueeze(0)
        sen_mask = sentences.new_zeros(sentences.size(0), sentences.size(1)).bool()
        for i in range(sentences.size(0)):
            sen_mask[i, :sentences.size(1)] = 1
        sen_mask = sen_mask.unsqueeze(1)
        parasen = self.encoder(sentences, sen_mask)
        return parasen


class Image_encoder(nn.Module): 
    def __init__(self, args):
        super(Image_encoder, self).__init__()

        self.emb_dp = args.input_drop_ratio
        self.model_dp = args.drop_ratio

        h_dim = args.d_rnn

        selfatt_layer = EncoderLayer(h_dim, 4, 512, args.attdp)
        self.encoder = Encoder(selfatt_layer, args.gnnl)

        self.classifier = Classifier(args)

        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

    def forward(self, img):
        paramemory = self.encode(img).squeeze(0) 
        new_shape = (paramemory.size(0), paramemory.size(0), 2*paramemory.size(1))
        cp_mem = torch.zeros(new_shape).cuda()
        for i in range(paramemory.size(0)):
            for j in range(paramemory.size(0)):
                con_sub = torch.cat((paramemory[i], paramemory[j]), dim=0)
                cp_mem[i, j] = con_sub
        out = self.classifier(cp_mem) 
    
        return out

    def eval_forward(self, img, rand_idx):
        paramemory = self.encode(img).squeeze(0) 
        paramemory = paramemory[rand_idx]

        new_shape = (paramemory.size(0), paramemory.size(0), 2*paramemory.size(1))
        cp_mem = torch.zeros(new_shape).cuda()
        for i in range(paramemory.size(0)):
            for j in range(paramemory.size(0)):
                con_sub = torch.cat((paramemory[i], paramemory[j]), dim=0)
                cp_mem[i, j] = con_sub

        out = self.classifier(cp_mem) 
    
        return out

    
    def encode(self, imgfeature):
        out = self.clip.get_image_features(pixel_values=imgfeature) 
        imgfeatures = out / out.norm(p=2, dim=-1, keepdim=True)
        imags = imgfeatures.unsqueeze(0)
        ima_mask = imags.new_zeros(imags.size(0), imags.size(1)).bool()
        for i in range(imags.size(0)):
            ima_mask[i, :imags.size(1)] = 1
        ima_mask = ima_mask.unsqueeze(1)
        paraim = self.encoder(imags, ima_mask)  

        return paraim


class natorderNet(nn.Module):
    def __init__(self, args):
        super(natorderNet, self).__init__()
        self.args = args

        self.emb_dp = args.input_drop_ratio
        self.model_dp = args.drop_ratio

        h_dim = args.d_rnn
        self.thr_thi = args.thr_thi
        self.thr_iht = args.thr_iht
        self.img_flow = Image_encoder(args)
        self.txt_flow = Text_encoder(args)
        self.clip_align = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        if args.cp:
            self.checkpoint_clip = torch.load(args.cp, map_location='cpu') 
            self.clip_align.load_state_dict(self.checkpoint_clip['model'])  

    def model_img_train(self, images, captions, cap_clip):
        out_im = self.img_flow(images) 
        out_t = self.txt_flow(captions) 

        clip_ali_out = self.clip_align(pixel_values = images, input_ids = cap_clip['input_ids'],
        attention_mask = cap_clip['attention_mask'])
        biatten_t2i = clip_ali_out[1]
        thr = self.thr_thi
        upper = []
        lower = []
        soft_t = F.softmax(out_t, dim=-1)
        txt_result = soft_t[..., 0]
        for i in range(txt_result.size(0)):
            for j in range(i+1, txt_result.size(0)):
                if torch.any(txt_result[i][j] > thr):
                    upper.append([i,j])

        for i in range(txt_result.size(0)):
            for j in range(0, i):
                if torch.any(txt_result[i][j] > thr):
                    lower.append([i,j])

        all = upper + lower 
        result_con_list = []
        for index_pair in all:
            row1, row2 = index_pair
            max_indices = [int(torch.argmax(biatten_t2i[row1])), int(torch.argmax(biatten_t2i[row2]))]
            result_con_list.append(max_indices)
        
        for pair_id in range(len(all)):
            sub1 = all[pair_id]  
            sub2 = result_con_list[pair_id]   
            out_im[sub2[0]][sub2[1]] = out_im[sub2[0]][sub2[1]] + out_t[sub1[0]][sub1[1]]
        loss = loss1(out_im)
        return loss

    def model_txt_train(self, images, captions, cap_clip):
        out_im = self.img_flow(images) 
        out_t = self.txt_flow(captions)
        clip_ali_out = self.clip_align(pixel_values = images, input_ids = cap_clip['input_ids'],
        attention_mask = cap_clip['attention_mask'])
        biatten_i2t = clip_ali_out[0]
        thr = self.thr_iht
        upper = []
        lower = []
        soft_t = F.softmax(out_im, dim=-1)
        img_result = soft_t[..., 0]
        for i in range(img_result.size(0)):
            for j in range(i+1, img_result.size(0)):
                if torch.any(img_result[i][j] > thr):
                    upper.append([i,j])
        for i in range(img_result.size(0)):
            for j in range(0, i):
                if torch.any(img_result[i][j] > thr):
                    lower.append([i,j])

        all = upper + lower 
        result_con_list = []
        for index_pair in all:
            row1, row2 = index_pair
            max_indices = [int(torch.argmax(biatten_i2t[row1])), int(torch.argmax(biatten_i2t[row2]))]
            result_con_list.append(max_indices)
        
        for pair_id in range(len(all)):
            sub1 = all[pair_id]  
            sub2 = result_con_list[pair_id]  
            out_t[sub2[0]][sub2[1]] = out_t[sub2[0]][sub2[1]] + self.args.lamda_i * out_im[sub1[0]][sub1[1]]

        loss = loss1(out_t)

        return loss
    

def print_params(model):
    print('total parameters:', sum([np.prod(list(p.size())) for p in model.parameters()]))

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """
    Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

def get_loss(imgs, caps, margin):
    sims = []
    n_image = imgs.size(0)
    for i in range(n_image):
        img_i = imgs[i]  
        img_rep = img_i.repeat(n_image, 1)
        sim_i = cosine_similarity(img_rep, caps)
        sims.append(sim_i.unsqueeze(0))

    sims = torch.cat(sims, 0)  

    diagonal = sims.diag().view(imgs.size(0), 1)
    d1 = diagonal.expand_as(sims)
    d2 = diagonal.t().expand_as(sims)
    loss_t2i = (margin + sims - d1).clamp(min=0.).sum()
    losses = loss_t2i
    return losses

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def train(args, train_iter, dev, test_real, checkpoint_txt, checkpoint_img):
    model = natorderNet(args)
    model.cuda()

    model.txt_flow.load_state_dict(checkpoint_txt['model'])
    model.img_flow.load_state_dict(checkpoint_img['model'])

    for pt in model.txt_flow.bert.parameters():
        pt.requires_grad = False
        
    for pt1 in model.txt_flow.bert.encoder.layer[10].parameters():
        pt1.requires_grad = True
        
    for pt2 in model.txt_flow.bert.encoder.layer[11].parameters():
        pt2.requires_grad = True
    
    for pi in model.img_flow.clip.parameters():
        pi.requires_grad = False
        
    for pi1 in model.img_flow.clip.visual_projection.parameters():
        pi1.requires_grad = True
    
    for pca in model.clip_align.parameters():
        pca.requires_grad = False
        

    lr = args.lr
    wd = 1e-5
    params_im_bi = [{'params': model.img_flow.clip.visual_projection.parameters(), 'lr': 0.1*lr},
              {'params': model.img_flow.encoder.parameters(), 'lr': 0.5*lr},
              {'params': model.img_flow.classifier.parameters(), 'lr': 0.5*lr}]
    opt_im_bi = torch.optim.Adam(params_im_bi, lr=args.lr, weight_decay=wd)

    params_tx_bi = [{'params': model.txt_flow.bert.encoder.layer[10].parameters(), 'lr': 0.1*lr},
              {'params': model.txt_flow.bert.encoder.layer[11].parameters(), 'lr': 0.1*lr},
              {'params': model.txt_flow.encoder.parameters(), 'lr': 0.5*lr},
              {'params': model.txt_flow.classifier.parameters(), 'lr': 0.5*lr}]
    opt_tx_bi = torch.optim.Adam(params_tx_bi, lr=args.lr, weight_decay=wd)

    best_score = -np.inf
    best_iter = 0
    start = time.time()

    early_stop = args.early_stop

    for epc in range(0, args.maximum_steps):
        for iters, (image_stories, captions_set, captions_set_for_clip) in enumerate(train_iter):
            model.train()
            opt_im_bi.zero_grad()
            loss_im = 0

            t1 = time.time()
            for si, data in enumerate(zip(image_stories, captions_set, captions_set_for_clip)):
                imgfeature = data[0]
                captions = data[1]
                cap_clip = data[2]
                imgfeature = imgfeature.cuda()
                captions['input_ids'] = captions['input_ids'].cuda()
                captions['attention_mask'] = captions['attention_mask'].cuda()
                captions['token_type_ids'] = captions['token_type_ids'].cuda()
                cap_clip['input_ids'] = cap_clip['input_ids'].cuda()
                cap_clip['attention_mask'] = cap_clip['attention_mask'].cuda()

                loss1 = model.model_img_train(imgfeature, captions, cap_clip)  
                loss_im = loss1 + loss_im

            loss_im /= (args.batch_size)

            totalloss = loss_im
            totalloss.backward()

            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt_im_bi.step()

            t2 = time.time()
            if iters % 20 == 0:
                print('epc:{} iter:{} point:{:.2f} t:{:.2f}'.format(epc, iters + 1, loss_im,
            t2 - t1))
        
        for iters, (image_stories, captions_set, captions_set_for_clip) in enumerate(train_iter):
            model.train()
            opt_tx_bi.zero_grad()
            loss_tx = 0

            t1 = time.time()
            for si, data in enumerate(zip(image_stories, captions_set, captions_set_for_clip)):
                imgfeature = data[0]
                captions = data[1]
                cap_clip = data[2]
                imgfeature = imgfeature.cuda()
                captions['input_ids'] = captions['input_ids'].cuda()
                captions['attention_mask'] = captions['attention_mask'].cuda()
                captions['token_type_ids'] = captions['token_type_ids'].cuda()
                cap_clip['input_ids'] = cap_clip['input_ids'].cuda()
                cap_clip['attention_mask'] = cap_clip['attention_mask'].cuda()

                loss1 = model.model_txt_train(imgfeature, captions, cap_clip)
                loss_tx= loss1 + loss_tx

            loss_tx /= (args.batch_size)

            totalloss = loss_tx
            totalloss.backward()

            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt_tx_bi.step()

            t2 = time.time()
            if iters % 20 == 0:
                print('epc:{} iter:{} point:{:.2f} t:{:.2f}'.format(epc, iters + 1, loss_tx,
            t2 - t1))

        with torch.no_grad():
            print('valid..............')
            score = valid_model(args, model, dev)
            all_score = score
            print('epc:{}, acc:{:.2%}, best:{:.2%}'.format(epc, all_score, best_score))

            if all_score > best_score:
                best_score = all_score
                best_iter = epc

                print('save best model at epc={}'.format(epc))
                checkpoint = {'model': model.state_dict(),
                              'args': args,
                              'best_score': best_score}
                torch.save(checkpoint, '{}/{}.best.pt'.format(args.model_path, args.model))

            if early_stop and (epc - best_iter) >= early_stop:
                print('early stop at epc {}'.format(epc))
                break

    print('\n*******Train Done********{}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    minutes = (time.time() - start) // 60
    print("train_minutes", minutes)


def valid(model, images, captions, cap_clip):

    out_im = model.img_flow(images) 
    out_t = model.txt_flow(captions) 

    clip_ali_out = model.clip_align(pixel_values = images, input_ids = cap_clip['input_ids'], attention_mask = cap_clip['attention_mask'])
    
    biatten_i2t = clip_ali_out[0]
    biatten_t2i = clip_ali_out[1]

    thr = model.thr_thi
    upper = []
    lower = []
    soft_t = F.softmax(out_t, dim=-1)
    txt_result = soft_t[..., 0]
    for i in range(txt_result.size(0)):
        for j in range(i+1, txt_result.size(0)):
            if torch.any(txt_result[i][j] > thr):
                upper.append([i,j])

    for i in range(txt_result.size(0)):
        for j in range(0, i):
            if torch.any(txt_result[i][j] > thr):
                lower.append([i,j])


    all = upper + lower 
    result_con_list = []
    for index_pair in all:
        row1, row2 = index_pair
        max_indices = [int(torch.argmax(biatten_t2i[row1])), int(torch.argmax(biatten_t2i[row2]))]
        result_con_list.append(max_indices)
        
    for pair_id in range(len(all)):
        sub1 = all[pair_id]  
        sub2 = result_con_list[pair_id]   

        out_im[sub2[0]][sub2[1]] = out_im[sub2[0]][sub2[1]] + model.args.lamda_t * out_t[sub1[0]][sub1[1]]


    max_indices = torch.argmax(out_im, dim=2)

    count_upper_ones = torch.sum(torch.triu(max_indices, diagonal=1) == 1)

    lower = 0
    for i in range(1, max_indices.size(0)):
        count_zeros = torch.sum(max_indices[i, :i] == 0)
        lower = lower + count_zeros

    total_im = (count_upper_ones.item()+lower.item())/(max_indices.size(0)*(max_indices.size(0)-1))
    
    thr = model.thr_iht
    upper = []
    lower = []
    soft_tt = F.softmax(out_im, dim=-1)
    img_result = soft_tt[..., 0]
    for i in range(img_result.size(0)):
        for j in range(i+1, img_result.size(0)):
            if torch.any(img_result[i][j] > thr):
                upper.append([i,j])

    for i in range(img_result.size(0)):
        for j in range(0, i):
            if torch.any(img_result[i][j] > thr):
                lower.append([i,j])

    all = upper + lower 
    result_con_list = []

    for index_pair in all:
        row1, row2 = index_pair
        max_indices = [int(torch.argmax(biatten_i2t[row1])), int(torch.argmax(biatten_i2t[row2]))]
        result_con_list.append(max_indices)

    for pair_id in range(len(all)):
        sub1 = all[pair_id]  
        sub2 = result_con_list[pair_id]   
        out_t[sub2[0]][sub2[1]] = out_t[sub2[0]][sub2[1]] + model.args.lamda_i * out_im[sub1[0]][sub1[1]]

    max_indices_t = torch.argmax(out_t, dim=2)
    count_upper_ones_t = torch.sum(torch.triu(max_indices_t, diagonal=1) == 1)
    lower_t = 0
    for i in range(1, max_indices_t.size(0)):
        count_zeros_t = torch.sum(max_indices_t[i, :i] == 0)
        lower_t = lower_t + count_zeros_t

    total_t = (count_upper_ones_t.item()+lower_t.item())/(max_indices_t.size(0)*(max_indices_t.size(0)-1))

    return total_im, total_t


def valid_model(args, model, dev, dev_metrics=None, shuflle_times=1):
    model.eval()
    predicted = []

    for j, (image_stories, captions_set, captions_set_for_clip) in enumerate(dev):

        for si, data in enumerate(zip(image_stories, captions_set, captions_set_for_clip)):
            imgfeature = data[0]
            captions = data[1]
            cap_clip = data[2]

            imgfeature = imgfeature.cuda()
            captions['input_ids'] = captions['input_ids'].cuda()
            captions['attention_mask'] = captions['attention_mask'].cuda()
            captions['token_type_ids'] = captions['token_type_ids'].cuda()
            cap_clip['input_ids'] = cap_clip['input_ids'].cuda()
            cap_clip['attention_mask'] = cap_clip['attention_mask'].cuda()
            pred_im, pred_txt = valid(model, imgfeature, captions, cap_clip) 

            predicted.append((pred_im+pred_txt)/2)
    
    print("valid_score:", float(np.mean(predicted)))
    return float(np.mean(predicted))
