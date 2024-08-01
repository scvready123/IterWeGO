import os 
import argparse
import json
import torch
import codecs
import csv
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from PIL import Image
from model import natorderNet
from util import it_guide_iht, it_guide_thi
from tp_sort import readf, convert_to_graph
from transformers import AutoTokenizer, AutoProcessor

random.seed(1234)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  
processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")  
tokenizer_for_clip = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")  

def map_from_indices(ori, index_list, indices1):
    t = []
    for index in index_list:
        t.append(ori[indices1[index[0]], indices1[index[1]]])
    return torch.cat(t, dim=0).reshape(len(index_list), -1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_file_path", default='', type=str, help="The image file path")
    parser.add_argument("--img_test_data_path", default='', type=str, help="txt csv data path")
    parser.add_argument("--txt_test_data_path", default='', type=str, help="img csv data path")
    parser.add_argument('--n_heads', type=int, default=4, help='number of heads')
    parser.add_argument('--d_rnn', type=int, default=768, help='hidden dimention size')
    parser.add_argument('--d_mlp', type=int, default=768, help='dimention size for FFN')
    parser.add_argument('--gnnl', default=2, type=int, help='stacked layer number')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--thr_thi', type=float, default=0.8)  
    parser.add_argument('--thr_iht', type=float, default=0.8)  
    parser.add_argument('--drop_ratio', type=float, default=0.0, help='dropout ratio')    
    parser.add_argument('--input_drop_ratio', type=float, default=0.0, help='dropout ratio only for inputs')   
    parser.add_argument('--lamda_t', type=float, default=1)
    parser.add_argument('--lamda_i', type=float, default=0.25)
    parser.add_argument('--attdp', default=0.0, type=float, help='self-att dropout')  
    parser.add_argument('--load_model',default='', help='trained model path')
    parser.add_argument('--do_test', default= True, help='')  
    parser.add_argument('--cp', default = None, help='') 
    parser.add_argument('--output_img', default='./img_out.txt', action='store_true')
    parser.add_argument('--output_txt', default='./txt_out.txt', action='store_true')

    args = parser.parse_args()

    load_from = '{}'.format(args.load_model)
    print("loading")
    checkpoint = torch.load(load_from, map_location='cpu')
    model = natorderNet(args)
    model.load_state_dict(checkpoint['model'])
    model.cuda()
    model.eval()

    image_dir = args.img_file_path

    best_acc_t = -100
    best_acc_im = -100

    t_eval_out = None
    im_eval_out = None

    for round in range(3):

        fhandle_im = codecs.open(args.img_test_data_path, "r", "utf-8")  
        tsv_reader_im = csv.reader(fhandle_im, delimiter='\t')
        fhandle_t = codecs.open(args.txt_test_data_path, "r", "utf-8") 
        tsv_reader_t = csv.reader(fhandle_t, delimiter='\t')

        count = 0
        images, texts = [], []
        sub_dict_im, sub_dict_t = {}, {}
        current_order_im, current_order_t = [], []
        outputs_im, outputs_t, predictions_im, predictions_t = [], [], [], []

        for line_im, line_t in zip(tsv_reader_im, tsv_reader_t):
            count += 1
            
            images.append(line_im)
            imgid1 = line_im[1]
            imgid2 = line_im[2]
            imgid1_ord = line_im[4]
            imgid2_ord = line_im[5]

            texts.append(line_t)
            text1 = line_t[1]
            text2 = line_t[2]
            textid1_ord = line_t[4]
            textid2_ord = line_t[5]
    
            current_order_im.append([int(imgid1_ord), int(imgid2_ord)])
            sub_dict_im.update({imgid1: int(imgid1_ord)})
            sub_dict_im.update({imgid2: int(imgid2_ord)})

            current_order_t.append([int(textid1_ord), int(textid2_ord)])
            sub_dict_t.update({text1: int(textid1_ord)})
            sub_dict_t.update({text2: int(textid2_ord)})

            if count % 10 == 0:
                img_ord = [0, 1, 2, 3, 4]
                txt_ord = [0, 1, 2, 3, 4]
                sub_im_list = [] 
                sorted_dict_im = dict(sorted(sub_dict_im.items(), key=lambda x: x[1])) 
                lowercase_keys_list_im = [key.lower() for key in sorted_dict_im.keys()]

                image_formats = ['.jpg', '.gif', '.png', '.bmp']
                for im_name in lowercase_keys_list_im:
                    for image_format in image_formats:
                            try:
                                image = Image.open(os.path.join(image_dir, str(im_name) + image_format)).convert('RGB')
                            except Exception:
                                continue
                
                    image_pro = processor(images=image, return_tensors='pt')
                    image_p = image_pro.pixel_values
                    sub_im_list.append(image_p.squeeze(0))
                img_for_clip = torch.stack(sub_im_list).cuda()
                random.shuffle(img_ord)
                shuff_img_for_clip = img_for_clip[img_ord]
                indices_im = np.argsort(img_ord)
 
                sorted_dict_t = dict(sorted(sub_dict_t.items(), key=lambda x: x[1])) 
                lowercase_keys_list_t = [key.lower() for key in sorted_dict_t.keys()]
                random.shuffle(txt_ord)
                shuff_lowercase_keys_list_t = [lowercase_keys_list_t[i] for i in txt_ord]
                indices_tx = np.argsort(txt_ord)
                captions = tokenizer(shuff_lowercase_keys_list_t, padding='longest', max_length = 40, truncation=True, return_tensors="pt")
                captions['input_ids'] = captions['input_ids'].cuda()
                captions['attention_mask'] = captions['attention_mask'].cuda()
                captions['token_type_ids'] = captions['token_type_ids'].cuda()
                captions_for_clip = tokenizer_for_clip(shuff_lowercase_keys_list_t, padding='longest', max_length = 40, truncation=True, return_tensors="pt")
                captions_for_clip['input_ids'] = captions_for_clip['input_ids'].cuda()
                captions_for_clip['attention_mask'] = captions_for_clip['attention_mask'].cuda()

                out_im = model.img_flow(shuff_img_for_clip) 
                out_t = model.txt_flow(captions) 

                out_im = out_im.detach().cpu()
                out_t = out_t.detach().cpu()

                clip_ali_out = model.clip_align(pixel_values = shuff_img_for_clip, input_ids = captions_for_clip['input_ids'],
                attention_mask = captions_for_clip['attention_mask'])
                biatten_t2i = clip_ali_out[1]
                biatten_i2t = clip_ali_out[0]

                for it_num in range(round+1):
                    out_im = it_guide_thi(out_im, out_t, args.thr_thi, biatten_t2i, args.lamda_t)
                    out_t = it_guide_iht(out_im, out_t, args.thr_iht, biatten_i2t, args.lamda_i)
                
                scores_img = nn.functional.softmax(out_im, dim=-1)
                seleceted_img = map_from_indices(scores_img, current_order_im, indices_im)
                pred_img = seleceted_img.max(1)[1]
                outputs_im += seleceted_img.data.tolist()
                predictions_im += pred_img.data.tolist()

                scores_t = nn.functional.softmax(out_t, dim=-1)
                seleceted_t = map_from_indices(scores_t, current_order_t, indices_tx)
                pred_t = seleceted_t.max(1)[1]
                outputs_t += seleceted_t.data.tolist()
                predictions_t += pred_t.data.tolist()

                sub_dict_im, sub_dict_t = {}, {}
                current_order_im, current_order_t = [], []
                img_ord, txt_ord = [], []
                shuff_img_for_clip, shuff_lowercase_keys_list_t, indices_im, indices_tx = None, None, None, None

        data_img = []
        data_txt = []

        for ni in range(len(images)):
            data_img.append([images[ni][0], images[ni][1], images[ni][2], \
                     images[ni][3], images[ni][4], images[ni][5], \
                     outputs_im[ni][0], outputs_im[ni][1], predictions_im[ni]])
        
        for nt in range(len(texts)):
            data_txt.append([texts[nt][0], texts[nt][1], texts[nt][2], \
                     texts[nt][3], texts[nt][4], texts[nt][5], \
                     outputs_t[nt][0], outputs_t[nt][1], predictions_t[nt]])

        stats_img = convert_to_graph(data_img)
        stats_txt = convert_to_graph(data_txt)

        img_acc, img_pmr, img_tao = stats_img.get_eval_results()
        txt_acc, txt_pmr, txt_tao = stats_txt.get_eval_results()

        if best_acc_t < txt_acc:
            best_acc_t = txt_acc
            t_eval_out = [txt_acc, txt_pmr, txt_tao]
            if args.output_txt:
                with codecs.open(args.output_txt, "w", "utf-8") as outF:
                    tsv_writer = csv.writer(outF, delimiter='\t')
                    for i in range(len(texts)):
                        tsv_writer.writerow(
                            [texts[i][0], texts[i][1], texts[i][2], \
                                texts[i][3], texts[i][4], texts[i][5], \
                                outputs_t[i][0], outputs_t[i][1], predictions_t[i]])
        
        if best_acc_im < img_acc:
            best_acc_im = img_acc
            im_eval_out = [img_acc, img_pmr, img_tao]
            if args.output_img:
                with codecs.open(args.output_img, "w", "utf-8") as outF:
                    tsv_writer = csv.writer(outF, delimiter='\t')
                    for i in range(len(images)):
                        tsv_writer.writerow(
                            [images[i][0], images[i][1], images[i][2], \
                                images[i][3], images[i][4], images[i][5], \
                                outputs_im[i][0], outputs_im[i][1], predictions_im[i]])

        print("best_acc_t, best_acc_im", best_acc_t, best_acc_im)

    print("img_eval_result:")
    print("acc:", im_eval_out[0])
    print("pmr:", im_eval_out[1])
    print("tao:", im_eval_out[2])
    print("txt_eval_result:")
    print("acc:", t_eval_out[0])
    print("pmr:", t_eval_out[1])
    print("tao:", t_eval_out[2])


    
if __name__ == "__main__":
    main()