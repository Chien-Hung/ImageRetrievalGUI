# __author__ = 'ChienHung Chen in Academia Sinica IIS'
# 移除 tooltiops，直接使用 label 顯示類別 (完成)
# 開發 ranking map + tooltips (完成)
# 移除 tooltips ，直接使用 label 顯示類別 (完成)
# 多執行序開圖 (完成)
# 計算所有ap值 + 上色 (完成，計算很慢，不推薦使用)
# feature vectors + aspect ratio change + open_image (完成)
# recall computation (完成) + recall highlight (計算出的數值偏低，需要再驗證)

import os, sys
import os.path as osp
import json
import pickle
import numpy as np
import matplotlib
import xml.etree.ElementTree as ET
import argparse
import platform
import itertools
import time
import threading
import matplotlib.pyplot as plt
import torch

from torchvision import transforms
from pathlib import Path
from tkinter import *
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from libs.utils import apk, fig2img
from libs.models import ResNet50 as model

means = [0.485, 0.456, 0.406]    # imagenet
stds = [0.229, 0.224, 0.225]     # imagenet

# ==========================================================

parser = argparse.ArgumentParser(description="DetVisGUI")

# dataset information
parser.add_argument('--dataset', default='cub200', help='cars196 / cub200')
parser.add_argument('--ckpt', default='cub200_checkpoint.pth.tar', help='model checkpoint path')
parser.add_argument('--map', default='no_use', help='compute map and highlight list, no_use / compute / map path')
parser.add_argument('--k_vals',       nargs='+', default=[1,2,4,8], type=int, help='Recall @ Values.')
parser.add_argument('--device', default='cuda', help='cpu / cuda')

args = parser.parse_args()

# ==========================================================


class dataset:
    def __init__(self):
        self.dataset = args.dataset
        self.img_root = Path('Datasets/{}/images'.format(args.dataset))

        self.features = np.load('features/feats_{}.npy'.format(args.dataset))
        self.features = torch.Tensor(self.features).to(args.device)

        self.val_img_list = []
        val_img_list = np.load('features/names_{}.npy'.format(args.dataset))
        
        for x in val_img_list:
            splits = x.split('/')
            self.val_img_list.append(splits[3] + '/' + splits[4])
        
        # get query image
        self.query_img_list = []
        for x in sorted(os.listdir('query_images')):
            self.query_img_list.append(x)
        
        self.img_list = self.val_img_list
        self.query_root = 'query_images'


    def get_img_by_name(self, name):    
        if self.is_query(name):
            img = Image.open(osp.join(self.query_root, Path(name))).convert('RGB')
        else:
            img = Image.open(osp.join(self.img_root, Path(name))).convert('RGB')
        return img
     

    def get_img_by_index(self, idx):
        return self.get_img_by_name(self.img_list[idx])


    def get_feat_by_name(self, name):
        idx = np.where(np.asarray(self.img_list) == name)[0]
        return self.features[idx]


    def get_feat_by_idx(self, idx):
        return self.features[idx]


    def is_query(self, name):
        return False if name in self.val_img_list else True


    def switch_img_list(self):
        if self.img_list == self.val_img_list:
            self.img_list = self.query_img_list
        elif self.img_list == self.query_img_list:
            self.img_list = self.val_img_list
        return self.img_list


    def get_img_by_names_mt(self, top_idx, img_dict):
        t_list = []
        self.img_dict = img_dict
        names = np.asarray(self.val_img_list)[top_idx]

        def get_img(idx, name, img_dict):
            img = Image.open(osp.join(self.img_root, name)).convert('RGB')
            img_dict[idx] = img

        for i in range(len(names)):
            t_list.append(threading.Thread(target=get_img, args=(top_idx[i], names[i], img_dict)))
            t_list[i].start()

        for i in t_list:
            i.join()


    def is_CARS196(self):
        return True if self.dataset == 'cars196' else False


# main GUI
class vis_tool:
    def __init__(self):
        self.data_info = dataset()

        self.window = Tk()
        self.menubar = Menu(self.window)

        self.listBox1 = Listbox(self.window, width=50, height=37 if self.data_info.is_CARS196() else 28, font=('Times New Roman', 10))
        self.scrollbar1 = Scrollbar(self.window, width=15, orient="vertical")

        self.listBox1_info = StringVar()
        self.listBox1_label = Label(self.window, font=('Arial', 11), bg='yellow', width=4, height=1, textvariable=self.listBox1_info)

        self.frame1 = ttk.Notebook(self.window)
        self.tab_pred = Frame(self.frame1)
        self.tab_ans = Frame(self.frame1)
        self.tab_rank = Frame(self.frame1)

        self.frame1.add(self.tab_pred, text="Top 20", compound=TOP)
        self.frame1.add(self.tab_ans, text="Answer", compound=TOP)
        self.frame1.add(self.tab_rank, text="RankingMap", compound=TOP)

        self.eval_label = Label(self.tab_pred, font=('Arial', 11), bg='yellow', width=120)
        self.title_label_ans = Button(self.tab_ans, cursor="hand1", font=('Arial', 11), bg='yellow', width=120)

        # load image and show it on the window
        self.img = self.data_info.get_img_by_index(0)
        self.label_img1 = Label(self.window, height=300, width=300, highlightthickness=4, highlightbackground='#1f77b4')     # query image
        self.label_img2 = Label(self.window)                           # ranking bar
        self.label_img3 = Label(self.tab_rank)                         # rank image

        self.title_label_ranking1 = Label(self.tab_rank, font=('Arial', 11), bg='yellow', width=120)
        self.title_label_ranking2 = Label(self.tab_rank, font=('Arial', 11), bg='#D9D9D9', width=120)

        self.panel = Frame(self.tab_rank)
        self.label_img4 = Label(self.panel, height=300, width=300)                            # select image
        self.label_img5 = Label(self.panel)                            # feature vectors

        self.model = model(embed_dim=128)

        if osp.exists(args.ckpt):
            print('Loading checkpoints from {} ...'.format(args.ckpt))
            state_dict = torch.load(args.ckpt)['state_dict']
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(args.device)
            self.model.eval()
            print('Done')
        else:
            print('No checkpoint !')

        # ---------------------------------------------
        self.box_num = 20
        self.label_img_list = [Label(self.tab_pred) for _ in range(self.box_num)]
        self.photo_list = [ImageTk.PhotoImage(self.img) for _ in range(self.box_num)]
        self.label_list = [Label(self.tab_pred, font=('Arial', 11), bg='yellow', width=10, height=1) for _ in range(self.box_num)]
        self.label_cls_list = [Label(self.tab_pred, font=('Arial', 8), bg='#fdfd96', width=10, height=1) for _ in range(self.box_num)]

        self.ans_label_img_list = [Label(self.tab_ans) for _ in range(self.box_num)]
        self.ans_photo_list = [ImageTk.PhotoImage(self.img) for _ in range(self.box_num)]
        self.ans_label_list = [Label(self.tab_ans, font=('Arial', 11), bg='yellow', width=10, height=1) for _ in range(self.box_num)]

       # ---------------------------------------------
        self.find_name = ""
        self.find_label = Label(self.window, font=('Arial', 11), bg='yellow', width=10, height=1, text="find")
        self.find_entry = Entry(self.window, font=('Arial', 11), textvariable=StringVar(self.window, value=str(self.find_name)), width=10)
        self.find_button = Button(self.window, text='Enter', height=1, command=self.findname)

        self.listBox1_idx = 0  # image listBox

        # ====== ohter attribute ======
        self.img_name = ''
        self.keep_aspect_ratio = False
        self.img_list = self.data_info.img_list
        self.transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                             transforms.ToTensor(), 
                                             transforms.Normalize(means, stds)])

    def findname(self, event=None):
        self.find_name = self.find_entry.get()
        new_list = []

        if self.find_name == '':
            new_list = self.data_info.img_list
        else:
            for img_name in self.data_info.img_list:
                if self.find_name[0] == "!":
                    if self.find_name[1:] not in img_name:
                        new_list.append(img_name)
                else:
                    if self.find_name in img_name:
                        new_list.append(img_name)

        if len(new_list) != 0:
            self.img_list = new_list
            self.clear_add_listBox1()
        else:
            self.window.title("Can't find any image about '{}'".format(self.find_name))


    def clear_add_listBox1(self):
        self.listBox1.delete(0, 'end')  # delete listBox1 0 ~ end items

        # add image name to listBox1
        for item in self.img_list:
            self.listBox1.insert('end', item)

        self.listBox1.select_set(0)
        self.listBox1.focus()
        self.change_img()


    def extract_feature(self, name):
        if self.data_info.is_query(name):
            path = osp.join(self.data_info.query_root, name)
        else:
            path = osp.join(self.data_info.img_root, name)

        # open image
        img = Image.open(path).convert('RGB')
        img = self.transform(img).unsqueeze(0)

        with torch.no_grad():
            img = img.to(args.device) 
            feature = self.model(img).view(1, -1)

        return feature


    def compute_map(self):
        # compute similarity
        simmat = torch.mm(self.data_info.features, self.data_info.features.T).squeeze()
        simmat_rank = simmat.argsort(1, descending=True).cpu().numpy()

        if args.map == 'no_use':
            pass

        elif os.path.exists(args.map):
            aps = np.load(args.map)

        elif args.map == 'compute':
            print('Compute mAP ... (need a little time)')
            t1 = time.time()
            # compute map, cub200: 57s, cars196: 130s
            aps = []
            for idx, name in enumerate(self.data_info.val_img_list):
                ans = []

                # find answer names
                for img_name in self.data_info.val_img_list:
                    if name.split('/')[0] in img_name:
                        ans.append(img_name)
    
                # compute similarity
                pred = np.asarray(self.data_info.val_img_list)[simmat_rank[idx]]
                ap = apk(ans, pred[1:], len(pred))
                aps.append(ap)

            np.save('{}_aps.npy'.format(args.dataset), aps)
            print('mAP spend time: {:.2} s'.format(time.time() - t1))

        if 'aps' in locals():
            print('map : {:7.4}%'.format(np.mean(aps) * 100))
    
            for i in range(self.listBox1.size()):
                get_color = lambda r,g,b: '#%02x%02x%02x' % (r, g, b)
                if aps[i] <= 0.5:
                    color = (np.asarray([255,0,0]) * (1-2*aps[i])).astype(np.uint8)
                else:
                    color = (np.asarray([0,0,255]) * 2*(aps[i]-0.5)).astype(np.uint8)

                color = get_color(*color)
                self.listBox1.itemconfig(i, fg=color)


    def change_img(self, event=None):
        self.title_label_ranking2['bg'] = '#D9D9D9'
        self.title_label_ranking2['text'] = ''
        self.label_img4.config(image='')
        self.label_img4.config(highlightthickness=0)
        self.label_img5.config(image='')
       
        if len(self.listBox1.curselection()) != 0:
            self.listBox1_idx = self.listBox1.curselection()[0]

        self.listBox1_info.set("Image  {:6}  / {:6}".format(self.listBox1_idx + 1, self.listBox1.size()))

        self.img_name = name = self.listBox1.get(self.listBox1_idx)
        
        self.window.title("DATASET : " + self.data_info.dataset + '   ' + name)
        self.photo = ImageTk.PhotoImage(self.scale_img(self.data_info.get_img_by_name(name), keep_aspect_ratio=self.keep_aspect_ratio))
        self.label_img1.config(image=self.photo)

        # ---------------------------------------------
        self.feat = self.extract_feature(name)

        # compute similarity
        simmat = torch.mm(self.feat, self.data_info.features.T).squeeze()
        simmat_rank = torch.argsort(simmat, descending=True)            # ranking
        self.simmat_rank = simmat_rank = simmat_rank.cpu().numpy()
        self.simmat = simmat = simmat.cpu().numpy()

        offset = 0 if self.data_info.is_query(name) else 1              # not self matching
        top_rank_idx = simmat_rank[offset:self.box_num+offset]
        top_score = simmat[top_rank_idx]

        ans = []
        ans_score = []

        # find answer names
        if not self.data_info.is_query(name):
            for idx in range(len(self.data_info.features)):
                img_name = self.data_info.val_img_list[idx]
                if name.split('/')[0] in img_name:
                    ans.append(img_name)
                    ans_score.append(simmat[idx])

        # show top-20 matching images
        if self.frame1.index(self.frame1.select()) == 0:
            # open images by multithreading
            img_dict = dict()        
            self.data_info.get_img_by_names_mt(top_rank_idx, img_dict)

            for i in range(self.box_num):
                if len(ans) == 0:
                    self.label_list[i].config(bg='#fdfd96')          # Pastel yellow
                    self.frame1.tab(1, state="disabled")
                elif self.data_info.val_img_list[top_rank_idx[i]] in ans:
                    self.label_list[i].config(bg='#00ff00')                        
                    self.frame1.tab(1, state="normal")
                else:
                    self.label_list[i].config(bg='#fd9696')          # very soft red
                    self.frame1.tab(1, state="normal")

                self.label_list[i]['text'] = '{:2} ({:5.4})'.format(i+1, top_score[i])
                self.label_cls_list[i]['text'] = '{}'.format(self.data_info.val_img_list[top_rank_idx[i]].split('/')[0])

                # img = self.data_info.get_img_by_name(self.data_info.val_img_list[top_rank_idx[i]])
                img = img_dict[top_rank_idx[i]]
                img = self.scale_img(img, fix_size=160, keep_aspect_ratio=self.keep_aspect_ratio)
                self.photo_list[i] = ImageTk.PhotoImage(img)
                self.label_img_list[i].config(image=self.photo_list[i])

        # show answer images and scores 
        if self.frame1.index(self.frame1.select()) == 1:
            idx = np.argsort(ans_score)[::-1]
            ans = np.asarray(ans)[idx]
            ans_score = np.asarray(ans_score)[idx]
            self.ans, self.ans_score = ans, ans_score
            
            ans_idx = [int(np.where(np.asarray(self.data_info.val_img_list) == a)[0]) for a in ans]
            
            # open images by multithreading
            img_dict = dict()        
            self.data_info.get_img_by_names_mt(ans_idx, img_dict)

            for i in range(self.box_num):
                if i >= len(ans):
                    self.ans_label_img_list[i].config(image='')
                    self.ans_label_list[i]['bg'] = '#D9D9D9'
                    self.ans_label_list[i]['text'] = ''
                else:
                    idx = ans_idx[i]
                    # img = self.data_info.get_img_by_name(self.data_info.val_img_list[idx])
                    img = img_dict[ans_idx[i]]
                    img = self.scale_img(img, fix_size=160, keep_aspect_ratio=self.keep_aspect_ratio)
                    self.ans_photo_list[i] = ImageTk.PhotoImage(img)
                    self.ans_label_img_list[i].config(image=self.ans_photo_list[i])
                    self.ans_label_list[i]['bg'] = 'yellow'
                    self.ans_label_list[i]['text'] = str(np.round(simmat[idx], 3))

            self.title_label_ans['text'] = '{} ~ {} / {}'.format(0, min(self.box_num, len(ans)), len(ans))
            self.cur_ans_idx = 0 if self.box_num + 1 >= len(self.ans) else self.box_num

        top_pred = [x for x in np.asarray(self.data_info.val_img_list)[top_rank_idx]]

        # plot rank bar
        band = 5
        bar_len = 1250
        pred = np.asarray(self.data_info.val_img_list)[simmat_rank]
        self.bar_img = np.ones((band, bar_len, 3)) * [253, 150, 150]
        self.bar_img = self.bar_img.astype(np.uint8)
        for a in ans:
            idx = int(np.where(pred == a)[0]) - offset
            if band*(idx + 1) <= bar_len and idx >=0 :
                self.bar_img[:, band*idx:band*(idx+1)] = [0, 255, 0]

        self.photo2 = ImageTk.PhotoImage(Image.fromarray(self.bar_img))
        self.label_img2.config(image=self.photo2)

        ap = apk(ans, pred[1:], len(pred))

        recall_all_k = []        
        for k in args.k_vals:
            recall_at_k = 1 if bool(set(pred[1:k+1]) & set(ans)) else 0
            recall_all_k.append(recall_at_k)

        self.title_label_ranking1['text'] = 'AP : {:6.4} | Recall@{}: {} | Recall@{}: {} | Recall@{}: {} | Recall@{}: {}'.format(ap, args.k_vals[0], recall_all_k[0], args.k_vals[1], recall_all_k[1], args.k_vals[2], recall_all_k[2], args.k_vals[3], recall_all_k[3])
        self.eval_label['text'] = '{:^4} / {:^4} in top 20 (AP : {:6.4})'.format(len(set(top_pred) & set(ans)), len(ans), ap)

        # plot rank image
        if self.frame1.index(self.frame1.select()) == 2:
            band = 5
            bar_len = 600
            doc_len = len(self.data_info.val_img_list)
            col_num = int(bar_len / band)                  # 550 / 5 = 110
            row_num = int(np.ceil(doc_len / col_num))
            img_height = row_num * band + (row_num+1) * 2
            img_width = bar_len + (col_num+1) * 2
            
            self.rank_img = np.ones((img_height, img_width, 3)) * [253, 150, 150]
            self.rank_img = self.rank_img.astype(np.uint8)

            # tail region
            bg_color = [200, 200, 200]
            last_col_idx = (doc_len % col_num) - offset
            self.rank_img[img_height-band-2:img_height, band*last_col_idx+last_col_idx*2:] = bg_color

            # row grid
            for i in range(0, img_height, band+2):
                self.rank_img[i:i+2, :] = bg_color

            # column grid
            for i in range(0, img_width, band+2):
                self.rank_img[:, i:i+2] = bg_color

            # answer block
            for a in ans:
                idx = int(np.where(pred == a)[0]) - offset
                if idx < 0 :
                    continue

                row_idx = int(np.ceil((idx+1) / col_num)) - 1
                col_idx = idx % col_num

                self.rank_img[band*row_idx+row_idx*2+2 : band*(row_idx+1)+row_idx*2+2, 
                              band*col_idx+col_idx*2+2 : band*(col_idx+1)+col_idx*2+2] = [0, 255, 0]

            self.photo3 = ImageTk.PhotoImage(Image.fromarray(self.rank_img))
            self.label_img3.config(image=self.photo3)
            self.label_img3._ref_img = self.rank_img

            self.rank_img_event(self.label_img3, pred[1:], band+2, col_num)

        self.window.update_idletasks()


    def rank_img_event(self, widget, img_list, stride, col_num):
        def leave(event):
            self.title_label_ranking2['bg'] = '#D9D9D9'
            self.title_label_ranking2['text'] = ''
            self.label_img4.config(image='')
            self.label_img4.config(highlightthickness=0)

            widget._show_img = ImageTk.PhotoImage(Image.fromarray(widget._ref_img.copy()))
            widget.config(image=widget._show_img)


        def motion(event):
            col_idx = max(0, int((event.x-2) // stride))
            row_idx = max(0, int((event.y-2) // stride))
            widget.block_num = row_idx * col_num + col_idx

            if widget.block_num >= len(img_list) or event.x >= stride * col_num:
                return

            if not hasattr(widget, 'last_num'):
                widget.last_num = None

            if widget.last_num != widget.block_num:
                widget.last_num = widget.block_num
                self.title_label_ranking2['bg'] = '#fdfd96'
                cls, name = img_list[widget.block_num].split('/')
                self.title_label_ranking2['text'] = 'Rank: {} | Similarity Score: {:6.4} \n Class: {}  Name: {}'.format(
                    widget.block_num + 1, self.simmat[self.simmat_rank[widget.block_num+1]], cls, name)

                img = self.scale_img(self.data_info.get_img_by_name(img_list[widget.block_num]), keep_aspect_ratio=self.keep_aspect_ratio)
                self.photo4 = ImageTk.PhotoImage(img)
                self.label_img4.config(image=self.photo4, highlightthickness=4, highlightbackground='#ff7f0e') # plot Color2

                img = Image.fromarray(np.zeros((250, 256, 3), dtype=np.uint8))

                # change rank image
                tmp_img = widget._ref_img.copy()
                old_color = tmp_img[stride*row_idx + 3, stride*col_idx + 3].copy()
                tmp_img[stride*row_idx:stride*(row_idx+1)+2, stride*col_idx:stride*(col_idx+1)+2] = [0, 0, 0]
                tmp_img[stride*row_idx+2:stride*(row_idx+1), stride*col_idx+2:stride*(col_idx+1)] = old_color
                widget._show_img = ImageTk.PhotoImage(Image.fromarray(tmp_img))
                widget.config(image=widget._show_img)


        def fix_image(event):
            widget.fix_image = not widget.fix_image
            if widget.fix_image == True:
                self.plot_feats(widget.block_num+1)  # +1 -> offset
                widget.unbind('<Enter>')
                widget.unbind('<Motion>')
                widget.unbind('<Leave>')
            else:
                self.label_img5.config(image='')
                widget.bind('<Enter>', motion)
                widget.bind('<Motion>', motion)
                widget.bind('<Leave>', leave)

        widget.fix_image = False
        widget.bind('<Leave>', leave)
        widget.bind('<Enter>', motion)
        widget.bind('<Motion>', motion)
        widget.bind('<Button-1>', fix_image)  # fix / release image by mouse left button
        widget.leave = leave


    def plot_feats(self, idx):
        plt.close()
        fig = plt.figure(figsize=(5.5, 3))
        plt.plot(self.feat[0].cpu().numpy(), 'C0', linewidth=1)

        feat2 = self.data_info.get_feat_by_idx(idx).cpu().numpy()        
        plt.plot(feat2, 'C1', linewidth=1)

        plt.plot(self.feat[0].cpu().numpy() + 0.5, 'C0', linewidth=1)
        plt.plot(feat2 - 0.5, 'C1', linewidth=1)

        plt.xticks(fontsize=8)
        plt.yticks([])
        plt.tight_layout()

        img = fig2img(fig)
        self.photo5 = ImageTk.PhotoImage(img)
        self.label_img5.config(image=self.photo5)


    def scale_img(self, img, fix_size=300, keep_aspect_ratio=True):
        if keep_aspect_ratio:
            if img.width >= img.height:
                scale = fix_size / img.width
            else:
                scale = fix_size / img.height
            return img.resize((int(img.width * scale), int(img.height * scale)), Image.ANTIALIAS)
        else:
            return img.resize((fix_size, fix_size), Image.ANTIALIAS)


    def on_tab_change(self, event):
        self.change_img()
        self.listBox1.focus()


    def on_ans_change(self, event):
        ans_idx = [int(np.where(np.asarray(self.data_info.val_img_list) == a)[0]) for a in self.ans[self.cur_ans_idx:]]
            
        # open images by multithreading
        img_dict = dict()        
        self.data_info.get_img_by_names_mt(ans_idx, img_dict)

        for i in range(self.box_num):
            cur_ans_idx = self.cur_ans_idx + i
            if cur_ans_idx >= len(self.ans):
                self.ans_label_img_list[i].config(image='')
                self.ans_label_list[i]['bg'] = '#D9D9D9'
                self.ans_label_list[i]['text'] = ''
            else:
                idx = ans_idx[i]
                # img = self.data_info.get_img_by_name(self.data_info.val_img_list[idx])
                img = img_dict[ans_idx[i]]
                img = self.scale_img(img, fix_size=160, keep_aspect_ratio=self.keep_aspect_ratio)
                self.ans_photo_list[i] = ImageTk.PhotoImage(img)
                self.ans_label_img_list[i].config(image=self.ans_photo_list[i])
                self.ans_label_list[i]['bg'] = 'yellow'
                self.ans_label_list[i]['text'] = str(np.round(self.simmat[idx], 3))

        self.title_label_ans['text'] = '{} ~ {} / {}'.format(self.cur_ans_idx, min(self.cur_ans_idx + self.box_num, len(self.ans)), len(self.ans))
        self.cur_ans_idx = 0 if cur_ans_idx + 1 >= len(self.ans) else self.cur_ans_idx + self.box_num


    def eventhandler(self, event):
        if self.window.focus_get() not in [self.find_entry]:            
            if event.keysym == 'Left':
                idx = (self.frame1.index('current')-1) % len(self.frame1.tabs())
                self.frame1.select(idx)
            elif event.keysym == 'Right':
                idx = (self.frame1.index('current')+1) % len(self.frame1.tabs())
                self.frame1.select(idx)
            elif event.keysym == 'q':
                self.window.quit()


    def switch_dataset(self):
        self.img_list = self.data_info.switch_img_list()
        self.data_info.query_root = 'query_images'
        self.clear_add_listBox1()


    def open_image(self):
        filetypes = (("jpeg files","*.jpg"), ("png files","*.png"), ("all files","*.*"))
        filename = filedialog.askopenfilename(initialdir='/', title="Select file", filetypes=filetypes)
        self.img_list = [filename.split('/')[-1]]
        self.data_info.query_root = filename.replace(filename.split('/')[-1], '')
        self.clear_add_listBox1()


    def change_aspect_ratio(self):
        self.keep_aspect_ratio = not self.keep_aspect_ratio
        self.change_img()


    def run(self):
        self.clear_add_listBox1()
        self.compute_map()

        self.window.title("DATASET : " + self.data_info.dataset)
        self.window.geometry('1250x900+350+50')
        self.menubar.add_command(label='QUIT', command=self.window.quit)

        self.data_menu = Menu(self.menubar, tearoff=0)
        self.data_menu.add_command(label='Validation Set', command=self.switch_dataset)
        self.data_menu.add_command(label='Query Set', command=self.switch_dataset)
        self.data_menu.add_command(label='Open Image ...', command=self.open_image)
        self.menubar.add_cascade(label='DATA', menu=self.data_menu)

        self.menubar.add_command(label='Keep aspect ratio', command=self.change_aspect_ratio)

        self.window.config(menu=self.menubar)                               # display the menu
        self.scrollbar1.config(command=self.listBox1.yview)
        self.listBox1.config(yscrollcommand=self.scrollbar1.set)
        self.listBox1_label.grid(row=0, column=0, sticky=N + S + E + W, columnspan=12)

        # find-name componets
        self.find_label.grid(row=20, column=0, sticky=E + W, columnspan=4)
        self.find_entry.grid(row=20, column=4, sticky=E + W, columnspan=4)
        self.find_button.grid(row=20, column=8, sticky=E + W, pady=3, columnspan=4)

        self.scrollbar1.grid(row=30, column=11, sticky=N + S + W)
        self.listBox1.grid(row=30, column=0, sticky=N + S + E + W, pady=3, columnspan=11)

        self.label_img1.grid(row=60, column=0, sticky=N, pady=3, columnspan=12)

        self.frame1.place(x = 375, y = 0, height=900, width=875, anchor=NW)

        # top-20 tab
        self.eval_label.place(x = 0, y = 0, height=20, width=875, anchor=NW)
        pad_x, pad_y, fix_w, fix_h = 5, 50, 170, 160
        for i in range(self.box_num):
            self.label_list[i].place(x = 0 + (fix_w + pad_x)*(i%5), y = 30 + (fix_h + pad_y)*(i//5), height=18, width=fix_w)
            self.label_cls_list[i].place(x = 0 + (fix_w + pad_x)*(i%5), y = 30 + 20 + (fix_h + pad_y)*(i//5), height=18, width=fix_w)
            self.label_img_list[i].place(x = 0 + (fix_w + pad_x)*(i%5), y = 30 + 43 + (fix_h + pad_y)*(i//5), height=fix_h, width=fix_w)

        # ans tab
        self.title_label_ans.place(x = 0, y = 0, height=20, width=875, anchor=NW)
        pad_x, pad_y, fix_w, fix_h = 5, 40, 170, 160
        for i in range(self.box_num):
            self.ans_label_list[i].place(x = 0 + (fix_w + pad_x)*(i%5), y = 30 + (fix_h + pad_y)*(i//5), height=18, width=fix_w)
            self.ans_label_img_list[i].place(x = 0 + (fix_w + pad_x)*(i%5), y = 30 + 30 + (fix_h + pad_y)*(i//5), height=fix_h, width=fix_w)
        
        self.label_img2.place(x = 0, y = 900, anchor=SW)

        # rank tab
        self.title_label_ranking1.pack()
        self.label_img3.pack(pady=10)
        self.title_label_ranking2.pack(pady=0)

        self.panel.pack(pady=10)
        self.label_img4.pack(side = 'left', padx=2)
        self.label_img5.pack(side = 'right', padx=2)
        
        self.listBox1.bind('<<ListboxSelect>>', self.change_img)
        self.listBox1.bind_all('<KeyRelease>', self.eventhandler)

        self.find_entry.bind('<Return>', self.findname)
        self.find_entry.bind('<KP_Enter>', self.findname)
        self.frame1.bind('<<NotebookTabChanged>>', self.on_tab_change)
        self.title_label_ans.bind('<Button-1>', self.on_ans_change)

        self.window.mainloop()


if __name__ == "__main__":
    vis_tool().run()


