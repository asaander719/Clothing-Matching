import torch
from torch.nn.init import uniform_
import torch.nn as nn
import torch.nn.functional as F
from Models.BPRs.BPR import BPR
from Models.BPRs.VTBPR import VTBPR
from Models.BPRs.TextCNN import TextCNN

class PCENET(nn.Module):
    def __init__(self, args, embedding_weight, visual_features, text_features):        
        super(PCENET, self) .__init__()
        self.args = args
        self.weight_P = args.weight_P
        self.hidden_dim = args.hidden_dim
        self.user_num = args.user_num
        self.item_num = args.item_num
        self.with_visual = args.with_visual
        self.with_text = args.with_text
        self.with_Nor = args.with_Nor
        self.cos = args.cos
        #for compatibility space
        self.visual_nn = nn.Sequential(nn.Linear(args.visual_feature_dim, args.visual_feature_dim),nn.Tanh())
        self.visual_nn[0].apply(lambda module: uniform_(module.weight.data,0,0.001))
        self.visual_nn[0].apply(lambda module: uniform_(module.bias.data,0,0.001))
   
        #for personalization space
        self.p_visual_nn = nn.Sequential(nn.Linear(args.visual_feature_dim, args.visual_feature_dim),nn.Tanh())
        self.p_visual_nn[0].apply(lambda module: uniform_(module.weight.data,0,0.001))
        self.p_visual_nn[0].apply(lambda module: uniform_(module.bias.data,0,0.001))

        # self.sigmoid = nn.Sigmoid()
       
        self.user_gama = nn.Embedding(self.user_num, self.hidden_dim)
        self.item_gama = nn.Embedding(self.item_num, self.hidden_dim)
        self.user_bia = nn.Embedding(self.user_num, 1)
        self.item_bia = nn.Embedding(self.item_num, 1)
        
        nn.init.uniform_(self.user_gama.weight, 0, 0.01)
        nn.init.uniform_(self.user_bia.weight, 0, 0.01)
        nn.init.uniform_(self.item_gama.weight, 0, 0.01)
        nn.init.uniform_(self.item_bia.weight, 0, 0.01)

        self.theta_user_visual = nn.Embedding(self.user_num, args.visual_feature_dim)
        nn.init.uniform_(self.theta_user_visual.weight, 0, 0.01)

        
        
        if self.with_visual:
            self.visual_features = visual_features.to(args.device)
        if self.with_text:
            self.text_features = text_features.to(args.device)
            if self.args.dataset == 'IQON3000':
                self.max_sentense_length = args.max_sentence
                self.text_embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=False)
                self.textcnn = TextCNN(args.textcnn_layer, sentence_size=(args.max_sentence, args.text_feature_dim), output_size=100 * args.textcnn_layer)
                self.text_nn = nn.Sequential(nn.Linear(100 * args.textcnn_layer, 100 * args.textcnn_layer),nn.ReLU()) 
                self.text_nn.append(nn.Sequential(nn.Linear(100 * args.textcnn_layer, 100 * args.textcnn_layer),nn.Softmax()) )
                # self.p_text_nn = nn.Sequential(nn.Linear(100 * args.textcnn_layer, 100 * args.textcnn_layer),nn.ReLU())
                # self.p_text_nn.append(nn.Sequential(nn.Linear(100 * args.textcnn_layer, 100 * args.textcnn_layer),nn.Softmax()))
                self.theta_user_text = nn.Embedding(self.user_num, 100 * args.textcnn_layer)
                nn.init.uniform_(self.theta_user_text.weight, 0, 0.01)
            elif self.args.dataset == 'Polyvore_519':
                self.text_nn = nn.Sequential(nn.Linear(args.text_feature_dim, args.text_feature_dim),nn.ReLU()) 
                self.text_nn.append(nn.Sequential(nn.Linear(args.text_feature_dim, args.text_feature_dim),nn.Softmax()) )
                # self.p_text_nn = nn.Sequential(nn.Linear(args.text_feature_dim, self.hidden_dim),nn.Sigmoid())
                self.theta_user_text = nn.Embedding(self.user_num, args.text_feature_dim)
                nn.init.uniform_(self.theta_user_text.weight, 0, 0.01)

        self.text_nn[0].apply(lambda module: uniform_(module.weight.data,0,0.001))
        self.text_nn[0].apply(lambda module: uniform_(module.bias.data,0,0.001))
        # self.p_text_nn[0].apply(lambda module: uniform_(module.weight.data,0,0.001))
        # self.p_text_nn[0].apply(lambda module: uniform_(module.bias.data,0,0.001))

        # self.vtbpr = VTBPR(self.user_num, self.item_num, hidden_dim=self.hidden_dim, 
        #     theta_text=self.with_text, theta_visual=self.with_visual, with_Nor=True, cos=True)
        # print('Module already prepared, {} users, {} items'.format(self.user_num, self.item_num))
        # self.bpr = BPR(self.user_num, self.item_num)
         
    def forward(self, batch, train =True, **args):
        Us = batch[0] #bs
        Is = batch[1]
        Js = batch[2]
        Ks = batch[3]

        user_gama = self.user_gama(Us)
        user_bia = self.user_bia(Us)
        item_gama_j = self.item_gama(Js)
        item_bia_j = self.item_bia(Js)
        item_gama_k = self.item_gama(Ks)
        item_bia_k = self.item_bia(Ks)
        batchsize = len(Us)

        if self.with_visual:
            vis_I = self.visual_features[Is] #bs,visual_feature_dim = 2048 = torch.Size([256, 2048])
            vis_J = self.visual_features[Js]
            vis_K = self.visual_features[Ks]
  
            I_visual_latent = self.visual_nn(vis_I) #bs, hidden_dim =torch.Size([256, 512])
            J_visual_latent = self.visual_nn(vis_J)
            K_visual_latent = self.visual_nn(vis_K)

            if self.cos:
                visual_ij = F.cosine_similarity(I_visual_latent*vis_I, J_visual_latent*vis_J, dim=-1)
                visual_ik = F.cosine_similarity(I_visual_latent*vis_I, K_visual_latent*vis_K, dim=-1)
            else:
                visual_ij = torch.sum((I_visual_latent*vis_I) * (J_visual_latent*vis_J), dim=-1)
                visual_ik = torch.sum((I_visual_latent*vis_I) * (K_visual_latent*vis_K), dim=-1)

        if self.with_text:
            if self.args.dataset == 'IQON3000':
                text_I = self.text_embedding(self.text_features[Is].long()) #256,83,300 text_I = self.text_embedding(self.text_features[Is]) 
                text_J = self.text_embedding(self.text_features[Js].long())
                text_K = self.text_embedding(self.text_features[Ks].long())

                I_text_fea = self.textcnn(text_I.unsqueeze(1))  #256,400
                J_text_fea = self.textcnn(text_J.unsqueeze(1))
                K_text_fea = self.textcnn(text_K.unsqueeze(1))

                I_text_latent = self.text_nn(I_text_fea) #256,512
                J_text_latent = self.text_nn(J_text_fea)
                K_text_latent = self.text_nn(K_text_fea)
                if self.cos:
                    text_ij = F.cosine_similarity(I_text_latent*I_text_fea, J_text_latent*J_text_fea, dim=-1)
                    text_ik = F.cosine_similarity(I_text_latent*I_text_fea, K_text_latent*K_text_fea, dim=-1)
                else:
                    text_ij = torch.sum((I_text_latent*I_text_fea) * (J_text_latent*J_text_fea), dim=-1)
                    text_ik = torch.sum((I_text_latent*I_text_fea) * (K_text_latent*K_text_fea), dim=-1)

            elif self.args.dataset == 'Polyvore_519':
                text_I = self.text_features[Is] #256,83,300
                text_J = self.text_features[Js]
                text_K = self.text_features[Ks]

                I_text_latent = self.text_nn(text_I) #256,512
                J_text_latent = self.text_nn(text_J)
                K_text_latent = self.text_nn(text_K)

                if self.cos:
                    text_ij = F.cosine_similarity(I_text_latent*text_I, J_text_latent*text_J, dim=-1)
                    text_ik = F.cosine_similarity(I_text_latent*text_I, K_text_latent*text_K, dim=-1)
                else:
                    text_ij = torch.sum((I_text_latent*text_I) * (J_text_latent*text_J), dim=-1)
                    text_ik = torch.sum((I_text_latent*text_I) * (K_text_latent*text_K), dim=-1)

        if self.with_visual and self.with_text:
            if self.cos:
                cuj = item_bia_j.view(batchsize) + user_bia.view(batchsize) + F.cosine_similarity(user_gama, item_gama_j, dim=-1)
                cuk = item_bia_k.view(batchsize) + user_bia.view(batchsize) + F.cosine_similarity(user_gama, item_gama_k, dim=-1)
            else:    
                cuj = item_bia_j.view(batchsize) + user_bia.view(batchsize) + torch.sum(user_gama * item_gama_j, dim=-1)
                cuk = item_bia_k.view(batchsize) + user_bia.view(batchsize) + torch.sum(user_gama * item_gama_k, dim=-1)

            theta_user_visual = self.theta_user_visual(Us)
            if self.cos:
                ui_visual_j = F.cosine_similarity(theta_user_visual, J_visual_latent*vis_J, dim=-1)
                ui_visual_k = F.cosine_similarity(theta_user_visual, K_visual_latent*vis_K, dim=-1)
            else:
                ui_visual_j = torch.sum(theta_user_visual * (J_visual_latent*vis_J), dim=-1)
                ui_visual_k = torch.sum(theta_user_visual * (K_visual_latent*vis_K), dim=-1)
            cuj += ui_visual_j
            cuk += ui_visual_k

            theta_user_text = self.theta_user_text(Us)  
            if self.args.dataset == 'IQON3000':
                if self.cos:
                    ui_text_j = F.cosine_similarity(theta_user_text, J_text_latent*J_text_fea, dim=-1)
                    ui_text_k = F.cosine_similarity(theta_user_text, K_text_latent*K_text_fea, dim=-1)
                else:
                    ui_text_j = torch.sum(theta_user_text * (J_text_latent*J_text_fea), dim=-1)
                    ui_text_k = torch.sum(theta_user_text * (K_text_latent*K_text_fea), dim=-1)
            elif self.args.dataset == 'Polyvore_519':
                if self.cos:
                    ui_text_j = F.cosine_similarity(theta_user_text, J_text_latent*text_J, dim=-1)
                    ui_text_k = F.cosine_similarity(theta_user_text, K_text_latent*text_K, dim=-1)
                else:
                    ui_text_j = torch.sum(theta_user_text * (J_text_latent*text_J), dim=-1)
                    ui_text_k = torch.sum(theta_user_text * (K_text_latent*text_K), dim=-1)
           
            cuj += ui_text_j
            cuk += ui_text_k

            p_ij = 0.5 * (visual_ij + text_ij)
            p_ik = 0.5 * (visual_ik + text_ik)

            pred = self.weight_P * p_ij + (1 - self.weight_P) * cuj - (self.weight_P * p_ik + (1 - self.weight_P) * cuk)
   
        return pred   

    def inference(self, batch, train=False, **args):
        Us = batch[0] #bs
        Is = batch[1]
        Js = batch[2]
        # Ks = batch[3]
        Ks_list = batch[3] #torch.Size([256, 1])

        bs = len(Us)
        candi_num = 1 + Ks_list.size(0)  # one positive + all negative #257

        user_gama = self.user_gama(Us)
        user_bia = self.user_bia(Us)
        item_gama_j = self.item_gama(Js)
        item_bia_j = self.item_bia(Js)
        
        if self.args.wide_evaluate:
            item_gama_k = self.item_gama(Ks_list.squeeze(1))
            item_bia_k = self.item_bia(Ks_list.squeeze(1))
        else:
            item_gama_k = self.item_gama(Ks_list)
            item_bia_k = self.item_bia(Ks_list)
     
        if self.with_visual:
            vis_I = self.visual_features[Is] #bs,visual_feature_dim = 2048 = torch.Size([256, 2048])
            vis_J = self.visual_features[Js]
            if self.args.wide_evaluate:
                vis_K = self.visual_features[Ks_list.squeeze(1)] 
            else:
                vis_K = self.visual_features[Ks_list] 
  
            I_visual_latent = self.visual_nn(vis_I) #bs, hidden_dim =torch.Size([256, 512])
            J_visual_latent = self.visual_nn(vis_J)
            K_visual_latent = self.visual_nn(vis_K)

            I_visual_latent, Jks_visual_latent = self.wide_infer(bs, candi_num, J_visual_latent, K_visual_latent, I_visual_latent)
            vis_I, Jks_visual= self.wide_infer(bs, candi_num, vis_J, vis_K, vis_I)
            if self.cos:
                visual_ijs_score = F.cosine_similarity(I_visual_latent*vis_I, Jks_visual_latent*Jks_visual, dim=-1)    #256,257    
            else:
                visual_ijs_score = torch.sum((I_visual_latent*vis_I) * (Jks_visual_latent*Jks_visual), dim=-1)

        if self.with_text:
            if self.args.dataset == 'IQON3000':
                text_I = self.text_embedding(self.text_features[Is].long()) #256,83,300 text_I = self.text_embedding(self.text_features[Is]) 
                text_J = self.text_embedding(self.text_features[Js].long())
                if self.args.wide_evaluate:
                    text_K = self.text_embedding(self.text_features[Ks_list.squeeze(1)].long())
                else:
                    text_K = self.text_embedding(self.text_features[Ks_list].long())

                I_text_fea = self.textcnn(text_I.unsqueeze(1))  #256,400
                J_text_fea = self.textcnn(text_J.unsqueeze(1))
                K_text_fea = self.textcnn(text_K.unsqueeze(1))

                I_text_latent = self.text_nn(I_text_fea) #256,512
                J_text_latent = self.text_nn(J_text_fea)
                K_text_latent = self.text_nn(K_text_fea)

                I_text_latent, Jks_text_latent = self.wide_infer(bs, candi_num, J_text_latent, K_text_latent, I_text_latent)
                I_text_fea, Jks_text = self.wide_infer(bs, candi_num, J_text_fea, K_text_fea, I_text_fea)
                if self.cos:
                    text_ijks = F.cosine_similarity(I_text_latent*I_text_fea, Jks_text_latent*Jks_text, dim=-1)
                else:
                    text_ijks = torch.sum((I_text_latent*I_text_fea) * (Jks_text_latent*Jks_text), dim=-1)

            elif self.args.dataset == 'Polyvore_519':
                text_I = self.text_features[Is] #256,83,300
                text_J = self.text_features[Js]
                if self.args.wide_evaluate:
                    text_K = self.text_features[Ks_list.squeeze(1)]
                else:
                    text_K = self.text_features[Ks_list]

                I_text_latent = self.text_nn(text_I) #256,512
                J_text_latent = self.text_nn(text_J)
                K_text_latent = self.text_nn(text_K)

                I_text_latent, Jks_text_latent = self.wide_infer(bs, candi_num, J_text_latent, K_text_latent, I_text_latent)
                text_I, Jks_text = self.wide_infer(bs, candi_num, text_J, text_K, text_I)
                if self.cos:
                    text_ijks = F.cosine_similarity(I_text_latent*text_I, Jks_text_latent, dim=-1)
                else:
                    text_ijks = torch.sum((I_text_latent*text_I) * (Jks_text_latent*Jks_text), dim=-1)

        if self.with_visual and self.with_text:
            user_gama, item_gama = self.wide_infer(bs, candi_num, item_gama_j, item_gama_k, user_gama) # bs, candi_num, hd
            user_bia, item_bia = self.wide_infer(bs, candi_num, item_bia_j, item_bia_k, user_bia) # bs, candi_num, 1
            if self.cos:
                score_c = item_bia.view(bs, candi_num) + user_bia.view(bs, candi_num) + F.cosine_similarity(user_gama, item_gama, dim=-1) #bs, candi_num
            else:    
                score_c = item_bia.view(bs, candi_num) + user_bia.view(bs, candi_num) + torch.sum(user_gama * item_gama, dim=-1)
      
            theta_user_visual = self.theta_user_visual(Us)
            theta_user_visual, visual_features = self.wide_infer(bs, candi_num, J_visual_latent*vis_J, K_visual_latent*vis_K, theta_user_visual)
            if self.cos:
                ui_visual = F.cosine_similarity(theta_user_visual, visual_features, dim=-1)
            else:
                ui_visual = torch.sum(theta_user_visual * visual_features, dim=-1)
            score_c += ui_visual

            theta_user_text = self.theta_user_text(Us)
            if self.args.dataset == 'IQON3000':
                theta_user_text, textural_features = self.wide_infer(bs, candi_num, J_text_latent*J_text_fea, K_text_latent*K_text_fea, theta_user_text)
            elif self.args.dataset == 'Polyvore_519':
                theta_user_text, textural_features = self.wide_infer(bs, candi_num, J_text_latent*text_J, K_text_latent*text_K, theta_user_text)
            if self.cos:
                ui_text = F.cosine_similarity(theta_user_text, textural_features, dim=-1)
            else:
                ui_text = torch.sum(theta_user_text * textural_features, dim=-1) 
            score_c += ui_text  
            
            score_p = 0.5 * (visual_ijs_score + text_ijks)  
            score = self.weight_P * score_p  + (1 - self.weight_P) * score_c
        return score
    
    def wide_infer(self, bs, candi_num, J, K, I):
        J = J.unsqueeze(1) #256,1,512
        K = K.unsqueeze(0).expand(bs, -1, -1) #1,256,512->256,256,512
        Jks = torch.cat([J, K], dim=1) #256,257,512 # dim=1 里面第一个为postive target(1+256)
        I= I.unsqueeze(1).expand(-1, candi_num, -1) # 256,257,512
        return I, Jks    