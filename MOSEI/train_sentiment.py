import numpy as np, torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
import argparse, time, pickle, os
from itertools import chain
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support
from model import CategoricalModel, MaskedNLLLoss
from dataloader import MOSEICategorical_Sentiment

from sentence_transformers import losses, SentenceTransformer
from config import MOSEI

np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_MOSEI_loaders(path, batch_size=128, num_workers=0, pin_memory=False):
    trainset = MOSEICategorical_Sentiment(path=path,  train= True,  bert_vectors = PATH.BERT_VECTORS, siamese_vectors = PATH.SBERT_VECTORS, visual_vectors = PATH.VISUAL_VECTORS)
    validset = MOSEICategorical_Sentiment(path=path, valid= True,  bert_vectors = PATH.BERT_VECTORS, siamese_vectors = PATH.SBERT_VECTORS, visual_vectors = PATH.VISUAL_VECTORS)
    # trainset = MOSEICategorical5(path=path, bert_vectors = 'sbert_vectors.p')
    
    train_loader = DataLoader(trainset, batch_size=batch_size, collate_fn=trainset.collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = DataLoader(validset, batch_size=batch_size,  collate_fn=validset.collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    testset = MOSEICategorical_Sentiment(path=path, train=False,   bert_vectors = PATH.BERT_VECTORS, siamese_vectors = PATH.SBERT_VECTORS, visual_vectors = PATH.VISUAL_VECTORS)
    # testset = MOSEICategorical5(path=path, train = False, bert_vectors = 'sbert_vectors.p')
    test_loader = DataLoader(testset,  batch_size=batch_size, collate_fn=testset.collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader

def train_or_eval_model(model,loss_function, dataloader, optimizer=None, train=False):    
    count = 0
    losses, losses_siamese, preds, preds_siamese, labels, labels_siamese, masks, alphas_f, alphas_b, vids = [], [], [], [], [], [], [], [], [], []
    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()
    for data in dataloader:
        count+=1
        if train:
            optimizer.zero_grad()
        textf, visuf, acouf, prior_textf, prior_visuf, prior_acouf, textf_, qmask, umask, label, prior, siamese_label =  [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        log_prob , log_siamese_prob, alpha_f,alpha_b  = model(textf, acouf, visuf, textf, textf_, prior_textf, qmask, umask, softmax_loss)   
        

        lp_ = log_prob.transpose(0,1).contiguous().view(-1,log_prob.size()[2]) 
        lp_siamese_ = log_siamese_prob.transpose(0,1).contiguous().view(-1,log_siamese_prob.size()[2]) 

        labels_ = label.view(-1) 
        siamese_labels_ = siamese_label.view(-1)

        loss = loss_function(lp_, labels_, umask)
        loss_siamese = loss_function2(lp_siamese_,siamese_labels_, umask)
        
        pred_siamese_ = torch.argmax(lp_siamese_, 1)
        pred_ = torch.argmax(lp_,1)

        preds.append(pred_.data.cpu().numpy())
        preds_siamese.append(pred_siamese_.data.cpu().numpy())
        
        labels.append(labels_.data.cpu().numpy())
        labels_siamese.append(siamese_labels_.data.cpu().numpy())
        
        masks.append(umask.view(-1).cpu().numpy())
        losses.append(loss.item()*masks[-1].sum())
        losses_siamese.append(loss_siamese.item()*masks[-1].sum())
        loss += loss_siamese
        if train:
            loss.backward()
            optimizer.step()
        else:
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids += data[-1]
    if preds!=[]:
        preds  = np.concatenate(preds)
        preds_siamese = np.concatenate(preds_siamese)

        labels = np.concatenate(labels)
        labels_siamese = np.concatenate(labels_siamese)

        masks  = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'),[]
    avg_loss = round(np.sum(losses)/np.sum(masks),4)
    avg_loss_siamese = round(np.sum(losses_siamese)/np.sum(masks),4)
    if train:
        method = "train"
    else:
        method = "val/test"

    
    avg_accuracy = round(accuracy_score(labels,preds,sample_weight=masks)*100,2)
    avg_accuracy_siamese = round(accuracy_score(labels_siamese,preds_siamese,sample_weight=masks)*100,2)

    avg_fscore = round(f1_score(labels,preds,sample_weight=masks,average='weighted')*100,2)
    avg_fscore_siamese = round(f1_score(labels_siamese,preds_siamese,sample_weight=masks,average='weighted')*100,2)
    
    print(f'avg_{method}_loss',avg_loss,f'avg_{method}_siamese_loss',avg_loss_siamese)
    print(f'avg_{method}_accuracy',avg_accuracy,f'avg_{method}_siamese_accuracy',avg_accuracy_siamese)
    print(f'avg_{method}_fscore',avg_fscore,f'avg_{method}_siamese_fscore',avg_fscore_siamese)

    if method == "val/test":
        print(f'SIAMESE REPORT {method}')
        print(classification_report(labels_siamese,preds_siamese,sample_weight=masks,digits=4))
        print(confusion_matrix(labels_siamese,preds_siamese,sample_weight=masks))


    print(f'EMOTION PREDICTION REPORT {method}')
    print(classification_report(labels,preds,sample_weight=masks,digits=4))
    print(confusion_matrix(labels,preds,sample_weight=masks))
    # print(f'avg_{method}_accuracy',avg_accuracy,)
    return avg_loss, avg_accuracy, labels, preds, masks,avg_fscore, [alphas_f, alphas_b, vids]

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Trains a categorical model for sentiment data with 1 as positive sentiment and 0 as negative sentiment")
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--rec-dropout', type=float, default=0.1, metavar='rec_dropout', help='rec_dropout rate')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=128, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=100, metavar='E', help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=True, help='class weight')
    parser.add_argument('--log_dir', type=str, default='logs/mosei_categorical', help='Directory for tensorboard logs')
    parser.add_argument('--model_save_name', type=str, required=True, help='Name of model file')
    parser.add_argument('--gpu', type=int, required=True, help='gpu core')

    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok = True)
    writer = SummaryWriter(args.log_dir)
    print(args)
    torch.cuda.set_device(args.gpu)

    
    # Run on either GPU or CPU
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')
    print("Tensorboard logs in " + args.log_dir)

    PATH = MOSEI


    batch_size = args.batch_size
    n_classes  = 2
    cuda       = args.cuda
    n_epochs   = args.epochs

    D_m_text, D_m_audio, D_m_video, D_m_context = 768, 384, 711, 768
    D_g, D_p, D_e, D_h, D_a = 150, 150, 100, 100, 100

    # Instantiate model
    model = CategoricalModel(D_m_text, D_m_audio, D_m_video, D_m_context, D_g, D_p, D_e, D_h, n_classes=n_classes, dropout_rec=args.rec_dropout, dropout=args.dropout)

    if cuda:
        model.cuda()
    
    loss_weights = torch.FloatTensor([1/0.2903, 1/0.7097])
    
    if args.class_weight:
        loss_function  = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
    else:
        loss_function = MaskedNLLLoss()
    
    loss_function2 = MaskedNLLLoss(torch.FloatTensor([1/0.6604, 1/0.3396]).cuda())
    
    sbert_model = SentenceTransformer(PATH.SIAMESE_MODEL, device = 'cuda')
    softmax_loss = losses.SoftmaxLoss(model=sbert_model, sentence_embedding_dimension=sbert_model.get_sentence_embedding_dimension(), num_labels=2).cuda()
    softmax_loss.load_state_dict(torch.load(PATH.SIAMESE_CLASSIFIER, map_location=f'cuda:{args.gpu}'))

    optimizer = optim.Adam(chain(model.parameters(),softmax_loss.parameters()), lr=args.lr, weight_decay=args.l2)
    train_loader, valid_loader, test_loader = get_MOSEI_loaders(PATH.CATEGORICAL_DATA, batch_size=batch_size, num_workers=0)
    best_loss, best_label, best_pred, best_mask, best_fscore = None, None, None, None, None

    
    # Training loop
    for e in tqdm(range(n_epochs), desc = 'MOSEI Categorical'):
        train_loss, train_acc, _,_,_,train_fscore,_ = train_or_eval_model(model, loss_function, train_loader, optimizer, True)
        val_loss, val_acc, val_label, val_pred, val_mask, val_fscore, attentions = train_or_eval_model(model,loss_function, valid_loader)
        writer.add_scalar("Train Loss - MOSEI Sentiment", train_loss, e)
        writer.add_scalar("Val Loss - MOSEI Sentiment", val_loss, e)

        print(e,'Train loss: ',train_loss,'Train acc: ', train_acc,'Train fscore: ',train_fscore, flush=True)
        print(e,'Val loss: ',val_loss,'Val acc: ', val_acc,'Val fscore: ',val_fscore, flush=True)
        
        if best_fscore == None or best_fscore < val_fscore:
            torch.save(model.state_dict(), args.model_save_name)
            best_loss, best_label, best_pred, best_mask, best_attn = val_loss, val_label, val_pred, val_mask, attentions


    model.load_state_dict(torch.load(args.model_save_name))
    test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions = train_or_eval_model(model,loss_function, test_loader)
    print('Test performance..')
    print('Loss {} accuracy {}'.format(test_loss, round(accuracy_score(test_label,test_pred,sample_weight=test_mask)*100,2)))
    print(classification_report(test_label,test_pred,sample_weight=test_mask,digits=4))
    print(confusion_matrix(test_label,test_pred,sample_weight=test_mask))
