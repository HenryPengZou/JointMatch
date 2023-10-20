
import os
import random
import sys

import numpy as np
import pandas as pd
import preprocessor as p
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn



#### Set Path ####
# go to the directory of this file
os.chdir(os.path.dirname(os.path.abspath(__file__)))
root = '../'
root_code = '../code'
sys.path.append(root)
sys.path.append(root_code)
print('current work directory: ', os.getcwd())


def oneRun(log_dir, output_dir_experiment, **params):
    """ Run one experiment """
    ##### Default Setting #####
    ## Set input data path
    try:
        data_path = root + 'data/' + params['dataset']
        print('\ndata_path: ', data_path)
    except:
        data_path = root + 'data/ag_news'
        print('\ndata_path is not specified, use default path: ', data_path)

    ## Set output directory: used to store saved model and other outputs
    output_dir_path = output_dir_experiment

    ## Set default hyperparameters
    n_labeled_per_class = 10        if 'n_labeled_per_class' not in params else params['n_labeled_per_class']
    bs = 8                          if 'bs' not in params else params['bs']      # original: 32
    ul_ratio = 10                   if 'ul_ratio' not in params else params['ul_ratio']     
    lr = 2e-5                       if 'lr' not in params else params['lr']      # original: 1e-4, 2e-5  
    lr_linear = 1e-3                if 'lr_linear' not in params else params['lr_linear'] # original: 1e-3      

    # - semi-supervised 
    weight_u_loss =0.1              if 'weight_u_loss' not in params else params['weight_u_loss']
    load_mode = 'semi'              if 'load_mode' not in params else params['load_mode'] # semi, sup_baseline

    # - pseudo-labeling
    psl_threshold_h = 0.98          if 'psl_threshold_h' not in params else params['psl_threshold_h']   # original: 0.75
    labeling_mode = 'hard'          if 'labeling_mode' not in params else params['labeling_mode'] # hard, soft
    adaptive_threshold = False      if 'adaptive_threshold' not in params else params['adaptive_threshold'] # True, False

    # - ensemble
    num_nets = 2                    if 'num_nets' not in params else params['num_nets']
    cross_labeling = False          if 'cross_labeling' not in params else params['cross_labeling'] # True, False

    # - weight disagreement
    weight_disagreement = False     if 'weight_disagreement' not in params else params['weight_disagreement'] # True, False
    disagree_weight = 1             if 'disagree_weight' not in params else params['disagree_weight']

    # - ema
    ema_mode = False                if 'ema_mode' not in params else params['ema_mode']
    ema_momentum = 0.9              if 'ema_momentum' not in params else params['ema_momentum'] # original: 0.99

    # - others
    seed = params['seed']
    device_idx = 1                  if 'device_idx' not in params else params['device_idx']
    val_interval = 25               if 'val_interval' not in params else params['val_interval'] # 20, 25
    early_stop_tolerance = 10       if 'early_stop_tolerance' not in params else params['early_stop_tolerance'] # 5, 6, 10
    max_epoch = 10000                if 'max_epoch' not in params else params['max_epoch'] # 100, 200
    max_step = 100000               if 'max_step' not in params else params['max_step'] # 100000, 200000


    ## Set random seed and device
    # Fix random seed
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    # Check & set device
    if torch.cuda.is_available():     
        device = torch.device("cuda", device_idx)
        print('\nThere are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU-', device_idx, torch.cuda.get_device_name(device_idx))
    else:
        print('\nNo GPU available, using the CPU instead.')
        device = torch.device("cpu")




    #### Load Data ####
    from utils.dataloader import get_dataloader
    train_labeled_loader, train_unlabeled_loader, dev_loader, test_loader, n_classes = get_dataloader(data_path, n_labeled_per_class, bs, load_mode)
    print('n_classes: ', n_classes)
    # # used for degugging
    # sys.exit()




    ##### Model & Optimizer & Learning Rate Scheduler #####
    from models.netgroup import NetGroup

    # Initialize model & optimizer & lr_scheduler
    net_arch = 'bert-base-uncased'
    netgroup = NetGroup(net_arch, num_nets, n_classes, device, lr, lr_linear)

    # Initialize EMA
    netgroup.train()
    if ema_mode:
        netgroup.init_ema(ema_momentum)




    ##### Training & Evaluation #####
    ## Set or import criterions & helper functions
    from utils.helper import format_time
    from criterions.criterions import ce_loss, consistency_loss


    ## Evaluation
    # define evaluation metrics
    # from torchmetrics import F1Score
    from sklearn.metrics import f1_score, accuracy_score
    from torchmetrics import Accuracy
    from torchmetrics.classification import MulticlassConfusionMatrix
    # f1 = F1Score(num_classes=n_classes, average='macro')
    # accuracy = Accuracy(num_classes=n_classes, average='micro')
    accuracy_classwise = Accuracy(num_classes=n_classes, average='none')
    confusion_matrix = MulticlassConfusionMatrix(num_classes=n_classes)


    @torch.no_grad() # no need to track gradients in validation
    def evaluation(loader, final_eval=False):
        """Evaluation"""
        # Put the model in evaluation mode--the dropout layers behave differently during evaluation.
        netgroup.eval()
        if ema_mode: # use ema model for evaluation
            netgroup.eval_ema()

        # Tracking variables
        preds_all = []
        target_all = []

        # Evaluate data for one epoch
        for batch in loader:
            b_labels = batch['label'].to(device)
            
            # forward pass
            outs = netgroup.forward(batch['x'], b_labels)

            # take average probs from all nets
            probs = torch.mean(torch.softmax(torch.stack(outs), dim=2), dim=0)

            # Move preds and labels to CPU
            preds = torch.argmax(probs, dim=1)
            target = b_labels

            # For calculating classwise acc
            preds_all.append(preds)
            target_all.append(target)


        # Calculate acc and macro-F1
        preds_all = torch.cat(preds_all).detach().cpu()
        target_all = torch.cat(target_all).detach().cpu()
        acc = accuracy_score(target_all, preds_all)
        f1 = f1_score(target_all, preds_all, average='macro')

        # Calculate classwise acc
        accuracy_classwise_ = accuracy_classwise(preds_all, target_all).numpy().round(3)

        if final_eval:
            # compute confusion matrix for the final evaluation on the saved model
            confmat_result = confusion_matrix(preds_all, target_all)
            return acc, f1, list(accuracy_classwise_), confmat_result
        else:
            return acc, f1, list(accuracy_classwise_)



    ## Training
    import time
    import torch.nn.functional as F
    from utils.helper import freematch_fairness_loss

    # Initialize variables
    t0 = time.time() # Measure how long the training takes.
    step = 0
    best_acc = 0
    best_model_step = 0
    pslt_global = 0
    psl_total_eval = 0
    psl_correct_eval = 0
    cw_psl_total, cw_psl_correct = torch.zeros(n_classes, dtype=int), torch.zeros(n_classes, dtype=int)
    cw_psl_total_eval, cw_psl_correct_eval = torch.zeros(n_classes, dtype=int), torch.zeros(n_classes, dtype=int)
    cw_psl_total_accum, cw_psl_correct_accum = torch.zeros(n_classes, dtype=int), torch.zeros(n_classes, dtype=int)
    training_stats = []
    early_stop_flag = False
    cw_avg_prob = (torch.ones(n_classes) / n_classes).to(device)    # estimate learning stauts of each class
    local_threshold = torch.zeros(n_classes, dtype=int)

    # Training
    netgroup.train()
    for epoch in range(max_epoch):
        if step > max_step:
            break
        if early_stop_flag:
            break
        for batch_label in train_labeled_loader:
            ## --- Evaluation: Check Performance on Validation set every val_interval batches ---##
            if step % val_interval == 0:   
                # acc_test, f1_test, acc_test_cw = evaluation(test_loader)
                acc_val, f1_val, acc_val_cw = evaluation(dev_loader)
                acc_train, f1_train, acc_train_cw = evaluation(train_labeled_loader)
                
                # restore training mode 
                if ema_mode:
                    netgroup.train_ema()
                netgroup.train()

                print('>>Epoch %d Step %d acc_val %f f1_val %f acc_train %f f1_train %f psl_cor %d psl_totl %d pslt_global %f' % 
                        (epoch, step, acc_val, f1_val, acc_train, f1_train, psl_correct_eval, psl_total_eval, pslt_global),
                        'Tim {:}'.format(format_time(time.time() - t0)))
                
                # Record all statistics from this evaluation.
                acc_psl = (psl_correct_eval/psl_total_eval) if psl_total_eval > 0 else None

                training_stats.append(
                    {   'step': step,
                        'acc_val': acc_val,
                        'f1_val': f1_val,
                        'acc_train': acc_train,
                        'f1_train': f1_train,
                        'psl_correct': psl_correct_eval,   
                        'psl_total': psl_total_eval,
                        'acc_psl': acc_psl, 
                        'pslt_global': pslt_global,  
                        'cw_acc_train': acc_train_cw,
                        'cw_acc_val': acc_val_cw,
                        'cw_avg_prob': cw_avg_prob.tolist(),
                        'local_threshold': local_threshold.tolist(),
                        # 'cw_psl_total': cw_psl_total.tolist(),
                        # 'cw_psl_correct': cw_psl_correct.tolist(),  
                        'cw_psl_total_eval': cw_psl_total_eval.tolist(),
                        'cw_psl_correct_eval': cw_psl_correct_eval.tolist(),
                        'cw_psl_acc_eval': (cw_psl_correct_eval/cw_psl_total_eval).tolist(),
                        'cw_psl_total_accum': cw_psl_total_accum.tolist(),
                        'cw_psl_correct_accum': cw_psl_correct_accum.tolist(),
                        'cw_psl_acc_accum': (cw_psl_correct_accum/cw_psl_total_accum).tolist(),
                    })

                # check classwise psl accuracy and total psl accuracy for the current eval
                print('cw_psl_eval: ', cw_psl_total_eval.tolist(), cw_psl_correct_eval.tolist())
                print('psl_acc: ', round((psl_correct_eval/psl_total_eval),3), end=' ') if psl_total_eval > 0 else print('psl_acc: None', end=' ')
                print('cw_psl_acc: ', (cw_psl_correct_eval/cw_psl_total_eval).tolist())

                # Early stopping & Save best model
                # - best criterion: acc_val 
                if acc_val > best_acc:
                    best_acc = acc_val
                    best_model_step = step
                    early_stop_count = 0
                    netgroup.save_model(output_dir_path, 'model', ema_mode=ema_mode)
                else:
                    early_stop_count+=1
                    if early_stop_count >= early_stop_tolerance:
                        early_stop_flag = True
                        print('Early stopping trigger at step: ', step)
                        print('Best model at step: ', best_model_step)
                        break

                # initialize pseudo labels evaluation
                psl_total_eval, psl_correct_eval = 0, 0
                cw_psl_total_eval, cw_psl_correct_eval = torch.zeros(n_classes, dtype=int), torch.zeros(n_classes, dtype=int)


            ## --- Training --- ##
            step += 1
            ## Process Labeled Data
            x_lb, y_lb = batch_label['x_w'], batch_label['label']
            # forward pass
            outs_x_lb = netgroup.forward(x_lb, y_lb.to(device))
            # compute loss for labeled data
            sup_loss_nets = [ce_loss(outs_x_lb[i], y_lb.to(device)) for i in range(num_nets)]
            # update netgorup from loss of labeled data
            netgroup.update(sup_loss_nets)
            if ema_mode:
                netgroup.update_ema()


            ## Process Unlabeled Data
            if load_mode == 'semi':
                for _ in range(ul_ratio):
                    # unlabeled data input for each round/batch
                    try:
                        batch_unlabel = next(data_iter_unl)
                    except:
                        data_iter_unl = iter(train_unlabeled_loader)
                        batch_unlabel = next(data_iter_unl)
                        
                    x_ulb_w, x_ulb_s = batch_unlabel['x_w'], batch_unlabel['x_s']

                    # forward pass
                    outs_x_ulb_s_nets = netgroup.forward(x_ulb_s)
                    with torch.no_grad(): # stop gradient for weak augmentation brach
                        outs_x_ulb_w_nets = netgroup.forward(x_ulb_w)

                    ## Generate pseudo labels and masks for all nets in one batch of unlabeled data
                    pseudo_labels_nets = []
                    u_psl_masks_nets = []

                    for i in range(num_nets):
                        ## generate pseudo labels
                        logits_x_ulb_w = outs_x_ulb_w_nets[i]
                        if labeling_mode=='soft':
                            pseudo_labels_nets.append(torch.softmax(logits_x_ulb_w, dim=-1))
                        else:
                            max_idx = torch.argmax(logits_x_ulb_w, dim=-1)
                            pseudo_labels_nets.append(F.one_hot(max_idx, num_classes=n_classes).to(device))

                        ## compute mask for pseudo labels
                        probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
                        max_probs, max_idx = torch.max(probs_x_ulb_w, dim=-1)
                        if not adaptive_threshold:      
                            # fixed hard threshold
                            pslt_global = psl_threshold_h
                            u_psl_masks_nets.append(max_probs >= pslt_global)
                        else:
                            # adaptive local threshold
                            pslt_global = psl_threshold_h
                            cw_avg_prob = cw_avg_prob * ema_momentum + torch.mean(probs_x_ulb_w, dim=0) * (1-ema_momentum)
                            local_threshold = cw_avg_prob / torch.max(cw_avg_prob,dim=-1)[0]
                            u_psl_mask = max_probs.ge(pslt_global * local_threshold[max_idx])
                            u_psl_masks_nets.append(u_psl_mask)

                    ## Compute loss for unlabeled data for all nets
                    total_unsup_loss_nets = []
                    for i in range(num_nets):
                        if cross_labeling:
                            # cross labeling and masking
                            pseudo_label = pseudo_labels_nets[(i+1)%num_nets]
                            u_psl_mask = u_psl_masks_nets[(i+1)%num_nets]
                        else:
                            # vanilla labeling
                            pseudo_label = pseudo_labels_nets[i]
                            u_psl_mask = u_psl_masks_nets[i]

                        if weight_disagreement == 'True':
                            # obtain the mask for disagreement and agreement across nets, note that they are derived from confident pseudo-label mask
                            # disagree_weight, agree_weight can be a specified scalar or a tensor calculated based on disagreement score
                            disagree_mask = torch.logical_xor(u_psl_masks_nets[(i)%num_nets], u_psl_masks_nets[(i+1)%num_nets])
                            agree_mask = torch.logical_and(u_psl_masks_nets[(i)%num_nets], u_psl_masks_nets[(i+1)%num_nets])  
                            disagree_weight_masked = disagree_weight * disagree_mask + (1-disagree_weight) * agree_mask
                        elif weight_disagreement == 'ablation_baseline':
                            disagree_weight = 0.5
                            disagree_mask = torch.logical_xor(u_psl_masks_nets[(i)%num_nets], u_psl_masks_nets[(i+1)%num_nets])
                            agree_mask = torch.logical_and(u_psl_masks_nets[(i)%num_nets], u_psl_masks_nets[(i+1)%num_nets])  
                            disagree_weight_masked = disagree_weight * disagree_mask + (1-disagree_weight) * agree_mask                            
                        else:
                            disagree_weight_masked = None

                        # compute loss for unlabeled data
                        unsup_loss = consistency_loss(outs_x_ulb_s_nets[i], pseudo_label, loss_type='ce', mask=u_psl_mask, disagree_weight_masked=disagree_weight_masked)    # loss_type: 'ce' or 'mse'

                        # compute total loss for unlabeled data
                        total_unsup_loss = weight_u_loss * unsup_loss
                        total_unsup_loss_nets.append(total_unsup_loss)

                    # update netgorup from loss of unlabeled data
                    netgroup.update(total_unsup_loss_nets)
                    if ema_mode:
                        netgroup.update_ema()   



                    #### -Info: for now we the last net in the group for info
                    ## Check the total and correct number of pseudo-labels
                    gt_labels_u = batch_unlabel['label'][u_psl_mask].to(device)
                    psl_total = torch.sum(u_psl_mask).item()

                    u_label_psl = pseudo_label[u_psl_mask]
                    u_label_psl_hard = torch.argmax(u_label_psl, dim=-1)
                    psl_correct = torch.sum(u_label_psl_hard == gt_labels_u).item()

                    psl_total_eval +=  psl_total
                    psl_correct_eval += psl_correct

                    # Check class-wise total and correct number of pseudo-labels 
                    cw_psl_total = torch.bincount(u_label_psl_hard, minlength=n_classes).to('cpu')
                    cw_psl_correct = torch.bincount(u_label_psl_hard[u_label_psl_hard == gt_labels_u], minlength=n_classes).to('cpu')

                    cw_psl_total_eval += cw_psl_total
                    cw_psl_correct_eval += cw_psl_correct

                    cw_psl_total_accum += cw_psl_total
                    cw_psl_correct_accum += cw_psl_correct    


    print("\nTraining complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-t0)))




    ##### Results Summary and Visualization
    # Load the saved best models and evaluate on the test set
    netgroup.load_model(output_dir_path, 'model', ema_mode=ema_mode)
    acc_test, f1_test, acc_test_cw, confmat_test = evaluation(test_loader, final_eval=True)


    ## Quantatitive Results
    # Save training statistics
    pd.set_option('precision', 4)
    df_stats= pd.DataFrame(training_stats)
    df_stats = df_stats.set_index('step')
    training_stats_path = log_dir + 'training_statistics.csv'   
    df_stats.to_csv(training_stats_path)     
    print('Save training statistics in: ', training_stats_path)

    # Save best record in both the training statisitcs and summary file
    cur_time = time.strftime("%Y%m%d-%H%M%S")
    best_data = {'record_time': cur_time,
                'best_step': best_model_step, 'test_acc':acc_test, 'test_f1': f1_test,     
                }
    print('\nBest_step: ', best_model_step, '\nbest_test_acc: ', acc_test, '\nbest_test_f1: ', f1_test)

    best_data.update(params) # record tuned hyper-params
    best_df = pd.DataFrame([best_data])         
    best_csv_path = log_dir_multiRun + 'summary.csv'
    if not os.path.exists(best_csv_path):
        best_df.to_csv(best_csv_path, mode='a', index=False, header=True)
    else:
        best_df.to_csv(best_csv_path, mode='a', index=False, header=False)
    best_df.to_csv(training_stats_path, mode='a', index=False, header=True)
    print('Save best record in: ', best_csv_path, end='')
        



    ## Visualization - Plot Training Curves
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Select data range and types to plot
    df_stats_1 = df_stats
    plot_types = ['f1', 'acc', 'psl', 'pslt']   # ['f1', 'acc', 'psl', 'pslt']

    for plot_type in plot_types:
        plt.figure()
        # Use plot styling from seaborn.
        sns.set(style='darkgrid')

        # Increase the plot size and font size.
        sns.set(font_scale=1.5)
        # plt.rcParams["figure.figsize"] = (12,6)

        # Plot the learning curve.
        for idx, key in enumerate(df_stats.keys().tolist()):
            if key.split('_')[0] == plot_type:
                plt.plot(df_stats_1[key], '--', label=key)
        # Label the plot.
        plt.xlabel("iteration")
        plt.ylabel("peformance")
        plt.legend()
        plt.savefig(log_dir+plot_type+'.png', bbox_inches='tight')

    # Visualize and save confusion matrix
    df_cm = pd.DataFrame(confmat_test, index = [i for i in range(n_classes)],
                    columns = [i for i in range(n_classes)], dtype=int)
    df_cm_norm = df_cm.div(df_cm.sum(axis=1), axis=0)
    plt.figure(figsize=(20,14))
    sns.heatmap(df_cm_norm, annot=True, annot_kws={"size": 10}, fmt='.2f', cmap='Reds')
    plt.savefig(log_dir+'confmat_norm.png', bbox_inches='tight')
    # df_cm.to_csv(log_dir+'confmat.csv', index=True, header=True)

    # return best_data to record multiRun results
    return best_data




##### multiRun ######
def multiRun(experiment_home=None, num_runs=3, unit_test_mode=False, **params):
    import statistics
    import pandas as pd
    import os
    import random
    import time

    ## Create folder structure
    # genereate a list of fixed seeds according to the number of runs
    seeds_list = list(range(num_runs)) if 'seeds_list' not in params else params['seeds_list']

    # create a folder for this multiRun
    cur_time = time.strftime("%Y%m%d-%H%M%S")
    if experiment_home is None:
        experiment_home = root + '/experiment/temp/'
    log_root = experiment_home + '/log/'
    global log_dir_multiRun
    log_dir_multiRun = log_root + cur_time + '/'

    if not os.path.exists(experiment_home):
        os.makedirs(experiment_home)
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    if not os.path.exists(log_dir_multiRun):
        os.makedirs(log_dir_multiRun)

    # create folders for each run
    for i in range(num_runs):
        log_dir = log_dir_multiRun + str(seeds_list[i]) + '/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    # create output folder for this experiment
    output_dir_experiment = experiment_home + '/output/'
    if not os.path.exists(output_dir_experiment):
        os.makedirs(output_dir_experiment)


    ## Obtain averaged results over multiple runs
    results = []
    test_accs = []
    test_f1s = []
    for i in range(num_runs):
        if not unit_test_mode:
            log_dir = log_dir_multiRun + str(seeds_list[i]) + '/'
            result = oneRun(log_dir, output_dir_experiment, **params, seed=seeds_list[i])
        else:
            result = {'test_acc': random.random(), 'test_f1': random.random()}
        results.append(result)
        
        test_accs.append(result['test_acc'])
        test_f1s.append(result['test_f1'])

    st_dev_acc = round(statistics.pstdev(test_accs), 4)
    mean_acc = round(statistics.mean(test_accs), 4)
    st_dev_f1 = round(statistics.pstdev(test_f1s), 4)
    mean_f1 = round(statistics.mean(test_f1s), 4)

    if len(seeds_list) != 1:
        final = {'record_time': cur_time,
                'Mean_std_acc': '%.2f ± %.2f' % (100*mean_acc, 100*st_dev_acc),
                'Mean_std_f1': '%.2f ± %.2f' % (100*mean_f1, 100*st_dev_f1)}
    else:
        final = {'record_time': cur_time,
                'Mean_std_acc': '%.2f' % (100*mean_acc),
                'Mean_std_f1': '%.2f' % (100*mean_f1)}

    final.update(params)
    final.update({'seeds_list': seeds_list})
    df = pd.DataFrame([final])
    csv_path = log_root + 'summary_avgrun.csv'
    df.to_csv(csv_path, mode='a', index=False, header=True)
    print('\nSave best record in: ', csv_path)








