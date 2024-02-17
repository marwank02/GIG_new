import mlflow
from mlflow.tracking import MlflowClient
from ogb.graphproppred.dataset_pyg import PygGraphPropPredDataset
from sklearn.model_selection import KFold, StratifiedKFold
import json
from utils_provider import *
from config import *
import argparse
from itertools import product
import warnings

#"E:\fields_of_study\artificial_intelligence\jobs\harvard\GiG_original\checkpoints\Final\PROTEINS_full_b_s_50_pooling_mean__GraphInGraph_LGLr_split_False_\run_0\1_folder\best_model.pth"

warnings.filterwarnings("ignore")

seed_everything(seed=1234)

def main(arg):
    config = MLConfig(arg)
    # print(config.arg)
    dataset = TUDataset(os.path.join('data', config.dataset), name=config.dataset, use_node_attr=True)

    config.output_dim = dataset.num_classes
    config.input_dim = dataset.num_features
    config.num_node_features = dataset.num_features
    writer_path_ = config.arg.writer_path
    saving_path_ = config.arg.saving_path
    print("???????????????????????????????")
    print(saving_path_)
    acc_list = []
    roc_list = []
    best_accuracy = 0.0  
    best_model_path = ''  

    for run_num in range(0,10):
        writer_path = writer_path_ + '/run_' + str(run_num) + '/'
        saving_path = saving_path_ + '/run_' + str(run_num) + '/'

        roc_auc_score_test = []
        acc_test = []
        if not config.random_split:

            kf = KFold(n_splits=10, shuffle=True,random_state=42)


            num_training = int(len(dataset) * 0.9)

            num_test = len(dataset) - (num_training)
            training_set, test_set = random_split(dataset, [num_training, num_test],
                                                    generator=torch.Generator().manual_seed(42))
            loader_test = DataLoader(test_set, batch_size=config.batch_size_test, shuffle=False)

            training_set = list(training_set)

            for i, (train_index, val_index) in enumerate(kf.split(training_set)):
                # print("TRAIN:", train_index, "TEST:", val_index)
                train_set = list(map(training_set.__getitem__, train_index))
                validation_set = list(map(training_set.__getitem__, val_index))

                loader_train = DataLoader(train_set, batch_size=config.batch_size_train, shuffle=False)
                loader_val = DataLoader(validation_set, batch_size=config.batch_size_val, shuffle=False)

                config.arg.writer_path = writer_path + '/'+str(i)+'_folder/'
                config.arg.saving_path = saving_path + '/'+str(i)+'_folder/'

                if not os.path.exists(config.arg.saving_path):
                    os.makedirs(config.arg.saving_path)
                if not os.path.exists(config.arg.writer_path):
                    os.makedirs(config.arg.writer_path)

                # config.kl_sum_par = 3
                model, fixed_test_graphs = train_model(config, loader_train, loader_val, loader_test)
                # load model
                # state_dict  = torch.load("models/best_model2.pth")
                # model.load_state_dict(state_dict)



                acc, roc = eval_model(config, model, loader_test, fixed_test_graphs)
                acc_test.append(acc)
                roc_auc_score_test.append(roc)
                
                if acc > best_accuracy:
                    print("saving model")
                    best_accuracy = acc
                    best_model_path = os.path.join("models", 'best_model2.pth')
                    torch.save(model.state_dict(), best_model_path)
            # print("acc_test", acc_test)
            # print("Final TEST acc", np.mean(acc_test))
            # print("Final TEST ROC", roc_auc_score_test)
            # print("Final TEST acc", np.mean(roc_auc_score_test))

        # else:
        #     for i in range(10):
        #         num_training_val = int(len(dataset) * 0.9)
        #         num_test = len(dataset) - (num_training_val)

        #         num_training = int(num_training_val * 0.9)
        #         num_val = num_training_val - (num_training)

        #         training_set, validation_set,test_set = random_split(dataset, [num_training, num_val,num_test])

        #         loader_train = DataLoader(training_set, batch_size=config.batch_size_train, shuffle=False)
        #         loader_val = DataLoader(validation_set, batch_size=config.batch_size_val, shuffle=False)
        #         loader_test = DataLoader(test_set, batch_size=config.batch_size_test, shuffle=False)

        #         config.arg.writer_path = writer_path + '/'+str(i)+'_folder/'
        #         config.arg.saving_path = saving_path + '/'+str(i)+'_folder/'

        #         if not os.path.exists(config.arg.saving_path):
        #             os.makedirs(config.arg.saving_path)
        #         if not os.path.exists(config.arg.writer_path):
        #             os.makedirs(config.arg.writer_path)

        #         # config.kl_sum_par = 3
        #         model, fixed_test_graphs = train_model(config, loader_train, loader_val, loader_test)
        #         acc, roc = eval_model(config, model, loader_test, fixed_test_graphs)
        #         acc_test.append(acc)
        #         roc_auc_score_test.append(roc)
        acc_list.append(np.mean(acc_test))
        roc_list.append(np.mean(roc_auc_score_test))
        with open(config.arg.saving_path + "results.txt", "w") as f:
            f.write("acc_test" + str(np.mean(acc_test)) + "roc_auc_score_test" + str(np.mean(roc_auc_score_test)))
            f.write("acc_test")
            f.write(json.dumps(acc_test))
            f.write("roc_auc_score_test")
            f.write(json.dumps(roc_auc_score_test))
        # print("Final TEST acc", np.mean(acc_test))
        # print("Final TEST ROC", np.mean(roc_auc_score_test))
            
    with open(saving_path_ + "final_results.txt", "w") as f:
        f.write(" acc_test " + str(np.mean(acc_list)) + 'std acc' + str(np.std(acc_list))
                + " roc_auc_score_test " + str(np.mean(roc_list)) + 'std roc' + str(np.std(roc_list)))
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=' '
                                                 '')
    parser.add_argument('--dataset', default='PROTEINS_full', type=str,
                        help='Name of dataset {"PROTEINS_full", "Tox21",  "HCP"}.')
    parser.add_argument('--node_level_module', default='GraphConv', type=str,
                        help='Name of node level module {"GraphConv", "GIN"}.')
    parser.add_argument('--DGCNN', default=False, type=bool,
                        help='if True uses DGCNN, if False uses FixedGraph, LGL, LGLKL: {True, False}.')
    parser.add_argument('--regularization', default="", type=str,
                        help='If '' and DGCNN=False and model='', then LGL. '
                             'If "KL_degree_dist" and DGCNN=False and model='', then LGLKL.: '
                             '{"KL_degree_dist"}.')
    parser.add_argument('--model', default="", type=str,
                        help='"FixedGraphInGraph" - uses Fixed graphs, precomputed'
                             ' "GCN" - just normal GCN model, '
                             ' "" - uses GraphInGraph model: '
                             '{"GCN", "FixedGraphInGraph"}.')
    parser.add_argument('--fix_population_graph', default="knn", type=str,
                        help='Version of fixed population graph:'
                             '{"knn", "random"}.')
    additon = "Final/"
    


    
    parser_args = parser.parse_args()
    # print(parser_args.dataset)
    
    # parser_args.dataset = "PROTEINS_full"
    # parser_args.node_level_module = 'GraphConv'
    # parser_args.DGCNN = False
    # parser_args.model = ""
    # parser_args.regularization = ""
    # parser_args.fix_population_graph = "knn"
    
    parser_args.task = 'gender'
    

    node_conv_size = [[30, 28]]
    graph_lin_size = [[24, 20]]
    graph_conv_size = [[20]]
    classif_fc = [[20, 16, 12]] #
    dgcnn_conv_size = [[128]]

    distr = ['normal', 'power_law']


    exp_list = product(node_conv_size, graph_lin_size, graph_conv_size, classif_fc, dgcnn_conv_size)#, distr)
    exp_list = list(exp_list)
    for i in exp_list:
        client = MlflowClient()
        existing_experiment = client.get_experiment_by_name(str(parser_args.dataset))
        if existing_experiment:
            experiment_id = existing_experiment.experiment_id
        else:
            experiment_id = client.create_experiment(str(parser_args.dataset))

        with mlflow.start_run(experiment_id=experiment_id, run_name=additon):
            # print(i)
            parsed_args = parser.parse_args()
            parsed_args.node_conv_size = i[0]
            parsed_args.graph_lin_size = i[1]
            parsed_args.graph_conv_size = i[2]
            parsed_args.classif_fc = i[3]
            parsed_args.dgcnn_conv_size = i[4]
            config = MLConfig(parsed_args)
            # print('config', config)

            parsed_args.writer_path = config.writer_path + additon + str(parsed_args.dataset) + '_b_s_'+ str(config.batch_size_train) + '_pooling_' + str(config.pooling) + '_'

            parsed_args.saving_path = "checkpoints/"\
                                      + additon + str(parsed_args.dataset) + '_b_s_'+ str(config.batch_size_train) + '_pooling_' + str(config.pooling) + '_'

            path_params_addition = ''
            config.task = parser_args.task
            
            path_params_addition += '_GraphInGraph_'
            path_params_addition += 'LGL'

            if config.regularization == 'KL_degree_dist':
                path_params_addition += '_kl_sumpar' + str(config.kl_sum_par) + '_'+ config.target_distribution + '_'
                if config.learnable_distr:
                    path_params_addition += 'learnable_params_'
            if config.dataset == 'PROTEINS_full' and config.model!='FixedGraphInGraph':
                path_params_addition += 'r_split_' + str(config.random_split) + '_'


            parsed_args.writer_path += path_params_addition
            parsed_args.saving_path += path_params_addition

            if not os.path.exists(parsed_args.saving_path):
                os.makedirs(parsed_args.saving_path)
            if not os.path.exists(parsed_args.writer_path):
                os.makedirs(parsed_args.writer_path)

            # print("parsed_args", parsed_args)
            main(parsed_args)
