import re

import matplotlib.pyplot as plt
import numpy as np
import xgboost
from xgboost import XGBRegressor, plot_importance
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
from bayes_opt import BayesianOptimization
import argparse
import time
import pandas as pd
import os

####
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-i","--input", help="Required. the path for normalized value of all gene expression data")
parser.add_argument("--config",default=None, help="Optional. the path for config to optimize hyper-parameters\n"
                                                  "This option is only used for bayes hyper-parameters optimization, so do not use it for training!!!")

parser.add_argument("-t","--tf", help="Required. the path for normalized value of TF expression data or a TF list")
parser.add_argument("-g","--genes", default=None,help="Optional. a gene list with comma separated or a list file")

parser.add_argument("-p","--threads", type=int, default=1,help="default. the threads for model training")
parser.add_argument("--log2",action="store_true", default=False,
					help="Optional. whether to use the log2-scaled convertion\n"
						 "default: False")
parser.add_argument("--test_size",type=float,default=0.2, help="Optional. the ratio for test dataset to train dataset in tain_test_split")
parser.add_argument("--save_model",action="store_true", default=False,
					help="Optional. whether to save the model and parameters (Recommend: do not save model for TF prediction!!!)\n"
						 "default: False")
parser.add_argument("--model_dir",default="./model", help="Optional. the path for model parameters\n"
                                                      "default=./model")
parser.add_argument("--model_name_prefix",default="model", help="Optional. the prefix for model file name\n"
                                                                "default=model")
parser.add_argument("-o","--output", help="Required. the output file path for TF weight in all samples/datasets")
parser.add_argument("--n_estimators",type=int, default=1000, help="Optional. the number of n_estimators")
# subsample,colsample_bytree,gamma,
parser.add_argument("--subsample",type=float, default=0.75, help="Optional. Subsample ratio of the training instance\n"
                                                                 "default=0.75")
parser.add_argument("--colsample_bytree",type=float, default=1.0, help="Optional. Subsample ratio of columns when constructing each tree.\n"
                                                                       "default=1")
parser.add_argument("--gamma",type=float, default=0, help="Optional. Minimum loss reduction required to make a further partition on a leaf node of the tree.\n"
                                                          "default=0")

parser.add_argument("--learning_rate",type=float, default=0.001, help="Optional. the learning_rate")
parser.add_argument("--max_depth",type=int, default=3, help="Optional. the max_depth")
parser.add_argument("--early_stopping_rounds",type=int, default=50, help="Optional. the early_stopping_rounds")
parser.add_argument("--reg_alpha",type=float, default=0, help="Optional. the L1 regularization alpha")
parser.add_argument("--reg_lambda",type=float, default=0.1, help="Optional. the L2 regularization lambda")
parser.add_argument("--random_state",type=int, default=0, help="Optional. the random state seed")

parser.add_argument("--device",default="cpu", help="Optional. the device for training model\n"
                                                   "Device ordinal, available options are cpu, cuda, and gpu.\n"
                                                   "default=cpu")
parser.add_argument("--verbose",action="store_true", default=False,
					help="Optional. whether to open the verbose mode\n"
						 "default: False")

args = parser.parse_args()

# Define a custom evaluation metric used for early stopping.
def eval_error_metric(predt, dtrain: xgboost.DMatrix):
    r2 = r2_score(predt,dtrain)
    return 'r2_square', r2

def read_gene_expression(gene_exp_input,header=0):
    input_suffix = os.path.basename(gene_exp_input).split(".")[-1]
    if input_suffix.lower() == "csv":
        gene_expr_data = pd.read_table(gene_exp_input,sep=",",header=header)
    else:
        gene_expr_data = pd.read_table(gene_exp_input,sep="\t",header=header)
    return gene_expr_data

def isNumber(f):
    return all([p.isdigit() for p in [f[:f.find('.')], f[f.find('.')+1:]] if len(p)])

# 自定义参数类型转换函数
def auto_type(s,return_int=True):
    if isNumber(s):
        if return_int:  # 判断是否为数字
            return int(s)
        else:
            return float(s)
    else:
        return s  # 否则返回字符串类型

def read_config_file(filename="hyper_parameters-config.txt"):
    config_dict = {}

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                key, val = line.split(':')
                key = key.strip().replace(" ","")
                val = val.strip().replace(" ","")

                if not val:
                    config_dict[key] = []
                else:
                    config_dict[key] = [auto_type(i,return_int=False) for i in val.split(",")]

    return config_dict

def remove_null_key(hyperparameters):
    # new_dict = hyperparameters
    keys = []
    for (key, val) in hyperparameters.items():
        if len(val) == 0:
            keys.append(key)
    for key in keys:
        del hyperparameters[key]
    return hyperparameters

def model_train_each(id,config_file,gene,real_tf_list,n_estimators,X_train,Y_train,X_test,Y_test,
                     subsample,colsample_bytree,gamma,learning_rate,max_depth,early_stopping_rounds,
                    reg_alpha,reg_lambda,random_state,
                    save_model,model_path,model_para_path,temp_dir,
                     threads,device):
    # X_train = xgboost.DMatrix(X_train)
    # Y_train = xgboost.DMatrix(Y_train)
    # X_test = xgboost.DMatrix(X_test)
    # Y_test = xgboost.DMatrix(Y_test)

    ## initialize XGBoost model
    fprint("Starting to train XGBoost model...")
    early_stop = xgboost.callback.EarlyStopping(rounds=early_stopping_rounds)

    # 定义XGBoost回归模型参数的搜索空间
    def objective(learning_rate,max_depth,gamma,reg_alpha,reg_lambda):
        params_opt = {
            'max_depth': int(max_depth),
            'learning_rate': learning_rate,
            'n_estimators': int(n_estimators),
            'gamma': gamma,
            'reg_alpha':reg_alpha,
            'reg_lambda':reg_lambda,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree
        }

        model = xgboost.XGBRegressor(objective='reg:squarederror', **params_opt)
        score = cross_val_score(model, X_train, Y_train, scoring='neg_mean_squared_error', cv=5).mean()  # 5折交叉验证
        return score  # 贝叶斯优化最小化目标函数，因此返回负MSE得分
    # 进行贝叶斯优化
    # result = gp_minimize(objective, space, n_calls=50)  # 调整n_calls参数控制搜索次数

    ## read config file
    if config_file:

        param_dict = remove_null_key(read_config_file(config_file))
        rf_bo = BayesianOptimization(
            f=objective,
            pbounds=param_dict)
        rf_bo.maximize()
        # 输出最优参数
        # best_params = {k: v for k, v in result.x.items()}
        print("Best parameters found: ", rf_bo.max)

    # xgb = XGBRegressor(n_estimators=n_estimators,
    #                    learning_rate=learning_rate,
    #                    subsample=subsample,
    #                    colsample_bytree=colsample_bytree,
    #                    max_depth=max_depth,
    #                    gamma=gamma,
    #                    reg_alpha=reg_alpha,
    #                    reg_lambda=reg_lambda,
    #                    eval_metrics=["rmse"],
    #                    n_jobs=threads,
    #                    random_state=random_state,
    #                    device=device)
    else:
        xgb = XGBRegressor(n_estimators=n_estimators,
                          learning_rate=learning_rate,
                          subsample=subsample,
                          colsample_bytree=colsample_bytree,
                          max_depth=max_depth,
                          gamma=gamma,
                          reg_alpha=reg_alpha,
                          reg_lambda=reg_lambda,
                          eval_metrics=["rmse"],
                          n_jobs=threads,
                          random_state=random_state,
                          device=device)
        xgb.fit(X_train,Y_train,eval_set=[(X_test, Y_test)],callbacks=[early_stop],verbose=verbose)
        fprint(xgb.score(X_train, Y_train))
        fprint(xgb.score(X_test,Y_test))
        fprint("Done for training XGBoost model!")
        if save_model:
            fprint("Starting to save XGBoost model...")
            model_save_type(xgb,model_path,model_para_path)
            fprint("Done for saving XGBoost model!")
        fprint("Starting to export the importance of features...")

        output_each = os.path.join(temp_dir,"importance_{}.txt".format(gene))
        feature_importance_selected(id,xgb,gene,real_tf_list,output_each)
        fprint("Done for {}!".format(gene))
        pass

def model_predict_each():
    pass

# def split_index(item):
#     return item.split("f")[-1]
#     pass
def feature_importance_selected(id,clf_model,gene,real_tf_list,output,plot=False):
    """模型特征重要性提取与保存"""
    # 模型特征重要性打印和保存
    feature_importance = clf_model.get_booster().get_fscore()
    feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    feature_ipt = pd.DataFrame(feature_importance, columns=['TF_index', 'importance'])
    feature_ipt["TF"] = feature_ipt.loc[:,"TF_index"].apply(func=lambda x:real_tf_list[int(x.split("f")[-1])])
    feature_ipt["gene"] = gene
    feature_ipt.to_csv(output, index=False)
    # if id == 0:
    #     feature_ipt.to_csv(output, index=False)
    # else:
    #     feature_ipt.to_csv(output, header=False,index=False)
    # print('importance:', feature_importance)

    # 模型特征重要性绘图
    output_dir = os.path.dirname(output)
    output_plot_path = os.path.join(output_dir,'./importance of TF.pdf')
    if plot:
        plot_importance(clf_model)
        # plt.show()
        plt.savefig(output_plot_path)

def model_save_type(clf_model,model_path,model_parameters_path):
    # 模型训练完成后做持久化，模型保存为model模式，便于调用预测
    clf_model.save_model(model_path)

    # 模型保存为文本格式，便于分析、优化和提供可解释性
    clf = clf_model.get_booster()
    clf.dump_model(model_parameters_path)

def fprint(msg,msg_type="INFO"):
    print("{}-[{}]: {}".format(time.asctime(), msg_type, msg))
    pass


def read_tf_list(tf,header=None):
    tf_data = pd.read_table(tf, header=header, sep="\t")
    tf_number = tf_data.shape[0]
    sample_number = tf_data.shape[1]
    if sample_number == 1:
        fprint("The sample number in TF file is 1, so it will be recognized as a tf list!","WARNING")
        return tf_data.iloc[:, 0].tolist()
    else:

        return tf_data

def check_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def main(input_file, config_file, tf, genes, log2, test_size,save_model,model_dir,model_name_prefix,
         output,n_estimators,subsample,colsample_bytree,gamma,
         learning_rate,max_depth,early_stopping_rounds,reg_alpha,reg_lambda,random_state,
         threads,device,verbose=False):
    ## read input gene expression data
    fprint("Reading input gene expression data...")
    gene_expr_data = read_gene_expression(input_file)
    if log2:
        gene_expr_data.iloc[:,1:] = np.log2(gene_expr_data.iloc[:,1:] + 1)

    keep_gene_list = None
    if genes:
        if os.path.exists(genes):
            keep_gene_list = read_tf_list(genes)
        else:
            keep_gene_list = genes.split(",")
    sample_list = gene_expr_data.columns[1:]
    gene_list = gene_expr_data.iloc[:,0].to_list()
    fprint("Sample size: {}\tGene number: {}".format(len(sample_list),len(gene_list)))
    fprint("Done for reading gene expression data!")

    ## read TF list or expression data
    fprint("Reading TF information...")

    tf_data = read_tf_list(tf)
    if type(tf_data) == pd.DataFrame:
        tf_list = tf_data.loc[:,0].to_list()
        tf_sample_list = tf_data.columns[1:]
        tf_expr_data = tf_data
        fprint("The size of TF expression data is {}".format(tf_expr_data.shape))
        fprint("TF size: {}\t Sample size in TF data: {}".format(len(tf_list), len(tf_sample_list)))
        fprint("The size of TF not found is {}".format(tf_expr_data.shape[0]))
        pass
    else:
        tf_list = tf_data
        tf_expr_data = gene_expr_data[gene_expr_data.iloc[:, 0].isin(tf_list)]
        tf_sample_list = sample_list
        fprint("The size of TF expression data found is {}".format(tf_expr_data.shape))
        fprint("TF size: {}\t Sample size will be the same with gene expression data due to the list only provided!".format(len(tf_list)))
        fprint("The size of TF not found is {}".format(len(tf_list) - tf_expr_data.shape[0]))
        pass

    fprint("Done for reading TF information!")

    ## Intersection for samples from TF data and expression data
    inter_sample_list = sorted(list(set(sample_list) & set(tf_sample_list)))
    fprint("Actual sample size is {}".format(len(inter_sample_list)))
    ## Model Training
    # for i in range(len(inter_sample_list)):
    # pass
    # sample = inter_sample_list[i]
    # fprint("Training under Sample {}".format(sample))
    fprint("Generating the training and test datasets...")
    dataset_sample_X = tf_expr_data.loc[:,["gene"] + inter_sample_list]
    dataset_sample_Y = gene_expr_data.loc[:, ["gene"] + inter_sample_list]
    # print(keep_gene_list)
    if keep_gene_list:
        dataset_sample_Y = dataset_sample_Y.loc[dataset_sample_Y.loc[:,"gene"].isin(keep_gene_list),:]
        fprint("Genes used to analysis: {}".format(dataset_sample_Y.shape[0]))
    # print(dataset_sample_X.shape)
    # print(dataset_sample_Y.shape)
    final_keep_gene_list = dataset_sample_Y.loc[:, "gene"].to_list()
    # print(dataset_sample_X.columns)
    # print(dataset_sample_X.iloc[0, 0])
    # print(dataset_sample_X.iloc[0, 1])
    # fprint("Writing the ")
    real_tf_list = dataset_sample_X.loc[:,"gene"].to_list()
    fprint("Done for generating the training and test datasets!")
    check_dir(model_dir)

    temp_dir = os.path.join(os.path.dirname(output),"temp")
    check_dir(temp_dir)
    for i in range(len(final_keep_gene_list)):
        pass
        gene = final_keep_gene_list[i]
        model_path = os.path.join(model_dir, model_name_prefix + "_{}.model".format(gene))
        model_para_path = os.path.join(model_dir, model_name_prefix + "_{}.model".format(gene))
        fprint("Training on gene: {}".format(gene))
        X_each_gene = np.array(dataset_sample_X.loc[:, inter_sample_list]).reshape(len(inter_sample_list),-1)
        y_each_gene = np.array(dataset_sample_Y.loc[dataset_sample_Y.loc[:,"gene"]==gene, inter_sample_list]).reshape(len(inter_sample_list),-1)
        X_train, X_test, Y_train, Y_test = train_test_split(X_each_gene,y_each_gene,test_size=test_size,random_state=random_state)
        # print(X_train[0,0])
        model_train_each(i,config_file,gene,real_tf_list,n_estimators,X_train,Y_train,X_test,Y_test,
                         subsample, colsample_bytree,gamma, learning_rate, max_depth,early_stopping_rounds,
                         reg_alpha, reg_lambda,random_state,
                         save_model,model_path,model_para_path, temp_dir, threads,device)
        # break
    ## combine the importance for all genes
    combine_all_temp_output(output,temp_dir,"importance_.*.txt",verbose)
    pass

def output_func(output_file, content, line_break=True,init=True):
    if init:
        with open(output_file,'w') as handle:
            pass
    else:

        if line_break:
            break_label = '\n'
        else:
            break_label = ''

        with open(output_file, 'a') as handle:
            handle.write("{}{}".format(content,break_label))
            pass

# combine all temp gff file by each epoch
def combine_all_temp_output(output_file,temp_output_dir,temp_output_pattern,verbose):
    '''
    :param temp_gff_dir:
    :param temp_gff_pattern: xxx.\d+.gff
    :return:
    '''
    all_files = os.listdir(temp_output_dir)
    count = 0
    all_valid_files = []
    for file in all_files:
        if len(re.findall(temp_output_pattern,file)) > 0:
            count += 1
            all_valid_files.append(file)
    all_valid_files.sort()
    fprint("LOG","Finding {} temp output files".format(count))
    data_list = []
    for index, file in enumerate(all_valid_files):
        temp_data = pd.read_table(os.path.join(temp_output_dir, file),sep="\t")
        data_list.append(temp_data)

        pass

    final_data = pd.concat(data_list,axis=0)
    final_data.to_csv(output_file,index=False,header=True,sep="\t")
    # output_func(output_file,'',init=True,line_break=False)
    # for index,file in enumerate(all_valid_files):
    #     file_path = os.path.join(temp_output_dir,file)
    #     f = open(os.path.join(temp_output_dir,file)).read()
    #     if verbose:
    #         fprint("LOG","Combing the {}-th file".format(index))
    #     output_func(output_file,f,init=False,line_break=False)
    #     os.remove(file_path)

if __name__ == '__main__':
    input_file, tf, test_size,model_dir,model_name_prefix = args.input, args.tf, args.test_size, args.model_dir,args.model_name_prefix
    config_file = args.config
    log2 = args.log2
    genes = args.genes
    output,n_estimators,learning_rate,max_depth = args.output,args.n_estimators,args.learning_rate,args.max_depth
    subsample, colsample_bytree, gamma, = args.subsample,args.colsample_bytree,args.gamma
    early_stopping_rounds = args.early_stopping_rounds
    reg_alpha, reg_lambda = args.reg_alpha,args.reg_lambda
    random_state = args.random_state
    save_model = args.save_model
    threads = args.threads
    device = args.device
    verbose = args.verbose
    main(input_file,config_file, tf,genes, log2, test_size,save_model,model_dir,model_name_prefix,
         output,n_estimators,subsample,colsample_bytree,gamma,learning_rate,max_depth,early_stopping_rounds,reg_alpha,reg_lambda,random_state,
         threads,device,verbose)
    ####
    # data = load_diabetes()
    # # print(data['target'])
    # # print(data.keys())  # 查看键(属性)     ['data','target','feature_names','DESCR', 'filename']
    # # print(data.data.shape, data.target.shape)  # 查看数据的形状 (506, 13) (506,)
    # # print(data.target)
    # # print(data.feature_names)  # 查看有哪些特征 这里共13种
    # # print(data.DESCR)  # described 描述这个数据集的信息
    # # print(data.filename)  # 文件路径
    # X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], test_size=.2)
    # # create model instance
    # bst = XGBRegressor(n_estimators=1000,
    #     learning_rate=0.08,
    #     subsample=0.75,
    #     colsample_bytree=1,
    #     max_depth=7,
    #     gamma=0,device="cuda")
    # # fit model
    # bst.fit(X_train, y_train)
    # # make predictions
    # preds = bst.predict(X_test)
    # print(preds)