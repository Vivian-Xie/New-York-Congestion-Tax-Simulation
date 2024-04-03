from USTGCN import *

def draw_curve(predicts:list,new_predicts:list,site:int,mean:bool,draw,path
               ):
    import matplotlib.pyplot as plt
    # predicts -> 277,228,3 -> list
    # new_predicts -> 277,228,3 -> list
    # site -> 选择绘制228个站点中的哪一个
    # mean -> 是否使用228个站点的平均值绘制

    predicts = np.array(predicts)
    new_predicts = np.array(new_predicts)
    if mean:
        mean_arr = np.mean(predicts, axis=1)
        result = mean_arr[:,2].tolist()
        mean_arr2 = np.mean(new_predicts, axis=1)
        result2 = mean_arr2[:, 2].tolist()
    else:
        result = predicts[:,site-1,2].tolist()
        result2 = new_predicts[:, site - 1, 2].tolist()

    if draw:
        plt.plot(result,label='original speed')
        plt.plot(result2,label='optimized speed')
        # 添加图例
        plt.legend()
        plt.savefig(path)
    else:
        plt.plot(result, label='original speed')
        plt.plot(result2, label='optimized speed')
        # 添加图例
        plt.legend()
        plt.show()




if __name__ == '__main__':

    print("test")
    parser = argparse.ArgumentParser(description='pytorch version of USTGCN')
    parser.add_argument('-f')

    parser.add_argument('--dataset', type=str, default='PeMSD7')
    parser.add_argument('--GNN_layers', type=int, default=3)
    parser.add_argument('--num_timestamps', type=int, default=12)
    parser.add_argument('--pred_len', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', default=False, help='use CUDA')
    parser.add_argument('--trained_model', default='USTGCN/USTGCN-master/PeMSD7/bestTmodel_15minutes.pth')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--input_size', type=int, default=8)

    parser.add_argument('--the_day', type=int, default=23) # 修改 选择第几天的数据进行预测
    parser.add_argument('--site', type=int, default=12) # 修改 选择第几个站点的预测结果进行绘制
    parser.add_argument('--mean', type=bool, default=True) # 修改 是否使用228个站点的平均结果来绘制
    parser.add_argument('--draw', type=bool, default=False)  # 修改 是否保存绘制图像,在不确定时可先设为false，会弹出预览图
    parser.add_argument('--path', type=str, default='test.jpg')  # 修改 图像保存路径

    args = parser.parse_args()

    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    print('DEVICE:', device)

    """# Main Function"""

    print('Traffic Forecasting GNN with Historical and Current Model')

    # set user given seed to every random generator
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    PATH = os.getcwd() + "/"
    config_file = PATH + "experiments.conf"

    config = pyhocon.ConfigFactory.parse_file(config_file)
    ds = args.dataset
    pred_len = args.pred_len
    data_loader = DataLoader(config, ds, pred_len,the_day=args.the_day) #修改
    test_data , adj = data_loader.load_predict_data()#修改 （288,12，228，8）

    num_timestamps = args.num_timestamps
    GNN_layers = args.GNN_layers
    input_size = args.input_size
    out_size = args.input_size
    epochs = args.epochs

    save_flag = args.save_model
    t_debug = False
    b_debug = False
    hModel = TrafficModel(None, None, test_data, None, adj, config, ds, input_size,
                          out_size, GNN_layers, epochs, device, num_timestamps, pred_len, save_flag,
                          PATH, t_debug, b_debug)

    print("Running Trained Model...")

    predicts_one_day , predicts_new_day = hModel.run_predict()  # 277,288,3 -> list

    draw_curve(predicts_one_day,predicts_new_day,args.site,args.mean,args.draw,args.path)
    print("Success!")
