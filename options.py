import argparse
import os

"""
调用prepare_train_args，就会创建一个包含所有公共参数和训练参数的parser，然后创建一个模型目录，并调用save_args函数保存所有参数，返回对应的args。
保存参数这一步十分重要，能够避免模型训练完成之后，脚本或命令找不到，忘记自己训练的模型配置这种尴尬局面。
测试时也类似，调用prepare_test_args，创建parser，创建目录，保存参数，并返回对应的args。
"""

def parse_common_args(parser):
    parser.add_argument('--model_type', type=str, default='base_model', help='used in model_entry.py')

    # 数据集的名字，配合data目录和data_entry.py使用；
    parser.add_argument('--data_type', type=str, default='base_dataset', help='used in data_entry.py')
    
    # 训练时：实验的名字，可以备注自己改了那些重要组件，具体的参数，会用于创建保存模型的目录；
    # 测试时：测试的名字，可以备注测试时做了哪些配置，会用于创建保存测试结果的目录；
    parser.add_argument('--save_prefix', type=str, default='pref', help='some comment for model or test result dir')

    parser.add_argument('--load_model_path', type=str, default='checkpoints/base_model_pref/0.pth',
                        help='model path for pretrain or test')
        
    # load_match_dict函数（utils/torch_utils.py），允许加载的模型和当前模型的参数不完全匹配，可多可少，如果打开这个选项，
    # 就会调用此函数，这样我们就可以修改模型的某个组件，然后用之前的模型来做预训练啦！
    # 如果关闭，就会用torch原本的加载逻辑，要求比较严格的参数匹配
    parser.add_argument('--load_not_strict', action='store_true', help='allow to load only common state dicts')
    
    # 训练时可以传入验证集list，测试时可以传入测试集list；
    parser.add_argument('--val_list', type=str, default='/data/dataset1/list/base/val.txt',
                        help='val list in train, test list path in test')
    # 可以配置训练或测试时使用的显卡编号，在多卡训练时需要用到，测试时也可以指定显卡编号，绕开其他正在用的显卡，当然你也可以在命令行里export
    parser.add_argument('--gpus', nargs='+', type=int)
    parser.add_argument('--seed', type=int, default=1234)
    return parser


def parse_train_args(parser):
    parser = parse_common_args(parser)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum for sgd, alpha parameter for adam')
    parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                        help='beta parameters for adam')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay')
    
    # 模型的存储目录，留空，不用传入，会在get_train_model_dir函数中确定这个字段的值，创建对应的目录，填充到args中，方便其他模块获得模型路径
    parser.add_argument('--model_dir', type=str, default='', help='leave blank, auto generated')
    
    # 训练集list路径
    parser.add_argument('--train_list', type=str, default='/data/dataset1/list/base/train.txt')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    return parser


def parse_test_args(parser):
    parser = parse_common_args(parser)
    parser.add_argument('--save_viz', action='store_true', help='save viz result in eval or not')
    
    # 可视化结果和测试结果的存储目录，留空，不用传入，会在get_test_result_dir中自动生成，自动创建目录，
    # 这个目录通常位于模型路径下，形如checkpoints/model_name/checkpoint_num/val_info_save_prefix
    parser.add_argument('--result_dir', type=str, default='', help='leave blank, auto generated')
    return parser


def get_train_args():
    parser = argparse.ArgumentParser()
    parser = parse_train_args(parser)
    args = parser.parse_args()
    return args


def get_test_args():
    parser = argparse.ArgumentParser()
    parser = parse_test_args(parser)
    args = parser.parse_args()
    return args


def get_train_model_dir(args):
    model_dir = os.path.join('checkpoints', args.model_type + '_' + args.save_prefix)
    if not os.path.exists(model_dir):
        os.system('mkdir -p ' + model_dir)
    args.model_dir = model_dir


def get_test_result_dir(args):
    ext = os.path.basename(args.load_model_path).split('.')[-1]
    model_dir = args.load_model_path.replace(ext, '')
    val_info = os.path.basename(os.path.dirname(args.val_list)) + '_' + os.path.basename(args.val_list.replace('.txt', ''))
    result_dir = os.path.join(model_dir, val_info + '_' + args.save_prefix)
    if not os.path.exists(result_dir):
        os.system('mkdir -p ' + result_dir)
    args.result_dir = result_dir


def save_args(args, save_dir):
    args_path = os.path.join(save_dir, 'args.txt')
    with open(args_path, 'w') as fd:
        fd.write(str(args).replace(', ', ',\n'))


def prepare_train_args():
    args = get_train_args()
    get_train_model_dir(args)
    save_args(args, args.model_dir)
    return args


def prepare_test_args():
    args = get_test_args()
    get_test_result_dir(args)
    save_args(args, args.result_dir)
    return args


if __name__ == '__main__':
    train_args = get_train_args()
    test_args = get_test_args()
