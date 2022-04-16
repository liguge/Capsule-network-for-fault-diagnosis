import sys
sys.setrecursionlimit(15000)

import torch
import torch.nn.functional as F
from torch import nn
# from wmodels import Laplace_fast as fast

BATCH_SIZE = 100
NUM_CLASSES = 10
NUM_EPOCHS = 500
NUM_ROUTING_ITERATIONS = 3



# class h_sigmoid(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_sigmoid, self).__init__()
#         self.relu = nn.ReLU6(inplace=inplace)
#
#     def forward(self, x):
#         return self.relu(x + 3) / 6
#
#
# class h_swish(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_swish, self).__init__()
#         self.sigmoid = h_sigmoid(inplace=inplace)
#
#     def forward(self, x):
#         return x * self.sigmoid(x)
#
#
# class CoordAtt(nn.Module):
#     def __init__(self, inp, oup, reduction=32):
#         super(CoordAtt, self).__init__()
#         # self.pool_w = nn.AdaptiveAvgPool1d(1)
#         self.pool_w = nn.AdaptiveMaxPool1d(1)
#         mip = max(6, inp // reduction)
#         self.conv1 = nn.Conv1d(inp, mip, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm1d(mip)
#         self.act = h_swish()
#         self.conv_w = nn.Conv1d(mip, oup, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#         identity = x
#         n, c, w = x.size()
#         x_w = self.pool_w(x)
#         y = torch.cat([identity, x_w], dim=2)
#         y = self.conv1(y)
#         y = self.bn1(y)
#         y = self.act(y)
#         x_ww, x_c = torch.split(y, [w, 1], dim=2)
#         a_w = self.conv_w(x_ww)
#         a_w = a_w.sigmoid()
#         out = identity * a_w
#         return out

def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)

#数据增强
# def augmentation(x, max_shift=2):
#     _, _, height, width = x.size()
#
#     h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
#     source_height_slice = slice(max(0, h_shift), h_shift + height)
#     source_width_slice = slice(max(0, w_shift), w_shift + width)
#     target_height_slice = slice(max(0, -h_shift), -h_shift + height)
#     target_width_slice = slice(max(0, -w_shift), -w_shift + width)
#
#     shifted_image = torch.zeros(*x.size())
#     shifted_image[:, :, source_height_slice, source_width_slice] = x[:, :, target_height_slice, target_width_slice]
#     return shifted_image.float()


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=NUM_ROUTING_ITERATIONS):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations
        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)])

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]

            logits = torch.zeros(*priors.size()).cuda()
            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            # for capsule in self.capsules:
            #     outputs = capsule(x)
            #     a = outputs.view(x.size(0),-1,1)
            # print(outputs.size())
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs


class CapsuleNet(nn.Module):
    def __init__(self):
        super(CapsuleNet, self).__init__()
        self.p1_1 = nn.Sequential(fast(out_channels=50, kernel_size=18, stride=2),
                                  # nn.Conv1d(1, 50, kernel_size=18, stride=2),#输出通道50，
                                  nn.BatchNorm1d(50),
                                  nn.ReLU(inplace=True))
        self.p1_2 = nn.Sequential(nn.Conv1d(50, 30, kernel_size=10, stride=2),
                                  nn.BatchNorm1d(30),
                                  nn.ReLU(inplace=True))
        self.p1_3 = nn.LPPool1d(2, kernel_size=2)
        # self.p1_3 = nn.MaxPool1d(kernel_size=2)
        self.p2_1 = nn.Sequential(fast(out_channels=50, kernel_size=6, stride=1),
                                  # nn.Conv1d(1, 50, kernel_size=6, stride=1),
                                  nn.BatchNorm1d(50),
                                  nn.ReLU(inplace=True))
        self.p2_2 = nn.Sequential(nn.Conv1d(50, 40, kernel_size=6, stride=1),
                                  nn.BatchNorm1d(40),
                                  nn.ReLU(inplace=True))
        self.p2_3 = nn.LPPool1d(2, kernel_size=2)
        # self.p2_3 = nn.MaxPool1d(kernel_size=2)
        self.p2_4 = nn.Sequential(nn.Conv1d(40, 30, kernel_size=6, stride=1),
                                  nn.BatchNorm1d(30),
                                  nn.ReLU())
        self.p2_5 = nn.Sequential(nn.Conv1d(30, 30, kernel_size=6, stride=2),
                                  nn.BatchNorm1d(30),
                                  nn.ReLU())
        self.p2_6 = nn.LPPool1d(2, kernel_size=2)
        self.primary_capsules = CapsuleLayer(num_capsules=4, num_route_nodes=-1, in_channels=30, out_channels=20,
                                             kernel_size=5, stride=2)
        self.digit_capsules = CapsuleLayer(num_capsules=NUM_CLASSES, num_route_nodes=20 * 15 * 4, in_channels=4,
                                           out_channels=8)

        self.decoder = nn.Sequential(
            nn.Linear(8 * NUM_CLASSES, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        p1 = self.p1_3(self.p1_2(self.p1_1(x)))
        p2 = self.p2_6(self.p2_5(self.p2_4(self.p2_3(self.p2_2(self.p2_1(x))))))
        x = torch.mul(p1, p2)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)


        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)

        if y is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = classes.max(dim=1)
            y = torch.eye(NUM_CLASSES).cuda().index_select(dim=0, index=max_length_indices.data)

        reconstructions = self.decoder((x * y[:, :, None]).reshape(x.size(0), -1))

        return classes, reconstructions


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(reduction='sum')

    def forward(self, images, labels, classes, reconstructions):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()

        assert torch.numel(images) == torch.numel(reconstructions)
        images = images.view(reconstructions.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)


if __name__ == "__main__":
    from torch.autograd import Variable
    from torch.optim import Adam
    from torchnet.engine import Engine
    from torchnet.logger import VisdomPlotLogger, VisdomLogger
    from torchvision.utils import make_grid
    from datasave import Train, Test
    from tqdm import tqdm
    import torchnet as tnt

    model = CapsuleNet()
    # model.load_state_dict(torch.load('epochs/epoch_327.pt'))
    model.cuda()

    print("# parameters:", sum(param.numel() for param in model.parameters()))

    optimizer = Adam(model.parameters())

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
    confusion_meter = tnt.meter.ConfusionMeter(NUM_CLASSES, normalized=True)

    train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss'})
    train_error_logger = VisdomPlotLogger('line', opts={'title': 'Train Accuracy'})
    test_loss_logger = VisdomPlotLogger('line', opts={'title': 'Test Loss'})
    test_accuracy_logger = VisdomPlotLogger('line', opts={'title': 'Test Accuracy'})
    confusion_logger = VisdomLogger('heatmap', opts={'title': 'Confusion matrix',
                                                     'columnnames': list(range(NUM_CLASSES)),
                                                     'rownames': list(range(NUM_CLASSES))})
    ground_truth_logger = VisdomLogger('image', opts={'title': 'Ground Truth'})
    reconstruction_logger = VisdomLogger('image', opts={'title': 'Reconstruction'})

    capsule_loss = CapsuleLoss()

#这是载入数据集的接口
    def get_iterator(mode):
        if mode is True:
            dataset = Train
        elif mode is False:
            dataset = Test
        data = dataset.Data.float()
        labels = dataset.Label

        tensor_dataset = tnt.dataset.TensorDataset([data, labels])

        return tensor_dataset.parallel(batch_size=BATCH_SIZE, num_workers=0, shuffle=mode)


    def processor(sample):
        data, labels, training = sample

        data = data.cuda()
        labels = labels.long().cuda()

        #labels = torch.eye(NUM_CLASSES).index_select(dim=0, index=labels)
        labels = F.one_hot(labels, num_classes=10)
        # data = Variable(data).cuda()
        # labels = Variable(labels).cuda()

        if training:
            classes, reconstructions = model(data, labels)
        else:
            classes, reconstructions = model(data)

        loss = capsule_loss(data, labels, classes, reconstructions)

        return loss, classes


    def reset_meters():
        meter_accuracy.reset()
        meter_loss.reset()
        confusion_meter.reset()


    def on_sample(state):
        state['sample'].append(state['train'])


    def on_forward(state):
        meter_accuracy.add(state['output'].data, torch.LongTensor(state['sample'][1].long()))
        confusion_meter.add(state['output'].data, torch.LongTensor(state['sample'][1].long()))
        meter_loss.add(state['loss'].item())


    def on_start_epoch(state):
        reset_meters()
        state['iterator'] = tqdm(state['iterator'])


    def on_end_epoch(state):
        print('[Epoch %d] Training Loss: %.4f (Accuracy: %.2f%%)' % (
            state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

        train_loss_logger.log(state['epoch'], meter_loss.value()[0])
        train_error_logger.log(state['epoch'], meter_accuracy.value()[0])

        reset_meters()

        engine.test(processor, get_iterator(False))
        test_loss_logger.log(state['epoch'], meter_loss.value()[0])
        test_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0])
        confusion_logger.log(confusion_meter.value())

        print('[Epoch %d] Testing Loss: %.4f (Accuracy: %.2f%%)' % (
            state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

        torch.save(model.state_dict(), 'epochs/epoch_%d.pt' % state['epoch'])

        # Reconstruction visualization.

        test_sample = next(iter(get_iterator(False)))

        ground_truth = (test_sample[0])
        _, reconstructions = model(Variable(ground_truth).cuda())
        reconstruction = reconstructions.cpu().view_as(ground_truth).data
        ground_truth_logger.log(
            make_grid(ground_truth.reshape(100, 1, 32, 32), nrow=int(BATCH_SIZE ** 0.5), normalize=True, value_range=(0, 1)).numpy())
        reconstruction_logger.log(
            make_grid(reconstruction.reshape(100, 1, 32, 32), nrow=int(BATCH_SIZE ** 0.5), normalize=True, value_range=(0, 1)).numpy())

    # def on_start(state):
    #     state['epoch'] = 327
    #
    # engine.hooks['on_start'] = on_start
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    engine.train(processor, get_iterator(True), maxepoch=NUM_EPOCHS, optimizer=optimizer)