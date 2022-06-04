from tqdm.notebook import tqdm
import torch
import numpy as np
from math import ceil as ceiling
from torch import Tensor
from torch import nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as rplc
import time
from torch import nn
from torch import bmm as matrix_product
from torch import transpose as transp
from torch import cat as concat
import matplotlib.pyplot as plt
b_s = 200
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
loader_tr = rplc.DataLoader(torchvision.datasets.CIFAR10("data/cifar",transform=transform_train, download = True),shuffle=True, batch_size=b_s)
loader_te = rplc.DataLoader(torchvision.datasets.CIFAR10("data/cifar",train=False, transform=transform_test , download = True),batch_size=b_s)

def train(epoch, model, opt):
    losses = []
    loss_func = nn.CrossEntropyLoss()
    for id, (images, labels) in tqdm(enumerate(loader_tr),
	total=len(loader_tr)):
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        opt.zero_grad()
        outs = model(images)
        cur_losses = loss_func(outs, labels)
        cur_losses.backward()
        opt.step()
        losses += [cur_losses.item()]
    print('Train current loss: {:.2f}'.format(np.mean(losses)))
    return np.mean(losses)
def test(model):
    acc = 0.0
    counter = 0.0
    model.eval()
    loss_func = nn.CrossEntropyLoss()
    with torch.no_grad():
        for id, (images, labels) in tqdm(enumerate(loader_te),
		total=len(loader_te)):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            outs = model(images)
            tmp = torch.sum(torch.max(outs, dim=1)[1] == labels).item()
            acc = acc + tmp
            counter = counter + images.shape[0]
    acc = (acc / counter) * 100.
    print('Test current accuracy: {:.2f}'.format(acc))
    return acc
def norm(out_sizes):
    return nn.BatchNorm2d(out_sizes)

def conv_nxn(in_sizes, out_sizes):
    return nn.Conv2d(in_sizes, out_sizes, 3, 1, 1)

def cat_time(ts, times):
    bias, tmp, we, hid = ts.shape
    return concat((ts, times.expand(bias, 1, we, hid)), dim=1)

class MYF(nn.Module):
  def compute_f_adf(self, h, times, g_outs):
        req_o = self.forward(h, times)
        b_s = h.shape[0]
        tmp = g_outs
        unchangable = tuple(self.parameters())
        adf_d_hidden, adfd_times, *adf_d_parameters = torch.autograd.grad((req_o,), (h, times) + unchangable, grad_outputs=(tmp))
        adf_d_parameters = concat([parameter_gradients.flatten() 
        for parameter_gradients in adf_d_parameters])
        adf_d_parameters = adf_d_parameters[None, :]
        adf_d_parameters = adf_d_parameters.expand(b_s, -1) / b_s
        adfd_times = adfd_times.expand(b_s, 1) / b_s
        return req_o, adf_d_hidden, adfd_times, adf_d_parameters

  def flat_to_one(self):
        shapes = []
        flatten = []
        for i in self.parameters():
            shapes.append(i.size())
            flatten.append(i.flatten())
        tmp = concat(flatten)
        return tmp

def euler_solve(h, times, times_last, f, acc_param = 0.05):
    steps = ceiling(abs(times_last - times) / acc_param)
    time_step = (times_last - times)/steps
    for i in range(steps):
        h = h + time_step * f(h, times) 
        times = times + time_step
    return h

class Adjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h_begin, times, parameters, dynamics_f, 
    ode_accuracy):
        max_time = times.size(0)
        bias, *shapes_h = h_begin.size()
        with torch.no_grad():
            h = torch.zeros(max_time, bias, *shapes_h).to(h_begin)
            h[0] = h_begin
            for cur_t in range(max_time - 1):
                h_begin = euler_solve(h_begin, times[cur_t], 
                times[cur_t+1],
                dynamics_f, ode_accuracy)
                h[cur_t+1] = h_begin
        ctx.dynamics_f = dynamics_f
        ctx.ode_accuracy = ode_accuracy
        ctx.save_for_backward(times, h.clone(), parameters)
        return h

    @staticmethod
    def backward(ctx, dLdh):
        ode_accuracy = ctx.ode_accuracy
        dynamics_f = ctx.dynamics_f
        times, h, params = ctx.saved_tensors
        max_time, bias, *shapes_h = h.size()
        parameters_cnt = params.size(0)
        dimensions = np.prod(shapes_h)

        def aug_dynamics(aug_h_iter, times_iter):
            adf_d_hidden = torch.zeros(bias, *shapes_h)
            adf_d_parameters = torch.zeros(bias, parameters_cnt)
            adfd_times = torch.zeros(bias, 1)
            h_iter, grad_outs = aug_h_iter[:, :dimensions], aug_h_iter[:,dimensions:2*dimensions]  
            h_iter = h_iter.reshape(bias, *shapes_h)
            grad_outs = grad_outs.reshape(bias, *shapes_h)
            with torch.set_grad_enabled(True):
                times_iter = times_iter.requires_grad_(True)
                h_iter = h_iter.requires_grad_(True)
                dadt = dynamics_f.compute_f_adf(h_iter, times_iter, g_outs = grad_outs)
                function_out, adf_d_hidden, adfd_times, adf_d_parameters = dadt
                adf_d_hidden = adf_d_hidden.to(h_iter)
                adf_d_parameters = adf_d_parameters.to(h_iter)
                adfd_times = adfd_times.to(h_iter)
            function_out = function_out.reshape(bias, dimensions)
            adf_d_hidden = adf_d_hidden.reshape(bias, dimensions) 
            return concat((function_out, -adf_d_hidden, -adf_d_parameters, -adfd_times),dim=1)

        dLdh = dLdh.reshape(max_time, bias, dimensions)
        
        with torch.no_grad():
            adjoint_h = torch.zeros(bias, dimensions).to(dLdh)
            adjoint_p = torch.zeros(bias, parameters_cnt).to(dLdh)
            adjoint_times = torch.zeros(max_time, bias, 1).to(dLdh)

            for cur_t in range(max_time - 1, 0, -1):
                h_iter = h[cur_t]
                times_iter = times[cur_t]
                f_iter = dynamics_f(h_iter, 
                times_iter).reshape(bias, dimensions)
                dLdh_iter = dLdh[cur_t]
                dLdtimes_iter = matrix_product(transp(dLdh_iter[:,
                :, None],1, 2), f_iter[:, :, None])[:, 0]
                adjoint_h += dLdh_iter          
                adjoint_times[cur_t] = adjoint_times[cur_t]
                - dLdtimes_iter
                augmented_h = concat((h_iter.reshape(bias, 
                dimensions), adjoint_h, 
                torch.zeros(bias, parameters_cnt).to(h), 
                adjoint_times[cur_t]), dim=-1)
                augmented_solution = euler_solve(augmented_h, times_iter
                , times[cur_t-1], aug_dynamics, ode_accuracy)
                adjoint_h[:] = augmented_solution[:, 
                dimensions:2*dimensions]           
                adjoint_p[:] += augmented_solution[:,
                2*dimensions:2*dimensions + parameters_cnt]
                adjoint_times[cur_t-1] = augmented_solution[:,
                2*dimensions + parameters_cnt:]
                del augmented_h, augmented_solution
            dLdh_0 = dLdh[0]
            dLdtimes_0 = matrix_product(transp(dLdh_0[:,
            :, None], 1, 2), f_iter[:, :, None])[:, 0]
            adjoint_h += dLdh_0
            adjoint_times[0] = adjoint_times[0] - dLdtimes_0
        return adjoint_h.reshape(bias, *shapes_h), adjoint_times, adjoint_p, None, None

class HelperClass(nn.Module):
    def __init__(self, dynamics_f, ode_accuracy):
        super().__init__()
        self.dynamics_f = dynamics_f
        self.ode_accuracy = ode_accuracy

    def fwd_bcwd(self, h0, times = [0., 1.]):
        times = Tensor(times)
        times = times.to(h0)
        h = Adjoint.apply(h0, times, self.dynamics_f.flat_to_one(), 
        self.dynamics_f, self.ode_accuracy)
        return h[-1]

class res_block(nn.Module):
    def __init__(self, out_sizes):
        super().__init__()
        self.conv_out = conv_nxn(out_sizes, out_sizes)
        self.norm = norm(out_sizes)
        self.relu = nn.ReLU(inplace = True)
        
    def forward(self, h):
        enter = h
        h = self.relu(self.norm(self.conv_out(h)))
        h = self.relu(self.norm(self.conv_out(h)))
        h = self.relu(h + enter)
        return h

class DynamicsBlock(MYF):
    def __init__(self, out_sizes):
        super().__init__()
        self.conv_out = conv_nxn(out_sizes + 1, out_sizes)
        self.norm = norm(out_sizes)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, h, times):
        h = cat_time(h, times)
        h = self.relu(self.norm(self.conv_out(h)))
        h = cat_time(h, times)
        h = self.relu(self.norm(self.conv_out(h)))
        return h

def parameters_summary(model):
  parameters = [l for l in model.parameters()
  if l.requires_grad]
  layers = [k for k in model.children()]
  second_counter = 0
  total_params = 0
  for element in layers:
    tmp_parameters_cnt = 0
    try:
      bias = (element.bias is not None)
    except:
      bias = False 
    if (bias):
      tmp_parameters_cnt = parameters[second_counter].numel()
      second_counter = second_counter + 1
    else:
      tmp_parameters_cnt = parameters[second_counter].numel()
      + parameters[second_counter + 1].numel()
      second_counter = second_counter + 2
    total_params += tmp_parameters_cnt
  return total_params

def memory_usage(model):
  memory_parameters = sum([param.nelement()*param.element_size()
  for param in model.parameters()])
  memory_buffers = sum([buffer.nelement()*buffer.element_size()
  for buffer in model.buffers()])
  used_memory = memory_parameters + memory_buffers
  return used_memory

epochs_cnt = 50

class ODEClassifier(nn.Module):
  def __init__(self, ode, base_channels):
        super().__init__()
        self.to_16 = conv_nxn(3, 16)
        self.to_32 = conv_nxn(16, 32)
        self.to_64 = conv_nxn(32, 64)
        self.relu = nn.ReLU(inplace=True) 
        self.norm1 = norm(base_channels)
        self.norm2 = norm(base_channels * 2)
        self.norm3 = norm(base_channels * 4)
        self.avg_pool2 = nn.AvgPool2d(2)

        self.ode1 = ode[0]
        self.ode2 = ode[1]

        self.linear = nn.Linear(base_channels * 16, 10)

  def forward(self, inout):
        inout = self.to_16(inout)
        inout = self.norm1(inout)
        inout = self.relu(inout)
        inout = self.avg_pool2(inout)
        inout = self.to_32(inout)
        inout = self.norm2(inout)
        inout = self.relu(inout)
        inout = self.avg_pool2(inout)
        inout = self.ode1.fwd_bcwd(inout)
        inout = self.to_64(inout)
        inout = self.norm3(inout)
        inout = self.relu(inout)
        inout = self.avg_pool2(inout)
        inout = self.ode2.fwd_bcwd(inout)
        inout = self.avg_pool2(inout) 
        inout = inout.view(inout.size(0), -1)
        out = self.linear(inout)

        return out

train_losses1 = [[],[],[]]
test_accuracy1 = [[],[],[]]
times1 = [[],[],[]]
memory_usage1 = [[],[],[]]
parameters1 = [[],[],[]]
for hyper in ([0, 0.5], [1, 0.1], [2, 0.25]):
    ode1 = HelperClass(DynamicsBlock(32), hyper[1])
    ode2 = HelperClass(DynamicsBlock(64), hyper[1])
    model1 = ODEClassifier([ode1, ode2], 16)
    opt1 = torch.optim.Adam(model1.parameters(), lr = 0.001)
    if torch.cuda.is_available():
      model1 = model1.to("cuda")
    
    start_time = time.time()
    for epoch in range(1, epochs_cnt + 1):
        train_losses1[hyper[0]].append(train(epoch, model1, opt1))
        test_accuracy1[hyper[0]].append(test(model1))
        print(test_accuracy1)
        print(train_losses1)

    elapsed_time = time.time() - start_time
    print("Time spent:", elapsed_time)
    times1[hyper[0]] = elapsed_time
    memory_usage1[hyper[0]] = memory_usage(model1)
    parameters1[hyper[0]] = parameters_summary(model1)
    print(memory_usage1[hyper[0]])
    print(parameters1[hyper[0]])
    del model1

print("times1:", times1,
      "memory_usage1:",  memory_usage1 ,
      "parameters1:",  parameters1)

class ResidualNetwork(nn.Module):
    def __init__(self, res_block, base_channels, repeat_cnt):
        super().__init__()
        self.to_16 = conv_nxn(3, 16)
        self.to_32 = conv_nxn(16, 32)
        self.to_64 = conv_nxn(32, 64)
        self.relu = nn.ReLU(inplace=True)
        self.norm1 = norm(base_channels)
        self.norm2 = norm(base_channels * 2)
        self.norm3 = norm(base_channels * 4)
        self.avg_pool2 = nn.AvgPool2d(2)

        self.res_block1 = self.create_block(res_block, 32, repeat_cnt)
        self.res_block2 = self.create_block(res_block, 64, repeat_cnt)

        self.linear = nn.Linear(base_channels * 16, 10)

    def create_block(self, block, out_sizes, cnt):
      layers = []
      for i in range (0, cnt):
        layers.append(block(out_sizes, out_sizes))
      return nn.Sequential(*layers)

    def forward(self, inout):
        inout = self.to_16(inout)
        inout = self.norm1(inout)
        inout = self.relu(inout)
        inout = self.avg_pool2(inout)
        inout = self.to_32(inout)
        inout = self.norm2(inout)
        inout = self.relu(inout)
        inout = self.avg_pool2(inout)
        inout = self.res_block1(inout)
        inout = self.to_64(inout)
        inout = self.norm3(inout)
        inout = self.relu(inout)
        inout = self.avg_pool2(inout)
        inout = self.res_block2(inout)
        inout = self.avg_pool2(inout) 
        inout = inout.view(inout.size(0), -1)
        out = self.linear(inout)
        return out

train_losses2 = [[],[],[]]
test_accuracy2 = [[],[],[]]
times2 = [[],[],[]]
memory_usage2 = [[],[],[]]
parameters2 = [[],[],[]]
for hyper in ([0, 1], [1, 2], [2, 3]):
    model2 = ResidualNetwork(res_block, 16, hyper[1])
    opt2 = torch.optim.Adam(model2.parameters(), lr = 0.001)
    if torch.cuda.is_available():
      model2 = model2.to("cuda")
    
    start_time = time.time()
    for epoch in range(1, epochs_cnt + 1):
        train_losses2[hyper[0]].append(train(epoch, model2, opt2))
        test_accuracy2[hyper[0]].append(test(model2))
        print(test_accuracy2)
        print(train_losses2)

    elapsed_time = time.time() - start_time
    print("Time spent:", elapsed_time)
    times2[hyper[0]] = elapsed_time
    memory_usage2[hyper[0]] = memory_usage(model2)
    parameters2[hyper[0]] = parameters_summary(model2)
    print(memory_usage2[hyper[0]])
    print(parameters2[hyper[0]])
    del model2

print("times2:", times2, "\n"
      "memory_usage2:",  memory_usage2 , "\n"
      "parameters2:",  parameters2 , "\n")

print(train_losses2)

class ResidualODE(nn.Module):
  def __init__(self, res_block, ode, base_channels, repeat_cnt):
        super().__init__()
        self.to_16 = conv_nxn(3, 16)
        self.to_32 = conv_nxn(16, 32)
        self.to_64 = conv_nxn(32, 64)
        self.relu = nn.ReLU(inplace=True)
        self.norm1 = norm(base_channels)
        self.norm2 = norm(base_channels * 2)
        self.norm3 = norm(base_channels * 4)
        self.avg_pool2 = nn.AvgPool2d(2)
        self.res_block1 = self.create_block(res_block, 32, repeat_cnt)
        self.ode = ode
        self.linear = nn.Linear(base_channels * 16, 10)

  def create_block(self, block, out_sizes, cnt):
      layers = []
      for i in range (0, cnt):
        layers.append(block(out_sizes, out_sizes))
      return nn.Sequential(*layers)

  def forward(self, inout):
        inout = self.to_16(inout)
        inout = self.norm1(inout)
        inout = self.relu(inout)
        inout = self.avg_pool2(inout)
        inout = self.to_32(inout)
        inout = self.norm2(inout)
        inout = self.relu(inout)
        inout = self.avg_pool2(inout)
        inout = self.res_block1(inout)
        inout = self.to_64(inout)
        inout = self.norm3(inout)
        inout = self.relu(inout)
        inout = self.avg_pool2(inout)
        inout = self.ode.fwd_bcwd(inout)
        inout = self.avg_pool2(inout) 
        inout = inout.view(inout.size(0), -1)
        out = self.linear(inout)
        return out
train_losses3 = [[], [], [], [], [], [], []]
test_accuracy3 = [[],[],[],[],[],[],[]]
times3 = [[],[],[],[],[],[],[]]
memory_usage3 = [[],[],[],[],[],[],[]]
parameters3 = [[],[],[],[],[],[],[]]
for hyper in ([0, 0.25, 1], [1, 0.25, 2], [2, 0.1, 1], 
[3, 0.1, 2], [4, 0.5, 1], [5, 0.5, 2], [6, 0.5, 3]):
    ode1 = HelperClass(DynamicsBlock(64), hyper[1])
    model3 = ResidualODE(res_block, ode1, 16, hyper[2])
    opt3 = torch.optim.Adam(model3.parameters(), lr = 0.001)
    if torch.cuda.is_available():
      model3 = model3.to("cuda")
    start_time = time.time()
    for epoch in range(1, epochs_cnt + 1):
        train_losses3[hyper[0]].append(train(epoch, model3, opt3))
        test_accuracy3[hyper[0]].append(test(model3))      
        print(test_accuracy3)
        print(train_losses3)
    elapsed_time = time.time() - start_time
    print("Time spent:", elapsed_time)
    times3[hyper[0]] = elapsed_time
    memory_usage3[hyper[0]] = memory_usage(model3)
    parameters3[hyper[0]] = parameters_summary(model3)
    print(memory_usage3[hyper[0]])
    print(parameters3[hyper[0]])
    del model3
print("times3:", times3, "\n"
      "memory_usage3:",  memory_usage3 , "\n"
      "parameters3:",  parameters3 , "\n")
size_a = 9
size_b = 5
plt.figure(figsize=(size_a, size_b))
epochs_cnt = 50
epochs = np.arange(1, epochs_cnt+1)
plt.plot(epochs, test_accuracy1[0], label = 'ODENet', color = 'blue')
plt.plot(epochs, test_accuracy2[0], label = 'ResidualODE', color = 'red')
plt.plot(epochs, test_accuracy3[0], label = 'ResNet', color = 'yellow')
plt.legend(['ODENet', 'ResidualODE', 'ResNet1']) 
plt.xlabel("epochs")
plt.ylabel("accuracy")
epochs = np.arange(1, epochs_cnt+1)
plt.plot(epochs, train_losses1[0], label = 'ODENet', color = 'blue')
plt.plot(epochs, train_losses2[0], label = 'ResidualODE', color = 'red')
plt.plot(epochs, train_losses3[0], label = 'ResNet', color = 'yellow')
plt.legend(['ODENet', 'ResidualODE', 'ResNet1']) 
plt.xlabel("epochs")
plt.ylabel("training loss")
