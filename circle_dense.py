import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import copy
# import sys, os 

def make_circle_points(num):
    # theta = np.linspace(0, 2*np.pi, num)
    theta = np.linspace(-np.pi, np.pi, num)
    x_points = np.array([np.cos(theta)])
    y_points = np.array([np.sin(theta)])
    points = np.concatenate(([x_points, y_points]), axis=0).T
    return points

def plot_normalize(points, rate):
    normalize_rate = rate / np.max(points)
    points = points * normalize_rate
    return points, normalize_rate

class point_dataset(torch.utils.data.Dataset):
    def __init__(self, point, is_Noise):
        self.point = point
        self.len = len(point)
        self.is_Noise = is_Noise

    def __getitem__(self, index):  # 処理はここにかく
        if self.is_Noise == True:
            input_data = self.point[index-1] + np.random.normal(scale=0.001)
            target_data = self.point[index] + np.random.normal(scale=0.001)
        else:
            input_data = self.point[index-1]  # indexの最初と最後の部分の重複に注意
            target_data = self.point[index]
        return input_data, target_data

    def __len__(self):
        return self.len


# Dense class
class DenseNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 32)
        self.fc2 = torch.nn.Linear(32, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        output = self.fc2(x)
        # output = torch.tanh(self.fc2(x))
        return output


def training(train_loader, model, criterion, optimizer):
    train_loss = 0
    for i, (input_data, target_data) in enumerate(train_loader):
        model.zero_grad()
        output = model(input_data.float())
        loss = criterion(output, target_data.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    ave_train_loss = train_loss / len(train_loader.dataset)
    return ave_train_loss

def training_openloop(train_loader, model, criterion, optimizer, normalize_rate):
    train_loss = 0
    begin = torch.tensor([[-1, 0]])
    begin = begin * normalize_rate
    for i, (input_data, target_data) in enumerate(train_loader):
        model.zero_grad()
        output = model(begin.float())
        begin = copy.deepcopy(output.detach())
        loss = criterion(output, target_data.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    ave_train_loss = train_loss / len(train_loader.dataset)
    return ave_train_loss

def testing(test_loader, model, criterion, optimizer):
    val_loss = 0
    record_point_output = [[0, 0]]
    # record_point_output = []
    for input_data, target_data in test_loader:
        point_output = model(input_data.float())
        loss = criterion(point_output, target_data.float())
        val_loss += loss.item()

        point_output = point_output.reshape(len(point_output),2).detach()
        record_point_output = np.concatenate(([record_point_output, point_output]), axis=0)
        # record_point_output.append(list(point_output))
    ave_val_loss = val_loss / len(test_loader.dataset)
    return ave_val_loss, record_point_output

def testing_openloop(test_loader, model, criterion, optimizer, normalize_rate):
    val_loss = 0
    record_point_output = [[0, 0]]
    begin = torch.tensor([[-1, 0]])
    begin = begin * normalize_rate
    for _, target_data in test_loader:
        point_output = model(begin.float())
        loss = criterion(point_output, target_data.float())
        begin = point_output
        point_output = point_output / normalize_rate
        val_loss += loss.item()

        point_output = point_output.reshape(len(point_output),2).detach()
        record_point_output = np.concatenate(([record_point_output, point_output]), axis=0)
    ave_val_loss = val_loss / len(test_loader.dataset)
    return ave_val_loss, record_point_output

def drawing_plots(points):
    path = 'movies/'
    test_fig = plt.figure(figsize=(6.4, 6.4))
    plt.grid()
    plt.scatter(points[:][0], points[:][1])
    test_fig.savefig(path + "circle_dense_plot_0509.png")
    plt.show()

def drawing_loss_graph(num_epoch, train_loss_list, val_loss_list):
    path = 'movies/'
    loss_fig = plt.figure()
    plt.plot(range(num_epoch), train_loss_list, color='blue', linestyle='-', label='train_loss')
    plt.plot(range(num_epoch), val_loss_list, color='green', linestyle='--', label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training and validation loss')
    plt.grid()
    loss_fig.savefig(path + "circle_dense_loss_0509.png")
    plt.show()

def frame_update(i, record_output, gif_plot_x0, gif_plot_x1):
    if i != 0:
        plt.cla()  # Clear the current graph.
    plt.xlim(-1.1, 1.1)  # range of the graph
    plt.ylim(-1.1, 1.1)  # range of the graph
    plt.title("circle plots by Dense Net")

    gif_plot_x0.append(record_output[i, 0])
    gif_plot_x1.append(record_output[i, 1])
    plt.grid()
    im_result = plt.scatter(gif_plot_x0, gif_plot_x1)

def make_gif(record_point_output):
    fig_RNN = plt.figure(figsize=(6.4, 6.4))
    path = 'movies/' 
    gif_plot_x0, gif_plot_x1 = [], []   
    ani = animation.FuncAnimation(fig_RNN, frame_update, 
                                fargs = (record_point_output, gif_plot_x0, gif_plot_x1), 
                                interval = 50, frames = 100)
    ani.save(path + "output_circle(Dense)_drawing_0509.gif", writer="imagemagick")

def main():
    rate = 0.8  # normalize range: -0.8 ~ 0.8
    num_div = 100  # numbers of plots
    num_epoch = 400
    num_batch = 1
    train_loss_list = []
    val_loss_list = []
    is_save = True  # save the model parameters 

    points = make_circle_points(num_div) 
    points, normalize_rate = plot_normalize(points, rate)  # range(-0.8~0.8)
    train_dataset = point_dataset(points, is_Noise=True)
    test_dataset = point_dataset(points, is_Noise=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=num_batch, 
                                                shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=num_batch, 
                                                shuffle=False, num_workers=4)

    model = DenseNet()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)   #adam  lr=0.0001

    for epoch in range(num_epoch):
        # train
        model.train()
        ave_train_loss = training(train_loader, model, criterion, optimizer)
        # ave_train_loss = training_openloop(train_loader, model, criterion, 
        #                                     optimizer, normalize_rate)

        # eval
        model.eval()
        # ave_val_loss, _ = testing(test_loader, model, criterion, optimizer)
        ave_val_loss, _ = testing_openloop(test_loader, model, criterion, optimizer, normalize_rate)
        print(f"Epoch [{epoch+1}/{num_epoch}], Loss: {ave_train_loss:.5f},"
            f"val_loss: {ave_val_loss:.5f}")
        
        # record losses
        train_loss_list.append(ave_train_loss)
        val_loss_list.append(ave_val_loss)
    
    drawing_loss_graph(num_epoch, train_loss_list, val_loss_list)

    # save parameters of the model
    if is_save == True:
        model_path = 'model_circle_dense.pth'
        optim_path = 'optim_circle_dense.pth'
        torch.save(model.state_dict(), model_path)
        torch.save(optimizer.state_dict(), optim_path)

    # initialize parameters
    model2 = DenseNet()
    optimizer2 = torch.optim.Adam(model.parameters(), lr=0.0001)  # adam  lr=0.0001
    # read parameters of the model
    model_path = 'model_circle_dense.pth'
    # optim_path = 'optim.pth'  # いらない
    model2.load_state_dict(torch.load(model_path))
    # optimizer2.load_state_dict(torch.load(optim_path))
    optimizer2 = torch.optim.Adam(model.parameters(), lr=0.0001)

    # test
    model2.eval()
    ave_test_loss, record_point_output = testing_openloop(test_loader, model2, criterion, 
                                                            optimizer2, normalize_rate)
    print(f"Test Loss: {ave_test_loss:.5f}")

    record_point_output = np.delete(record_point_output, obj=0, axis=0)  # Delete the 
                                                                         # initial value (Row: 0)
    drawing_plots([record_point_output[:, 0], record_point_output[:, 1]])

    make_gif(record_point_output)

if __name__ == "__main__":
    main()