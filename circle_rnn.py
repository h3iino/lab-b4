import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import sys, os 

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
        # self.point = np.concatenate(([point, point[:num_time]]))  # pointの先頭部分を後ろにつける
        # self.point = np.concatenate(([point, point[:1]]))  # pointの先頭部分を後ろにつける
        self.point = point
        self.len = len(point)
        # self.num_time = num_time
        self.is_Noise = is_Noise

    def __getitem__(self, index):  # 処理はここにかく
        if self.is_Noise == True:
            # input_data = self.point[index:index+self.num_time] + np.random.normal(scale=0.001)
            # target_data = self.point[index+self.num_time] + np.random.normal(scale=0.001)
            # input_data = self.point[index-1] + np.random.normal(scale=0.001)
            # target_data = self.point[index] + np.random.normal(scale=0.001)
            input_data = self.point[:-1] + np.random.normal(scale=0.001)
            target_data = self.point[1:] + np.random.normal(scale=0.001)

        else:
            # input_data = self.point[index:index+self.num_time]  # indexの最初と最後の部分の重複に注意
            # target_data = self.point[index+self.num_time]
            # input_data = self.point[index-1]
            # target_data = self.point[index]
            input_data = self.point[:-1]  # indexの最初と最後の部分の重複に注意
            target_data = self.point[1:]
        return input_data, target_data

    def __len__(self):
        return self.len


# Rnn class
class RnnNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.RNN(2, 32)
        self.fc = torch.nn.Linear(32, 2)

    def forward(self, x, hidden):
        x, h = self.rnn(x, hidden)
        output = self.fc(x)
        # output = torch.tanh(self.fc(x))
        return output, h


class RnncellNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnncell = torch.nn.RNNCell(2, 32)
        self.fc = torch.nn.Linear(32, 2)

    def forward(self, x, hidden):
        output = torch.Tensor()
        # for idx in range(len(x)):
        for idx in range(x.shape[1]):
            # print(idx, x.shape)
            hidden = self.rnncell(x[:, idx], hidden)
            output = torch.cat((output, hidden))
        output = output.reshape(len(x), -1, 32)
        # output = self.fc(output)
        # output = torch.tanh(self.fc(output[:, -1]))
        output = torch.tanh(self.fc(output))
        # print("output", output)
        return output, hidden


def training(train_loader, model, criterion, optimizer, model_flag):
    train_loss = 0
    model.zero_grad()

    # for batch_data in train_loader:
    #     input_data, target_data = batch_data
    for i, (input_data, target_data) in enumerate(train_loader):    
        # print(input_data.shape, target_data.shape)
        if model_flag == "Rnn":
            hidden = torch.zeros(1, 100, 32)  # (num_layers, num_batch, hidden_size)
            output, hidden = model(input_data.float(), hidden.float())  # by RnnNet class
            output = output[:, -1, :]  # by RnnNet class
        elif model_flag == "Rnncell":
            hidden = torch.zeros(1, 32)  # RnncellNet class (num_batch, hidden_size)
            # for k in range(input_data.size()[0]):  # 100回で1loop
            #     print(input_data.size()[0])
            #     # input_cell = input_data[0, k].unsqueeze(0)
            #     # input_cell = input_data.unsqueeze(0)
            #     # print(input_cell.shape)
            #     output, hidden = model(input_data.float(), hidden.float())
            # output, hidden = model(input_data[0].float(), hidden.float())
            output, hidden = model(input_data.float(), hidden.float())
            # output = output[-1]
        else:
            print("model flag Error")
            sys.exit()
        
        loss = criterion(output[:, -1], target_data[:, -1].float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    ave_train_loss = train_loss / len(train_loader.dataset)
    return ave_train_loss

def testing(test_loader, model, criterion, optimizer, normalize_rate):
    val_loss = 0
    record_point_output = [[0, 0]]
    for input_data, target_data in test_loader:
        hidden = torch.zeros(1, 10, 32)  # (num_layers, num_batch, hidden_size)
        point_output, hidden = model(input_data.float(), hidden.float())
        point_output = point_output[:, 9, :]
        # target_data = torch.unsqueeze(target_data, 0)
        loss = criterion(point_output, target_data.float())
        val_loss += loss.item()

        point_output = point_output / normalize_rate  # range(-1 ~ 1)
        point_output = point_output.reshape(len(point_output),2).detach()
        record_point_output = np.concatenate(([record_point_output, point_output]), axis=0)
    ave_val_loss = val_loss / len(test_loader.dataset)
    return ave_val_loss, record_point_output

def testing_openloop(test_loader, model, criterion, optimizer, normalize_rate, model_flag):
    val_loss = 0
    record_point_output = [[-1, 0]]
    begin = torch.tensor([[[-1, 0]]])
    begin = begin * normalize_rate
    if model_flag == "Rnncell":
        # begin = begin.squeeze(0)  # by RnncellNet class
        hidden = torch.zeros(1, 32)  # RnncellNet class (num_batch, hidden_size)
    for i, (_, target_data) in enumerate(test_loader):
        if i == 99:
            break

        if model_flag == "Rnn":
            hidden = torch.zeros(1, 1, 32)  # (num_layers, num_batch, hidden_size)
            point_output, hidden = model(begin.float(), hidden.float())
            begin = point_output
            point_output = point_output[:, 0, :]
        elif model_flag == "Rnncell":
            # hidden = torch.zeros(1, 32)  # RnncellNet class (num_batch, hidden_size)
            point_output, hidden = model(begin.float(), hidden.float())
            begin = point_output
        else:
            print("model flag Error")
            sys.exit()

        # print("aa", point_output.shape, target_data[:, i].unsqueeze(0).shape)
        loss = criterion(point_output, target_data[:, i].unsqueeze(0).float())
        val_loss += loss.item()

        point_output = point_output / normalize_rate
        point_output = point_output.reshape(len(point_output),2).detach()
        record_point_output = np.concatenate(([record_point_output, point_output]), axis=0)
    ave_val_loss = val_loss / len(test_loader.dataset)
    return ave_val_loss, record_point_output

def drawing_plots(normalize_points, points):
    path = 'movies/'
    test_fig = plt.figure(figsize=(6.4, 6.4))
    plt.grid()
    plt.plot(points[:][0], points[:][1], color='blue', alpha=0.2)
    plt.scatter(normalize_points[:][0], normalize_points[:][1], color='orange')
    test_fig.savefig(path + "circle_rnn_plot_0521(rnncell).png")
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
    loss_fig.savefig(path + "circle_rnn_loss_0521(rnncell).png")
    plt.show()

def frame_update(i, record_output, gif_plot_x0, gif_plot_x1):
    if i != 0:
        # Clear the current graph.
        plt.cla()
    # range of the graph
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.title("circle plots by RNN")  

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
    ani.save(path + "output_circle(Rnn)_drawing_0521(rnncell).gif", writer="imagemagick")

def main():
    rate = 0.8
    num_div = 100
    num_epoch = 100
    # num_time = 10
    num_batch = 1
    train_loss_list = []
    val_loss_list = []
    # model_flag = "Rnn"
    model_flag = "Rnncell"  # "Rnn" or "Rnncell"
    is_save = True  # save the model parameters 

    points = make_circle_points(num_div)
    normalize_points, normalize_rate = plot_normalize(points, rate)
    # train_dataset = point_dataset(normalize_points, num_time, is_Noise=False)
    # test_dataset = point_dataset(normalize_points, 1, is_Noise=True)
    train_dataset = point_dataset(normalize_points, is_Noise=False)
    test_dataset = point_dataset(normalize_points, is_Noise=True)


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=num_batch, 
                                                shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=num_batch, 
                                                shuffle=False, num_workers=4)

    if model_flag == "Rnn":
        model = RnnNet()
    elif model_flag == "Rnncell":
        model = RnncellNet()
    else:
        print("model flag Error")
        sys.exit()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)   #adam  lr=0.0001

    for epoch in range(num_epoch):
        # train
        model.train()
        ave_train_loss = training(train_loader, model, criterion, optimizer, model_flag)

        # eval
        model.eval()
        # ave_val_loss, _ = testing(test_loader, model, criterion, optimizer, normalize_rate)
        ave_val_loss, _ = testing_openloop(test_loader, model, criterion, optimizer, normalize_rate, model_flag)
        print(f"Epoch [{epoch+1}/{num_epoch}], Loss: {ave_train_loss:.5f},"
            f"val_loss: {ave_val_loss:.5f}")

        # record losses
        train_loss_list.append(ave_train_loss)
        val_loss_list.append(ave_val_loss)
    
    drawing_loss_graph(num_epoch, train_loss_list, val_loss_list)

    # save parameters of the model
    if is_save == True:
        model_path = 'model_circle_rnn.pth'
        optim_path = 'optim_circle_rnn.pth'
        torch.save(model.state_dict(), model_path)
        torch.save(optimizer.state_dict(), optim_path)

    # initialize parameters
    if model_flag == "Rnn":
        model2 = RnnNet()
    elif model_flag == "Rnncell":
        model2 = RnncellNet()
    else:
        print("model flag Error")
        sys.exit()
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.0001)   #adam  lr=0.0001
    # read parameters of the model
    model_path = 'model_circle_rnn.pth'
    model2.load_state_dict(torch.load(model_path))
    # optimizer2.load_state_dict(torch.load(optim_path))

    # test
    model2.eval()
    # ave_test_loss, record_point_output = testing(test_loader, model2, criterion, optimizer2, normalize_rate)
    ave_test_loss, record_point_output = testing_openloop(test_loader, model2, criterion, optimizer2, normalize_rate, model_flag)
    print(f"Test Loss: {ave_test_loss:.5f}")

    # record_point_output = np.delete(record_point_output, obj=0, axis=0)  # Delete the initial value (Row: 0)
    drawing_plots([record_point_output[:, 0], record_point_output[:, 1]], [points[:, 0], points[:, 1]])

    make_gif(record_point_output)

if __name__ == "__main__":
    main()