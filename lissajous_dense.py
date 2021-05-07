import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
# from torchvision import transforms
import sys, os 

def make_Lissajous_points(num):
    # theta = np.linspace(0, 2*np.pi, num)
    # theta = np.linspace(-np.pi, np.pi, num)
    theta = np.linspace(-0.5*np.pi, 1.5*np.pi, num)
    x_points = np.array([2 * np.sin(1 * theta)])
    y_points = np.array([1 * np.sin(2 * theta)])
    points = np.concatenate(([x_points, y_points]), axis=0).T
    return points

def plot_normalize(points, rate):
    normalize_rate = rate / np.max(points)
    points = points * normalize_rate
    return points, normalize_rate

def quadrant_label(point):
    # label = 0
    # for i in range(len(point)):
    if point[0] > 0:
        if point[1] > 0:
            label = 0  #1st
        else:
            label = 3  #4th
    else:
        if point[1] > 0:
            label = 1  #2nd
        else:
            label = 2  #3rd
    return label

# class point_dataset(torch.utils.data.Dataset):
#     def __init__(self, point, num_time, is_Noise):
#         self.point = point
#         self.len = len(point)
#         self.is_Noise = is_Noise

#     def __getitem__(self, index):  # 処理はここにかく
#         if self.is_Noise == True:
#             input_data = self.point[index-1] + np.random.normal(scale=0.001)
#             target_data = self.point[index] + np.random.normal(scale=0.001)
#         else:
#             input_data = self.point[index-1]  # indexの最初と最後の部分の重複に注意
#             target_data = self.point[index]
#         label = quadrant_label(input_data)
#         return input_data, target_data, label

#     def __len__(self):
#         return self.len


class point_dataset(torch.utils.data.Dataset):
    def __init__(self, point, num_time, is_Noise):
        self.point = np.concatenate(([point[len(point)-num_time:], point]))  # pointの最後尾を最初につける
        self.len = len(point)
        self.num_time = num_time
        self.is_Noise = is_Noise

    def __getitem__(self, index):  # 処理はここにかく
        if self.is_Noise == True:
            input_data = self.point[index:index+self.num_time] + np.random.normal(scale=0.001)
            target_data = self.point[index+self.num_time] + np.random.normal(scale=0.001)
        else:
            # input_data = self.point[index-1]  # indexの最初と最後の部分の重複に注意
            input_data = self.point[index:index+self.num_time]  # indexの最初と最後の部分の重複に注意
            target_data = self.point[index+self.num_time]
        # label = quadrant_label(input_data[index+self.num_time-1])
        label = quadrant_label(input_data[-1])
        return input_data, target_data, label
        # return input_data, target_data
    
    def __len__(self):
        return self.len


# Dense class
class DenseNet(torch.nn.Module):
    def __init__(self, num_time):
        super().__init__()
        # self.num_time = num_time
        self.fc1_1 = torch.nn.Linear(2, 32)
        # self.fc1_1 = torch.nn.Linear(self.num_time*2, 32)
        self.fc2_1 = torch.nn.Linear(32, 2)
        self.fc2_2 = torch.nn.Linear(32, 4)

    def forward(self, x):
        # x_1 = torch.tanh(self.fc1_1(x))
        x = torch.relu(self.fc1_1(x))
        # point_output = torch.tanh(self.fc2_1(x))
        point_output = self.fc2_1(x)
        label_output = torch.nn.functional.softmax(self.fc2_2(x), dim=1)
        # print(point_output.shape)
        # print(label_output.shape)
        # sys.exit()
        return point_output, label_output


def training(train_loader, model, criterion, criterion_q, optimizer):
    train_loss = 0
    train_q_loss = 0
    for i, (input_data, target_data, label) in enumerate(train_loader):
        model.zero_grad()
        point_output, label_output = model(input_data.float())
        label_output = label_output[:, -1]  # 最後のラベル
        point_output = point_output[:, -1]  # 最後の出力

        loss_mse = criterion(point_output, target_data.float())
        loss_q = criterion_q(label_output, label.long())
        loss  =loss_mse + loss_q
        loss.backward()
        optimizer.step()

        train_loss += loss_mse.item()
        train_q_loss += loss_q.item()
    ave_train_loss = train_loss / len(train_loader.dataset)
    ave_train_q_loss = train_q_loss / len(train_loader.dataset)
    return ave_train_loss, ave_train_q_loss

def testing(test_loader, model, criterion, criterion_q, optimizer, normalize_rate):
    with torch.no_grad():
        val_loss = 0
        val_q_loss = 0
        record_point_output = [[0, 0]]
        record_label_output = []
        for input_data, target_data, label in test_loader:
            point_output, label_output = model(input_data.float())
            label_output = label_output[:, -1]  # 最後のラベル
            point_output = point_output[:, -1]  # 最後の出力

            loss = criterion(point_output, target_data.float())
            loss_q = criterion_q(label_output, label.long())
            val_loss += loss.item()
            val_q_loss += loss_q.item()

            point_output = point_output / normalize_rate  # range(-2 ~ 2)
            point_output = point_output.reshape(len(point_output),2).detach()
            label_output = torch.argmax(label_output, axis=1)
            record_point_output = np.concatenate(([record_point_output, point_output]), axis=0)
            record_label_output.append(label_output.item())
        ave_val_loss = val_loss / len(test_loader.dataset)
        ave_val_q_loss = val_q_loss / len(test_loader.dataset)
    return ave_val_loss, ave_val_q_loss, record_point_output, record_label_output

def testing_openloop(test_loader, model, criterion, criterion_q, optimizer, num_time, normalize_rate):
    val_loss = 0
    val_q_loss = 0
    record_point_output = [[0, 0]]
    record_label_output = []
    begin = torch.tensor([[-2, 0]])
    begin = begin * normalize_rate  # range(-0.8~0.8)
    for _, target_data, label in test_loader:
        # point_output, label_output = model(input_data.float())
        point_output, label_output = model(begin.float())
        label_output = label_output[-1].reshape(1, 4)  # 最後のラベル
        if len(begin) < num_time:
            point_output = point_output[-1].reshape(1, 2)  # 最後の出力
            begin = torch.cat((begin, point_output), 0)
        else:
            begin = point_output
            point_output = point_output[-1].reshape(1, 2)  # 最後の出力
        # print(begin)
        loss_mse = criterion(point_output, target_data.float())
        loss_q = criterion_q(label_output, label.long())
        val_loss += loss_mse.item()
        val_q_loss += loss_q.item()

        point_output = point_output / normalize_rate  # range(-2 ~ 2)
        point_output = point_output.reshape(len(point_output),2).detach()
        label_output = torch.argmax(label_output, axis=1)
        record_point_output = np.concatenate(([record_point_output, point_output]), axis=0)
        record_label_output.append(label_output.item())
    ave_val_loss = val_loss / len(test_loader.dataset)
    ave_val_q_loss = val_q_loss / len(test_loader.dataset)
    return ave_val_loss, ave_val_q_loss, record_point_output, record_label_output

def set_color(label):
    if label == 0:
        return "blue"  # blue
    elif label == 1:
        return "green"  # green
    elif label == 2:
        return "yellow"
    else:
        return "red"  # red

def drawing_plots(points, label):
    path = 'movies/'
    color_list = list(map(set_color, label))  # scatter color list
    test_fig = plt.figure(figsize=(12.8, 6.4))
    plt.xlim(-2.2, 2.2)  # range of the graph
    plt.ylim(-1.1, 1.1)  # range of the graph
    plt.grid()
    # plt.scatter(points[:][0], points[:][1], c=label)
    plt.scatter(points[:][0], points[:][1], c=color_list)
    test_fig.savefig(path + "lissajous_dense_plot_0503.png")
    plt.show()

def drawing_loss_graph(num_epoch, train_loss_list, train_loss_q_list, val_loss_list, val_loss_q_list):
    path = 'movies/'
    loss_fig = plt.figure()
    plt.plot(range(num_epoch), train_loss_list, color='blue', linestyle='-', label='train_loss')
    plt.plot(range(num_epoch), train_loss_q_list, color='orange', linestyle='-', label='train_q_loss')
    plt.plot(range(num_epoch), val_loss_list, color='green', linestyle='--', label='val_loss')
    plt.plot(range(num_epoch), val_loss_q_list, color='red', linestyle='--', label='val_q_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training and validation loss')
    plt.grid()
    loss_fig.savefig(path + "lissajous_dense_loss_0503.png")
    plt.show()

def frame_update(i, record_output, gif_plot_x0, gif_plot_x1, record_label_output, color_list):
    if i != 0:
        plt.cla()  # Clear the current graph
    plt.xlim(-2.2, 2.2)  # range of the graph
    plt.ylim(-1.1, 1.1)  # range of the graph
    plt.title(f"{record_label_output[i] + 1} quadrant")  # label(0~3)->(1~4)

    color_list.append(set_color(record_label_output[i]))  #scatter color list

    gif_plot_x0.append(record_output[i, 0])
    gif_plot_x1.append(record_output[i, 1])
    plt.grid()
    im_result = plt.scatter(gif_plot_x0, gif_plot_x1, c=color_list)

def make_gif(record_point_output, record_label_output, num_div):
    fig_RNN = plt.figure(figsize=(12.8, 6.4))
    path = 'movies/' 
    gif_plot_x0, gif_plot_x1 = [], [] 
    color_list = []  
    ani = animation.FuncAnimation(fig_RNN, frame_update, 
                                fargs = (record_point_output, gif_plot_x0, gif_plot_x1, record_label_output, color_list), 
                                interval = 50, frames = num_div)
    ani.save(path + "output_lissajous(Dense)_drawing_0503.gif", writer="imagemagick")

def main():
    rate = 0.8
    num_div = 100
    num_epoch = 400
    num_time = 3
    num_batch = 1
    train_loss_list, train_loss_q_list = [], []
    val_loss_list, val_loss_q_list = [], []
    is_save = True  # save the model parameters 

    points = make_Lissajous_points(num_div)
    points, normalize_rate = plot_normalize(points, rate)
    train_dataset = point_dataset(points, num_time, is_Noise=True)
    # train_dataset = point_dataset(points, num_time, is_Noise=False)  # try
    test_dataset = point_dataset(points, num_time, is_Noise=False)
    # for i in range(3):
    #     print(train_dataset[i])
    # sys.exit()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=num_batch, 
                                                shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, 
                                                shuffle=False, num_workers=4)

    model = DenseNet(num_time)
    criterion = torch.nn.MSELoss()
    criterion_q = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)   # adam  lr=0.0001

    for epoch in range(num_epoch):
        # train
        model.train()
        ave_train_loss, ave_train_q_loss = training(train_loader, model, criterion, criterion_q, optimizer)

        # eval
        model.eval()
        # ave_val_loss, ave_val_q_loss, _, _ = testing(test_loader, model, criterion, criterion_q, optimizer, normalize_rate)
        ave_val_loss, ave_val_q_loss, record_point_output, record_label_output = testing_openloop(test_loader, model, criterion, criterion_q, optimizer, num_time, normalize_rate)
        print(f"Epoch [{epoch+1}/{num_epoch}], (point)loss: {ave_train_loss:.5f},"
            f"val_loss: {ave_val_loss:.5f} | (label)loss: {ave_train_q_loss:.5f}, {ave_val_q_loss:.5f}")

        # if epoch % 20 == 0:
        #     plt.scatter(record_point_output[:,0], record_point_output[:,1])
        #     plt.show()

        # record losses
        train_loss_list.append(ave_train_loss)
        train_loss_q_list.append(ave_train_q_loss)
        val_loss_list.append(ave_val_loss)
        val_loss_q_list.append(ave_val_q_loss)
    
    drawing_loss_graph(num_epoch, train_loss_list, train_loss_q_list, val_loss_list, val_loss_q_list)

    # save parameters of the model
    if is_save == True:
        model_path = 'model_lissajous_dense.pth'
        optim_path = 'optim_lissajous_dense.pth'
        torch.save(model.state_dict(), model_path)
        torch.save(optimizer.state_dict(), optim_path)

    # initialize parameters
    model2 = DenseNet(num_time)
    optimizer2 = torch.optim.Adam(model.parameters(), lr=0.0001)   # adam  lr=0.0001
    # read parameters of the model
    model_path = 'model_lissajous_dense.pth'
    model2.load_state_dict(torch.load(model_path))
    # optimizer2.load_state_dict(torch.load(optim_path))

    # test
    model2.eval()
    ave_test_loss, ave_test_q_loss, record_point_output, record_label_output = testing_openloop(test_loader, 
                                                        model2, criterion, criterion_q, optimizer2, num_time, normalize_rate)
    # ave_test_loss, ave_test_q_loss, record_point_output, record_label_output = testing(test_loader, 
    #                                                     model2, criterion, criterion_q, optimizer2, normalize_rate)
    print(f"Test Loss: {ave_test_loss:.5f}, label: {ave_test_q_loss:.5f}")

    record_point_output = np.delete(record_point_output, obj=0, axis=0)  # Delete the initial 
                                                                         # value (Row: 0)
    print(record_label_output)
    drawing_plots([record_point_output[:, 0], record_point_output[:, 1]], record_label_output)

    make_gif(record_point_output, record_label_output, num_div)

if __name__ == "__main__":
    main()