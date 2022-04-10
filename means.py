import numpy as np
import matplotlib.pyplot as plt
import math


# Инициализация данных
train = np.array([[3, 4], [3, 6], [7, 3], [4, 7], [3, 8], [8, 5], [4, 5], [4, 1], [7, 4], [5, 5]])

# Первоначальный выбор центра кластера (4,5), (5,5)
# center = np.array([[4, 5], [5, 5]], dtype=float)
center = np.array([[4, 8], [7, 4]], dtype=float)

# Инициализация изображения
def initgraph(flag, train):
    plt.scatter(train[:, 0], train[:, 1], color='b')
    if flag:
        plt.scatter(center[:, 0], center[:, 1], color='r')
    plt.grid()
    plt.xlabel("x")
    if flag:
        plt.title("Изображение с центрами", fontsize=16)
    else:
        plt.title("Исходное изображение", fontsize=16)
    plt.xticks(np.arange(0, 10, 1))
    plt.ylabel("y")
    plt.yticks(np.arange(0, 10, 1))
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.show()


# Использование евклидово расстояния
def kmeans(train, center, title):
    # Сохранение кластеризации каждой итерации
    list_d1 = []
    list_d2 = []
    for i in range(train.shape[0]):
        # Евклидово расстояние
        d1 = math.sqrt(math.pow((train[i, 0] - center[0, 0]), 2) + math.pow((train[i, 1] - center[0, 1]), 2))
        d2 = math.sqrt(math.pow((train[i, 0] - center[1, 0]), 2) + math.pow((train[i, 1] - center[1, 1]), 2))
        # Манхэттенское расстояние
        # d1 = (abs(train[i, 0] - center[0, 0])) + abs((train[i, 1] - center[0, 1]))
        # d2 = (abs(train[i, 0] - center[1, 0])) + abs((train[i, 1] - center[1, 1]))
        if d1 < d2:
            list_d1.append(i)
        else:
            list_d2.append(i)
    update(list_d1, list_d2, train, title)


# Итеративное обновление изображения
def update(list_d1, list_d2, train,title):
    center1 = np.empty([1, 2])
    center2 = np.empty([1, 2])
    for index in range(len(list_d1)):
        plt.scatter(train[list_d1[index], 0], train[list_d1[index], 1], color='g')
        train1 = np.array([train[list_d1[index], 0], train[list_d1[index], 1]])
        if index == 0:
            center1[index] = train1
        else:
            center1 = np.insert(center1, index, train1, axis=0)
    for index in range(len(list_d2)):
        plt.scatter(train[list_d2[index], 0], train[list_d2[index], 1], color='y')
        train2 = np.array([[train[list_d2[index], 0], train[list_d2[index], 1]]])
        if index == 0:
            center2[index] = train2
        else:
            center2 = np.insert(center2, index, train2, axis=0)
    # Определение новой центральной точки
    plt.scatter(np.average(center1[:, 0]), np.average(center1[:, 1]), color='r')
    plt.scatter(np.average(center2[:, 0]), np.average(center2[:, 1]), color='r')
    center[0] = np.array([[np.average(center1[:, 0]), np.average(center1[:, 1])]])
    center[1] = np.array([[np.average(center2[:, 0]), np.average(center2[:, 1])]])
    plt.grid()
    plt.xlabel("x")
    plt.title(" " + str(title) + ". " + "Cхема результатов кластеризации",
             fontsize=16)
    title += 1
    plt.xticks(np.arange(0, 10, 1))
    plt.ylabel("y")
    plt.yticks(np.arange(0, 10, 1))
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.show()


def show_img():
    img = plt.imread('/Users/a19796202/k_means/points.png')
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(type(img))
    fig.set_figwidth(10)
    fig.set_figheight(5)
    plt.title("Исходные точки", fontsize=16) 
    plt.show()


if __name__ == '__main__':
    show_img()
    initgraph(0, train)
    initgraph(1, train)
    # Выполнение двух кластеризациий, результаты показаны на рисунке
    title = 1
    for i in range(2):
        kmeans(train, center,title)
        title += 1