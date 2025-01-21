from sklearn import model_selection
from sklearn.metrics import cohen_kappa_score
import numpy as np
import os
import pickle
#  定义字典，便于来解析样本数据集txt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
from model.AccuracyOut import plot_confusion_matrix
from model.OneDCNN import oneDCNN


def Iris_label(s):
    # it = {b'river': 0, b'field': 1, b'buliding': 2}
    it = {b'city': 1, b'water': 2,  b'forest': 4, b'farmland': 3}
    return it[s]

feathernumber = 18
halfnumber = int(feathernumber/2)
attention = False


path = r"..\dataset\OurDataset\test.txt"

classes = ['city', 'water', 'farmland', 'forest']

#  1.读取数据集
data = np.loadtxt(path, dtype=float, delimiter=',', converters={feathernumber: Iris_label})

#  2.划分数据与标签
x, y = np.split(data, indices_or_sections=(feathernumber,), axis=1)  # x为数据，y为标签
x = x[:, 0:feathernumber]  # 选取前个波段作为特征
print(x.shape)
print(y.shape)
# print(y)

c = []
for i in range(0, len(y)):
    # print(i)
    # print(y[i])
   if y[i] == 1:
       c.append([1, 0, 0, 0])
       # print(c[i])
   if y[i] == 2:
        c.append([0, 1, 0, 0])
        # print(c[i])
   if y[i] == 3:
        c.append([0, 0, 1, 0])
        # print(c[i])
   if y[i] == 4:
        c.append([0, 0, 0, 1])
        # print(c[i])
   #  if y[i] == 5:
   #      c.append([0, 0, 0, 0, 1])
       # print(c[i])
c = np.array(c)  # 创建数组
x = np.array(x)  # 创建数组
print(c.shape)
print(x.shape)
train_data, test_data, train_label, test_label = model_selection.train_test_split(x, c, random_state=1,
                                                                                  train_size=0.01, test_size=0.99)
# print(train_data)
# print(train_label)
print(train_data.shape)
print(train_data[0])
# train_data = train_data.swapaxes(0, 1)
# print(train_data.shape)
# print(train_data[0])
# 中间2为特征数/2
train_data = train_data.reshape(-1, halfnumber, 2)
print(train_data.shape)
print(train_data[0])
print(train_label.shape)
# 中间2为特征数/2
test_data = test_data.reshape(-1, halfnumber, 2)
print("test_data.shape", test_data.shape)
print("test_label.shape", test_label.shape)

index = [i for i in range(len(train_data))]
np.random.shuffle(index)
train_data = train_data[index]
train_label = train_label[index]

n_initial = 5
initial_idx = np.random.choice(range(len(train_data)), size=n_initial, replace=False)
train_data_initial, train_label_initial = train_data[initial_idx], train_label[initial_idx]

train_data = np.delete(train_data, initial_idx, axis=0)
train_label = np.delete(train_label, initial_idx, axis=0)

def create_cnn_model():
    """
    创建1D-CNN模型，用于包装成自定义的主动学习方法
    """
    model = oneDCNN(halfnumber=9, isAttention=False).get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def entropy_uncertainty_percent(model, X, percent):
    """
    查询策略：基于熵的不确定性，选择占未标注数据百分比的样本。
    """
    proba = model.predict(X)  # 获取预测概率分布
    entropy = -np.sum(proba * np.log(proba + 1e-10), axis=1)  # 计算熵
    n_queries = max(1, int(len(X) * percent))  # 按百分比计算查询样本数，至少选择1个
    query_indices = np.argsort(entropy)[-n_queries:]  # 按熵值选择样本
    return query_indices, X[query_indices]

def save_model(model, round_num, save_dir="saved_models"):
    """
    保存训练模型及状态
    """
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"model_round_{round_num}.h5")
    model.save(model_path)
    print(f"Model saved to {model_path}")

def active_learning_with_percent(train_data_initial, train_label_initial, test_data, test_label, train_data, train_label,
                                 rounds=10, percent=0.1, save_dir="saved_models"):
    """
    主动学习框架，基于百分比定义每轮查询样本数。
    """
    # 创建并编译模型
    model = create_cnn_model()
    model.fit(train_data_initial, train_label_initial, epochs=10, batch_size=128, verbose=1)

    for round in range(rounds):
        print(f"=== Round {round + 1}/{rounds} ===")

        # 测试模型性能
        predictions = model.predict(test_data)
        test_pred = np.argmax(predictions, axis=1) + 1
        test_true = np.argmax(test_label, axis=1) + 1
        acc = accuracy_score(test_true, test_pred)
        print(f"Test Accuracy: {acc:.4f}")

        # 查询未标注数据中的高不确定性样本
        if len(train_data) == 0:
            print("No more unlabeled data available.")
            break

        query_idx, query_instance = entropy_uncertainty_percent(model, train_data, percent)
        print(f"Round {round + 1}: Queried {len(query_idx)} samples ({percent * 100:.1f}% of unlabeled data).")

        # 将查询的样本加入训练集
        model.fit(train_data[query_idx], train_label[query_idx], epochs=5, batch_size=128, verbose=0)

        # 移除已查询的样本
        train_data = np.delete(train_data, query_idx, axis=0)
        train_label = np.delete(train_label, query_idx, axis=0)

        # 输出分类报告和混淆矩阵
        print(classification_report(test_true, test_pred, target_names=classes))
        cm = confusion_matrix(test_true, test_pred)
        plot_confusion_matrix(classes, cm, f'Confusion_Matrix_Round{round + 1}.png',
                              title=f'Confusion matrix Round {round + 1}')

        # 保存模型和学习器状态
        save_model(model, round + 1, save_dir)

    return model

active_learning_with_percent(train_data_initial, train_label_initial, test_data, test_label, train_data, train_label,
                             rounds=10, percent=0.1, save_dir="saved_models")



