from keras.models import Sequential, load_model
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import random
import sign_name

# 载入验证集数据
validation_file = "./Data/valid.p"
with open(validation_file, mode="rb") as f:
    valid = pickle.load(f)

valid_input, label = valid["features"], valid["labels"]
valid_input_flatten = valid_input.reshape([4410, 32 * 32 * 3]) / 255

# 载入之前已训练好的模型
model = load_model('my_model.h5')

# 进行预测
index_random = random.sample(range(0, 4410), 10)  # random.sample()生成不相同的随机数
for i in range(10):
    print("\n*************************************************************************")
    print("\n================ 原始图像 ==================")
    plt.imshow(valid_input[index_random[i]])
    plt.show()
    print("=============================================")
    
    pre_input = valid_input_flatten[index_random[i]].reshape([1, 32 * 32 * 3])
    pre = model.predict_classes(pre_input)  # 预测
    
    print("\n=============== 预测结果 ===================")
    print("结果:")
    # 预测正确时
    if label[index_random[i]]==pre[0]:
        print("预测正确!")
        print("标志含义：")
        print(sign_name.sign_name_dic[str(label[index_random[i]])])

    # 预测错误时
    else:
        print("预测错误!")
        print("预测标志含义：")
        print(sign_name.sign_name_dic[str(pre[0])])
        print("实际标志含义：")
        print(sign_name.sign_name_dic[str(label[index_random[i]])])

print("\n*************************************************************************")
