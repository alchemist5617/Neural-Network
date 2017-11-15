from sklearn.datasets import load_svmlight_file
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import maxabs_scale, normalize
from sklearn.metrics import confusion_matrix,precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


def get_data():
    data = load_svmlight_file("AID827.svm")
    return data[0], data[1]

x, y = get_data() 
x = maxabs_scale(x)

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=1334)

active_index = (train_y==1)

landa = 1
sum_active= train_x[active_index,:].sum(0)
sum_inactive= train_x[~active_index,:].sum(0)
score = sum_active - sum_inactive

feature_treshold = 1
index = np.array((score>feature_treshold))[0]
train_x = train_x[:,index]
test_x = test_x[:,index]

del x, y

x_size = train_x.shape[1]   

l_size = 320 
z_size = 160               
class_size = 2
pos_weight = 20.00
neg_weight = 0.2

batch_size=380
targets = test_y

hot = np.array([train_y]).reshape(-1)
train_y = np.eye(class_size)[hot.astype(int)]

hot = np.array([test_y]).reshape(-1)
test_y = np.eye(class_size)[hot.astype(int)]

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, x_size],name="x")
    y = tf.placeholder(tf.float32, [None,class_size],name="y")
    learning_rate = tf.placeholder(tf.float32, shape=[],name="learning_rate")
    pkeep = tf.placeholder(tf.float32)

with tf.name_scope("weights"):
    w_1 = tf.Variable(tf.random_normal([x_size, l_size],0, 0.1),name="w_1")
    w_2 = tf.Variable(tf.random_normal([l_size, z_size],0, 0.1),name="w_2")  
    w_3 = tf.Variable(tf.random_normal([z_size, class_size],0, 0.1),name="w_3")
    
with tf.name_scope("biases"):
    b_1 = tf.Variable(tf.random_normal([l_size],0, 0.1),name="b_1") 
    b_2 = tf.Variable(tf.random_normal([z_size],0, 0.1),name="b_2")
    b_3 = tf.Variable(tf.random_normal([class_size],0, 0.1),name="b_3")
    
with tf.name_scope("Input_layer_1"):
    l = tf.nn.relu(tf.matmul(x, w_1)+b_1)
    ld = tf.nn.dropout(l, pkeep)
    
with tf.name_scope("Hidden_layer_1"):
    z = tf.nn.relu(tf.matmul(ld, w_2)+b_2)
    zd = tf.nn.dropout(z, pkeep)
    
with tf.name_scope("Hidden_layer_2"):
    yhat = tf.matmul(z, w_3) + b_3
    yhat1 = tf.nn.sigmoid(tf.matmul(z, w_3) + b_3)

with tf.name_scope("cost"):
    classes_weights = tf.constant([neg_weight, pos_weight])
    cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=y, logits=yhat, pos_weight=classes_weights))
    
with tf.name_scope("train"):
    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

predicted = tf.argmax(yhat, 1)
actual = tf.argmax(y, 1)

tp = tf.count_nonzero(predicted * actual)
fp = tf.count_nonzero(predicted * (actual - 1))
fn = tf.count_nonzero((predicted - 1) * actual)
    
with tf.name_scope("accuracy"):  
    correct_prediction = tf.equal(predicted, actual)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
sess.run(init)

h = train_y.shape[0]-int(train_y.shape[0]/10)
calibration_x = train_x[h:,:]
calibration_y = train_y[h:,:]

rate=0.0001
start_rate=rate
epochs_numbers=30
Decay_factor = 0.8
           
print("Training the Neural Network(UCS) in 30 epochs:\n")
for epoch in range(epochs_numbers): 
    total_batch = int((h-1)/batch_size)
    for i in range(total_batch):
        batch_x=train_x[i:(i+batch_size),:].todense()
        batch_y=train_y[i:(i+batch_size),:]          
        t,c= sess.run([train,cost], feed_dict={x: batch_x, y: batch_y, pkeep: 0.95, learning_rate:rate}) 
        if (i+1) == total_batch:
            test_cost = sess.run(cost, feed_dict={x: test_x.todense(), y: test_y, pkeep: 1.0})
            print("Epoch:", '%02d' % (epoch+1), "Train Cost=", "{:.5f}".format(c),"Test Cost=", "{:.5f}".format(test_cost)) 
    rate*=Decay_factor
    
print("\nTraining Finished!")
print("Running on Test set:")
a,ypred = sess.run([accuracy,yhat1], feed_dict={x: test_x.todense(), y: test_y, pkeep: 1.0})

ycal = sess.run(yhat1, feed_dict={x: calibration_x.todense(), y: calibration_y, pkeep: 1.0})

ypred = normalize(ypred, axis=1, norm='l1')
ycal = normalize(ycal, axis=1, norm='l1')

i = ypred[:,1]>=0.5
ypred_bin = np.zeros(ypred.shape[0])
ypred_bin[i]=1

print("Confusion Matrix:")
print(confusion_matrix(targets,ypred_bin))
p = precision_score(targets,ypred_bin)
r = recall_score(targets,ypred_bin)
f1 = f1_score(targets,ypred_bin)
roc_score = roc_auc_score(targets,ypred_bin)
print("%-40s"%"Test Accuracy: ",'%.2f' % (a))
print("%-40s"%"Test Precision: ",'%.2f' % (p))
print("%-40s"%"Test Recall: ",'%.2f' % (r))
print("%-40s"%"Test F1 Score: ",'%.2f' % (f1))
print("%-40s"%"Test Area Under the Curve(AUC): ",'%.2f' % (roc_score))
print("\n")

calibration_y = np.argmax(calibration_y,axis=1)

cal_index = (calibration_y == 1)
count = np.count_nonzero(cal_index)
p_value_active = np.zeros(ypred.shape[0])
for i in range(0,ypred.shape[0]):
    p_value_active[i] =(np.count_nonzero(ypred[i,1]>=ycal[cal_index,1])/count)
    
cal_index = (calibration_y == 0)
count = np.count_nonzero(cal_index)
p_value_inactive = np.zeros(ypred.shape[0])
for i in range(0,ypred.shape[0]):
    p_value_inactive[i] =(np.count_nonzero(ypred[i,0]>=ycal[cal_index,0])/count)
    
print("Conformal Prediction Results:")
print("%-15s"%"Significance","%-15s"%"A pred A","%-15s"%"I pred A","%-15s"%"I pred I","%-15s"%"A pred I","%-15s"%"Empty","%-15s"%"Uncertain","%-25s"%"Active Error Rate","%-25s"%"Inactive Error Rate\n")
for significance_level in [0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.7,0.8,0.9]:
    cp_active = p_value_active > significance_level
    cp_inactive = p_value_inactive > significance_level
    cp_result = cp_active*2 + cp_inactive

    active_i = cp_result == 2
    inactive_i = cp_result == 1
    empty = np.count_nonzero(cp_result == 0)
    uncertain = np.count_nonzero(cp_result == 3)

    cp_bin = np.zeros(ypred.shape[0])
    cp_bin[active_i] = 1

    active_summary = confusion_matrix(targets,cp_bin)

    cp_bin = np.zeros(ypred.shape[0])
    cp_bin[inactive_i] = 1
    inactive_summary = confusion_matrix(1-targets,cp_bin)

    print('%-15.2f' % (significance_level),'%-15d' % (active_summary[1][1]),'%-15d' % (active_summary[0][1]),'%-15d' % (inactive_summary[1][1]),'%-15d' % (inactive_summary[0][1]),'%-15d' % (empty),'%-15d' % (uncertain),'%-25.2f' % (inactive_summary[0][1]/(inactive_summary[0][1]+active_summary[1][1])),'%-25.2f' % (active_summary[0][1]/(inactive_summary[1][1]+active_summary[0][1])),"\n")
     
sess.close()