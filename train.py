import numpy as np

#word vector has a sense about the contexts
wordVectors = np.load('wordVectors.npy')

#loard the vectorized prepared dataset (from preprocess.py)
ids = np.load('idsMatrix2.npy')

maxSeqLength = 750 #number of words in an training data entry

numDimensions = 300

batchSize = 24 # number of samples that will be taken at a time to train the model. Then it require less memory
lstmUnits = 64 # number of LSTM cells.  each LSTM cell conssit of 4 gates by design
numClasses = 2
iterations = 100000 #number of iterations in nural network training (each iteration contain one foward and backward iteration)


from random import randint

def getTrainBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength]) #get zero filled array with the size of batchSize
    for i in range(batchSize):
        if (i % 2 == 0):
            num = randint(1,979)
            labels.append([1,0])
        else:
            num = randint(1019,1999)
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels

def getTestBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(979,1019)
        if (num <= 999):
            labels.append([1,0])
        else:
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels







import tensorflow as tf
tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])



data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)


lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)



weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)





correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))



loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)



import datetime

tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"

with tf.Session() as sess:
    writer = tf.summary.FileWriter(logdir, sess.graph)


#--------------------------------TRAINING ----------------------
''' 
sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

for i in range(iterations):
   #Next Batch of reviews
    nextBatch, nextBatchLabels = getTrainBatch();
    sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels});
    print(i);

    #Write summary to Tensorboard
    if (i % 50 == 0):
        summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
        writer.add_summary(summary, i)

    #Save the network every 10,000 training iterations
    if (i % 1000 == 0):
        save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
        print("saved to %s" % save_path)
writer.close()
'''




sess = tf.InteractiveSession()
saver = tf.train.Saver()

saver = tf.train.import_meta_graph('models/pretrained_lstm.ckpt-60000.meta')
saver.restore(sess,tf.train.latest_checkpoint('models'))

# Read the stored model
graph = tf.get_default_graph()
print([op.name for op in graph.get_operations()])

iterations = 10
for i in range(iterations):
    nextBatch, nextBatchLabels = getTestBatch();
    print("Accuracy for this batch:", (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100)
