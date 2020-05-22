import os

base_dir = '..\\..'  # relative path to the Fruit-Images-Dataset folder
test_dir = os.path.join(base_dir, 'Test')
train_dir = os.path.join(base_dir, 'Training')

labels = os.listdir(train_dir)

for label in labels:
    train_count = len(os.listdir(os.path.join(train_dir, label)))
    test_count = len(os.listdir(os.path.join(test_dir, label)))
    print("%s & %d & %d \\\\ \\hline" % (label, train_count, test_count))
