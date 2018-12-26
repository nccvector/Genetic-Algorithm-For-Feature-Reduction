# This code is not optimized and can be improved
# Give feedback if you find a bug or want more optimizations.
# I did not have enough time to optimize and test it thoroughly

# You can change the classifier from neural network to KNN, LDA, SVM or any other
# I have done it with KNN already, which was very fast, but decided to redo it with 
# Neural network just to feed my curiosity

import numpy as np
from numpy import genfromtxt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as Data
import random

# Genetic Algorithm parameters
population_size = 2
itterations = 2
chromosome_size = 80
mutation_chance = 0.05

# Classifier parameters
input_size = 80 # <---Dont change this

# You can change the parameters below
hidden_size1 = 30
hidden_size2 = 30
num_classes = 7
num_epochs = 10
learning_rate = 0.01
test_data_percentage = 0.5

### Preparing Data ###
imported_data_and_labels = genfromtxt('emg_all_features_labeled.csv', delimiter=',')
samples = len(imported_data_and_labels[:,0])
batch_size = int(samples/num_epochs)

np.random.shuffle(imported_data_and_labels) # Shuffling data

imported_data = imported_data_and_labels[:,0:input_size]
labels = imported_data_and_labels[:,input_size]

# Scaling features between 0 and 1
for i in range(input_size):
    imported_data[:,i] = abs(imported_data[:,i])
    max_val = max(imported_data[:,i])
    if max_val >= 1:
        imported_data[:,i] = imported_data[:,i]/max_val

# Converting labels to One-Hot enconding
labels_one_hot = []
for l in labels:
    temp = np.zeros(num_classes)
    temp[int(l-1)] = 1
    labels_one_hot.append(temp)
labels = np.asarray(labels_one_hot, dtype=np.float32)

# Function for returning features based on chromosome
def features_from_chromosome(chromosome):
    chromosome = np.asarray(chromosome)
    ind = np.flatnonzero(chromosome)
    new_data = imported_data[:,ind]
    return new_data

# Defining Neural Network Class
class Net(nn.Module):
        def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
                super(Net, self).__init__()
                self.fc1 = nn.Sequential(
                        nn.Linear(input_size,hidden_size1),
                        nn.ReLU()
                )
                self.fc2 = nn.Sequential(
                        nn.Linear(hidden_size1,hidden_size2),
                        nn.ReLU()
                )
                self.fc3 = nn.Sequential(
                        nn.Linear(hidden_size2,num_classes)
                )
        # Forward prop
        def forward(self, x):
                out = self.fc1(x)
                out = self.fc2(out)
                out = self.fc3(out)
                return out

# Model Train function
def train_model(model, criterion, optimizer, num_epochs):
    best_val_acc = 0.0
    best_train_acc = 0.0
    best_model_wts = model.state_dict()
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for data in dataloders[phase]:
                # get the inputs
                inputs, label = data
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(label.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(label)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                outputs = model(inputs)
                preds = torch.max(outputs, 1)[1]
                loss = criterion(outputs, torch.max(labels,1)[1])
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.item()
                for ind in range(len(preds)):
                    if preds[ind] == torch.max(label,1)[1][ind]:
                        running_corrects += 1
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            # deep copy the model
            if phase == 'val' and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_model_wts = model.state_dict()
            if phase == 'train' and epoch_acc > best_train_acc:
                best_train_acc = epoch_acc
                best_model_wts = model.state_dict()
    return model, best_train_acc, best_val_acc


# Genetic ALgorithm begins...
# Initial Random population
Gen = []
for p in range(population_size):
    temp = []
    for c in range(chromosome_size):
        temp.append(random.randint(0,1))
    Gen.append(temp)

# Itterations begin...
for itt in range(itterations):

    print('-' * 10)
    print('-' * 10)
    print('Itteration Number:',itt+1)
    f = open("Iterations.txt", "w+")
    best_model_wts = None
    max_fitness = 0

    # Fittness Calculation
    fitness = []
    for p in range(population_size):
        print('Gen {}/{}'.format(p+1,population_size))
        # Loading data from chromosome
        data = features_from_chromosome(Gen[p])
        
        # Splitting test and train data
        test_data = data[0:int(test_data_percentage*samples),:] # test:  first 30 percent
        train_data = data[int(test_data_percentage*samples):,:] # train:  remaining data
        test_labels = labels[0:int(test_data_percentage*samples),:] # test:  first 30 percent
        train_labels = labels[int(test_data_percentage*samples):,:] # train:  remaining data

        train_dataset = Data.TensorDataset(
            torch.from_numpy(train_data).float(),
            torch.from_numpy(train_labels).long())

        test_dataset = Data.TensorDataset(
            torch.from_numpy(test_data).float(),
            torch.from_numpy(test_labels).long())

        # Data Loader (Input Pipeline)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=batch_size,
                                                shuffle=False)

        #Make a dictionary defining training and validation sets
        dataloders = dict()
        dataloders['train'] = train_loader
        dataloders['val'] = test_loader
        dataset_sizes = {'train': int(samples*(1-test_data_percentage)), 'val': int(samples*test_data_percentage)}

        # Dictionary defining training and validation sets
        use_gpu = torch.cuda.is_available()

        # Defining net object
        input_size = len(data[0,:])
        net = Net(input_size, hidden_size1, hidden_size2, num_classes)

        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

        if use_gpu:
            model_ft, train_acc, test_acc = train_model(net.cuda(), criterion, optimizer, num_epochs)
        else:
            model_ft, train_acc, test_acc = train_model(net, criterion, optimizer, num_epochs)

        fitness.append((test_acc * 10)/(sum(Gen[p]) * 0.0001))

        if test_acc > max_fitness:
            max_fitness = test_acc
            best_model_wts = model_ft

    print('Best Accuracy: {:.4f}'.format(max_fitness))
    f.write('Best Accuracy: {:.4f}\n'.format(max_fitness))

    # Cummulative probability
    total = sum(fitness)
    cum_prob = []
    sum_prob = 0
    for i in range(0,population_size):
        fitness[i] = fitness[i]/total
        sum_prob += fitness[i]
        cum_prob.append(sum_prob)

    # Generating new population
    new_gen = []
    for i in range(0,population_size):
        # Selecting Parents
        r = random.random()
        for x in range(0,population_size):
            if r <= cum_prob[x]:
                parent_A = Gen[x]
                break
        r = random.random()
        for x in range(0,population_size):
            if r <= cum_prob[x]:
                parent_B = Gen[x]
                if parent_B == parent_A:
                    parent_B = Gen[random.randint(0,population_size-1)]
                break
        # Crossover
        child = []
        for x in range(0,chromosome_size):
            r = random.random()
            if r >= 0.5:
                child.append(parent_A[x])
            else:
                child.append(parent_B[x])
        # Mutation
        if random.random() <= mutation_chance:
            bit = random.randint(0,chromosome_size - 1)
            if child[bit] == 0:
                child[bit] = 1
            else:
                child[bit] = 0

        # Adding to generation
        new_gen.append(child)
    
    # Printing prev-gen and new gen to console and writing to file
    print('-' * 10)
    print('-' * 10)
    print("PREV-GEN:")
    PREV_GEN_WITH_FIT = str([Gen,fitness])
    print(PREV_GEN_WITH_FIT)
    print('-' * 10)
    print('-' * 10)

    f.write('-' * 10)
    f.write('-' * 10)
    f.write('PREV-GEN:')
    f.write(PREV_GEN_WITH_FIT)
    f.write('-' * 10)
    f.write('-' * 10)

    Gen = new_gen

    print('-' * 10)
    print('-' * 10)
    print("NEW GEN :")
    print(Gen)
    print('-' * 10)
    print('-' * 10)

    print('-' * 40)
    print('-' * 40)

    f.write('-' * 10)
    f.write('-' * 10)
    f.write("NEW GEN :")
    f.write(str(Gen))
    f.write('-' * 10)
    f.write('-' * 10)

    f.write('-' * 40)
    f.write('-' * 40)

    f.close()

# Saving model
torch.save(best_model_wts.state_dict(), 'save.pkl')